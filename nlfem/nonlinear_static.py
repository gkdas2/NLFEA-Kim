import numpy as np

from nlfem.shapefunctions import plset, itgzone
from nlfem.elastic import elast3d
from nlfem.hyperelastic import hyper3d

from nlfem.util import prout, output

# ----------------------------------------------------------------------------


def nlfea(itra, tol, atol, ntol, tims, nout, MID, prop, extforce, sdispt, xyz, le):
    r"""Main program for Hyperelastic/elastoplastic analysis.

    Inputs:
    itra: Max number of convergence iterations in the NR method. Bisection is invoked if itra is reached.
    tol: If the norm of the residual < tol, NR iteration is consiered converged.
    atol: The NR solution is considered diverging if the residual error > atol and bisection is invoked.
    ntol: Maximum number of consecutive bisections to perform before giving up.
    tims: Array to define load steps. Each row represents a load step. [[Start, End, Increment, InitialFactor, FinalFactor], ...]. For example, TIMS = [[0.0, 1.0, 0.1, 0.0, 0.5]] has 10 increments during which the load increases from 0 to 50% of the total load. The end time of previous load step and the start time of the following load step must be same. Same is true for the load factors.
    nout: Handle to file in which the nodal displacements and stresses at integration points are printed at the end of each load increment.
    MID: Material ID. Linear elastic = 0. Mooney-Rivlin hyperelasticity = -1, and infinitesimal elastoplasticity = 1.
    prop: Material properties. For linear elastic, prop = [lambda, mu] defines the two Lame's constants. For hyperleasticity, prop = [A10, A01, D]. For elastoplasticity, prop = [lambda, mu, hardening type ($\beta$), plastic modulis (H), initial yield strength ($Y_0$)]
    extforce: Array of applied external forces. [[node, DOF, value],...].
    sdispt: Array of prescribed displacements in order to prevent rigid motion. [[node, DOF, value],...].
    xyz: Array of nodal coordinates. [nnodes, 3].
    le: Array of conenctivity information for hexahedral elements. [nelements, 8].
    """

    """
    original global parameters
    gkf: [neq, neq]. Tangent matrix
    force: [neq, 1]. Residual vector
    disptd: [neq, 1]. Displacement vector
    dispdd: [neq]. Displacement increment.
    sigma: [6, 8, ne]: Stress at each integration points
    xq: [7, 8, ne]: History variable at each integration points
    """
    # -- open file to store output.
    outfile = open(nout, "w")

    numnp, ndof = np.shape(xyz)
    ne = np.shape(le)[0]
    neq = ndof * numnp
    disptd = np.zeros(neq)  # Nodal displacement
    dispdd = np.zeros(neq)  # Nodal increment

    # -- Initialize material properties
    # if MID >= 0:
    etan, sigma, xq = plset(prop, MID, ne)

    # -- Check element connectivity for bad elements
    _ = itgzone(xyz, le, nout)

    # -- Load increments [Start, End, Increment, InitialLoad, FinalLoad]
    nload = np.shape(tims)[0]
    iload = 1  # First load increment
    timef = tims[iload - 1, 0]  # Starting time
    timei = tims[iload - 1, 1]  # Ending time
    delta = tims[iload - 1, 2]  # Time increment
    cur1 = tims[iload - 1, 3]  # Start load factor
    cur2 = tims[iload - 1, 4]  # End load factor
    delta0 = delta  # Saved time increment
    time = timef  # starting time
    tdelta = timei - timef  # Total time interval for load step
    itol = 1  # Bisection level
    tary = np.zeros(ntol)  # Time stamps for bisections

    # ----------------------------------------------------------------------------
    # -- Load increment Loop
    istep = -1
    # First loop [10] is for load steps and load increments.
    # The loop has NLOAD load steps.Each load step is composed of multiple load increments. See Section 2.2.4 load increment force method.
    # The total load applied is divided by number of increments
    FLAG10 = 1
    while FLAG10 == 1:  # Previous solution has converged
        FLAG10 = 0
        # -- Store previously converged displacement for the sake of bisection
        cdisp = disptd.copy()

        if itol == 1:  # No bisection
            delta = delta0
            tary[itol - 1] = time + delta
        else:  # Recover previous bisection
            itol -= 1  # Reduce bisection level
            delta = tary[itol - 1] - tary[itol]  # New time increment
            tary[itol] = 0  # Empty converged bisection level
            istep = istep - 1  # Decrease load increment

        time0 = time  # Save current time

        # -- Update stress and history variables
        UPDATE = True
        LTAN = False
        gkf = np.zeros((neq, neq))  # make it sparse
        force = np.zeros(neq)

        if MID == 0:
            force, gkf, sigma = elast3d(
                etan, UPDATE, LTAN, ne, ndof, xyz, le, disptd, force, gkf, sigma
            )
        elif MID > 0:
            raise NotImplementedError("Plast3d not available")
        elif MID < 0:
            force, gkf, sigma = hyper3d(
                prop, UPDATE, LTAN, ne, ndof, xyz, le, disptd, force, gkf, sigma
            )
        else:
            raise NotImplementedError("Wrong material")

        # -- Print Results
        if istep >= 0:
            prout(outfile, time, numnp, ne, ndof, sigma, disptd)

        # ---------------------------------------------------------------------------
        # -- Begin adding load steps (increase in time)
        time += delta
        istep += 1

        # Check time and control bisection
        # Second loop [11] is for bisection, if convergence cannot be obtained for current load.
        # If convergence iteration fails, load increment is halved. Then loop 11 is repeated from previously converged point.
        # There is a maximum number of bisection that can be done before giving up.
        # For bisection, the previously converged displacement is stored in CDISP.
        # This is because the displacement field DISPTD is updated on the go, and only reverted back using previous value in CDISP if bisection is needed.
        FLAG11 = 1
        while FLAG11:  # Bisection loop starts
            FLAG11 = 0
            if (time - timei) > 1e-10:  # Time passed the end time
                if (timei + delta - time) > 1e-10:  # One more at the end time
                    delta = timei + delta - time  # Time increment to end
                    delta0 = delta  # Saved time increment
                    time = timei  # current time is the end
                else:
                    iload += 1  # Progress to next step
                    if iload > nload:  # Finished final load step
                        FLAG10 = 0  # Stop the program
                        break
                    else:  # Next load step
                        time -= delta
                        delta = tims[iload - 1, 2]
                        delta0 = delta
                        time = time + delta
                        timef = tims[iload - 1, 0]
                        timei = tims[iload - 1, 1]
                        tdelta = timei - timef
                        cur1 = tims[iload - 1, 3]
                        cur2 = tims[iload - 1, 4]

            # -- Load factor and prescribed displacements
            factor = cur1 + (time - timef) / tdelta * (cur2 - cur1)
            sdisp = delta * sdispt[:, 2] / tdelta * (cur2 - cur1)

            # -- Start NR convergence iteration
            # ------------------------------------------
            iter = 0
            dispdd = np.zeros(neq)
            # This third loop [20] is for NR convergence iteration.
            # The major part of this loop is devoted to calculating the residual vector FORCE and tangent matrix GKF.
            # If residual < threshold, iteration has converged. In such case, loop ends, and procedure moves to next load increment.
            #
            # If none of the iteration converges, invoke bisection: half the load increment, and try again with the NR process.
            FLAG20 = 1
            while FLAG20:
                FLAG20 = 0
                iter += 1
                # Initialize global stiffness and residual vector F
                gkf = np.zeros((neq, neq))  # make it sparse
                force = np.zeros(neq)
                # Assemble K and F
                UPDATE = False
                LTAN = True

                if MID == 0:
                    force, gkf, sigma = elast3d(
                        etan, UPDATE, LTAN, ne, ndof, xyz, le, disptd, force, gkf, sigma
                    )
                elif MID > 0:
                    raise NotImplementedError("implement plast3d")
                elif MID < 0:
                    force, gkf, sigma = hyper3d(
                        prop, UPDATE, LTAN, ne, ndof, xyz, le, disptd, force, gkf, sigma
                    )

                # -- Increase external force if exforce is not empty. First identify the correct global dof at which these loads are applied,i.e, ndof*node + dof.
                # -- Then add the current load increment to the position.
                if extforce.size > 0:
                    loc = ndof * (extforce[:, 0] - 1) + (extforce[:, 1] - 1)
                    # Make sure the indices are integers. Look for better implementation.
                    loc = np.array([int(iloc) for iloc in loc], dtype=np.int64)
                    force[loc] += factor * extforce[:, 2]

                # -- Prescribed displacement BC
                ndisp = np.shape(sdispt)[0]
                if ndisp:
                    fixeddof = ndof * (sdispt[:, 0] - 1) + sdispt[:, 1] - 1
                    fixeddof = np.array([int(ifixed) for ifixed in fixeddof])
                    gkf[fixeddof] = np.zeros([ndisp, neq])

                    Ieye = np.eye(ndisp)
                    for i in range(len(fixeddof)):
                        for j in range(len(fixeddof)):
                            gkf[fixeddof[i], fixeddof[j]] = prop[0] * Ieye[i, j]

                    force[fixeddof] = 0
                    if iter == 1:
                        force[fixeddof] = prop[0] * sdisp[:]

                # -- Check convergence
                if iter > 1:
                    fixeddof = ndof * (sdispt[:, 0] - 1) + sdispt[:, 1] - 1
                    alldof = np.arange(0, neq, dtype=np.int64)
                    freedof = np.setdiff1d(alldof, fixeddof)
                    resn = np.max(np.abs(force[freedof]))
                    output(1, iter, resn, time, delta)

                    if resn < tol:
                        FLAG10 = 1
                        break

                    if (resn > atol) or (iter >= itra):  # Start bisection
                        itol += 1
                        if itol <= ntol:
                            delta = delta / 2
                            time = time0 + delta
                            tary[itol - 1] = time
                            disptd = cdisp.copy()
                            print(
                                rf"Not converged. Bisecting load increment {itol}")
                        else:
                            raise RuntimeError(
                                "Maximum number of bisection reached without convergence. Terminating now..."
                            )

                        # -- If here, no runtime error raised. Proceed for the NR iterations
                        FLAG11 = 1
                        FLAG20 = 1
                        break

                # -----------------
                # -- Solve the system of equation
                if FLAG11 == 0:
                    soln = np.linalg.solve(gkf, force)
                    dispdd += soln
                    disptd += soln
                    FLAG20 = 1
                else:
                    FLAG20 = 0

                if FLAG10 == 1:
                    break
            # 20 Convergence iteration end
        # 11 Bisection interation end
    # 10 Load increment iteration end
    print(rf" *** Successful end of program *** ")
    outfile.close


if __name__ == "__main__":
    XYZ = np.array(
        [
            [0.0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ]
    )

    # Element connectivity
    LE = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])

    # External forces [Node, DOF, Value]
    EXTFORCE = np.array([[5, 3, 10e3], [6, 3, 10e3],
                        [7, 3, 10e3], [8, 3, 10e3]])

    # Prescribed displacement [Node, DOF, Value]
    SDISPT = np.array(
        [
            [1, 1, 0.0],
            [1, 2, 0],
            [1, 3, 0],
            [2, 2, 0],
            [2, 3, 0],
            [3, 3, 0],
            [4, 1, 0],
            [4, 3, 0],
        ]
    )

    # Material Properties
    # MID = 0(Linear elastic) PROP: [lambda, mu]
    MID = 0
    E = 2e11
    NU = 0.3
    LAMBDA = E * NU / ((1 + NU) * (1 - 2 * NU))
    MU = E / (2 * (1 + NU))
    PROP = np.array([1.1538e6, 7.6923e5])

    # Load increments [Start, End, Increment, InitialFactor, FinalFactor]
    TIMS = np.array([[0, 0.5, 0.1, 0, 0.5], [0.5, 1.0, 0.1, 0.5, 1]])

    # Set program parameter
    ITRA = 30
    ATOL = 1e5
    NTOL = 6
    TOL = 1e-6

    # Call main function
    NOUT = "output.out"
    out = nlfea(ITRA, TOL, ATOL, NTOL, TIMS, NOUT,
                MID, PROP, EXTFORCE, SDISPT, XYZ, LE)
# -------------------------------------------------------------------------
