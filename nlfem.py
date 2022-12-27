import numpy as np
from scipy import sparse


# -------------------------------------------------------------------------
def shapel(xi, elxy):
    """ Compute shape function, derivatives, and determinant of hex element. 

    Inputs: 
            xi: (xi, eta, zi) in the reference coordinates is the location where shape function and derivative are to be evaluated.
            elxy: 8x3 matrix. It contains nodal coordinates of eight nodes of the element.

    Returns:
            sf: 8x1 array. Shape functions
            gdsf: 3x8 matrix. Derivatives of the shape function, dN_dx
            det: Scalar. Determinant of the Jacobian of the mapping.            
    """
    # Isoparametric element coordinates
    xnode = np.array([[-1,  1,  1, -1, -1,  1,  1, -1],
                      [-1, -1,  1,  1, -1, -1,  1,  1],
                      [-1, -1, -1, -1,  1,  1,  1,  1]])
    quar = 0.125
    # Shape functions
    sf = np.zeros(8)
    # Derivative of shape function
    dsf = np.zeros((3, 8))

    for i in range(8):
        xp, yp, zp = xnode[:, i]    # xi_i, eta_i, zi_i
        # isoparametric shape functons. Eqn 1.136
        xi0 = np.array([1+xi[0]*xp, 1+xi[1]*yp, 1+xi[2]*zp])
        sf[i] = quar*xi0[0]*xi0[1]*xi0[2]   # N(xi_)
        dsf[0, i] = quar*xp*xi0[1]*xi0[2]   # dN_dxi
        dsf[1, i] = quar*yp*xi0[0]*xi0[2]   # dN_deta
        dsf[2, i] = quar*zp*xi0[0]*xi0[1]   # dN_dzi

    gj = dsf@elxy   # jacobian dx_dxi
    det = np.linalg.det(gj)
    gjinv = np.linalg.inv(gj)
    gdsf = gjinv@dsf  # DN_dx Eqn 1.138
    return sf, gdsf, det
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
def elast3d(etan, UPDATE, LTAN, ne, ndof, xyz, le, disptd, force, gkf, sigma):
    """
    Main program computing global stiffness residual force for linear elastic material model

    Inputs:
        etan: Size: [6,6]. Elastic stiffness matrix D for voigt notation. Eq 1.81
        UPDATE: Logical. True: save stress value
        LTAN: logical. True: calculate global stiffness matrix
        ne: integer. Total number of elements
        ndof: integer. Dimension of problem (3)
        xyz: [nnode, 3]. Coordinates of all nodes
        le: [ne, 8]. Element connectivity   
        sigma: stress at each integration point (updated)
        force: residual force array. 
    Calls:
        shapel() at each integration points    
    """
    # Integration points and weights
    xg = np.array([-0.57735026918, 0.57735026918])
    wgt = np.array([1.0, 1.0])
    # Stress storage index (No of integration points)
    intn = 0

    # -- Loop over elements to compute K and F
    for ie in np.arange(1, ne+1):
        # Nodal coordinates and incremental displacements
        elxy = xyz[le[ie-1]-1]
        #print(rf"elxy elast3d: {elxy}")
        # Local to global mapping
        # Each entry in idof corresponds to its actual number in global dofs
        idof = np.zeros(24, dtype=np.int32)
        for i, v in enumerate(le[ie-1]-1):
            idof[ndof*i] = ndof*v
            idof[ndof*i + 1] = ndof*v + 1
            idof[ndof*i + 2] = ndof*v + 2
        #print(rf"idof: {idof}")
        dsp = disptd[idof].reshape(ndof, 8, order='F')
        #print(rf"dsp elast3d: {dsp}")
        # Loop over integration points
        for lx in np.arange(2):
            for ly in np.arange(2):
                for lz in np.arange(2):
                    e1 = xg[lx]
                    e2 = xg[ly]
                    e3 = xg[lz]
                    intn += 1
                    # Determinant and shape function derivative
                    _, shpd, det = shapel(np.array([e1, e2, e3]), elxy)
                    #print(rf"shpd elast3d: {shpd}")
                    fac = wgt[lx]*wgt[ly]*wgt[lz]*det
                    # Strain
                    deps = dsp@shpd.T  # gradient of u, r=eqn 1.139
                    # Strain in Voight notation (note y12 = 2e12)
                    ddeps = np.array([deps[0, 0], deps[1, 1], deps[2, 2], deps[0, 1] +
                                     deps[1, 0], deps[1, 2]+deps[2, 1], deps[0, 2]+deps[2, 0]])
                    #print(rf"ddeps elast3d: {ddeps}")
                    # Stress
                    stress = etan@ddeps  # stress = D*strain using voigt notation

                    # Update stress(store stress in this big global array sigma, that acculumates stresses at all integration points) for all elements
                    if UPDATE:
                        sigma[:, intn-1] = stress.copy()

                    # Add residual force and stiffness matrix
                    # Assemble the B matrix = dN_dx_. Eqn 1.140
                    bm = np.zeros((6, 24))
                    for i in range(8):
                        col = np.arange(i*ndof,   i*ndof + ndof)
                        bm[:, col] = np.array([[shpd[0, i], 0, 0],
                                               [0, shpd[1, i], 0],
                                               [0, 0, shpd[2, i]],
                                               [shpd[1, i], shpd[0, i], 0],
                                               [0, shpd[2, i], shpd[1, i]],
                                               [shpd[2, i], 0, shpd[0, i]]])
                    # Residual force
                    # internal force  = integral(B.T * sigma )
                    force[idof] = force[idof] - fac*bm.T@stress

                    # Compute element tangent stiffness and iteratively add to global tangent stiffness matrix
                    if LTAN:
                        ekf = bm.T@etan@bm  # ke_intergationPoint  = B.T*D*B
                        for i in range(24):
                            for j in range(24):
                                gkf[idof[i], idof[j]] += fac*ekf[i, j]
    return force, gkf, sigma
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
def plset(prop, MID, ne):
    """
    Initialize history variables and elastic stiffness matrix.
    XQ : 1-6 = Back stress alpha, 7 = Effective plastic strain
    SIGMA : Stress for rate-form plasticity
            Left Cauchy-Green tensor XB for multiplicative plasticity
            ETAN : Elastic stiffness matrix
    """
    lam = prop[0]
    mu = prop[1]
    n = 8*ne    # 8 integration points per element

    if MID > 30:
        sigma = np.zeros((12, n))
        xq = np.zeros((4, n))
        sigma[6:9, :] = 1
        etan = np.array([[lam + 2*mu, lam, lam],
                         [lam, lam+2*mu, lam],
                         [lam, lam, lam+2*mu]])
    else:
        sigma = np.zeros((6, n))
        xq = np.zeros((7, n))
        etan = np.array([
                        [lam+2*mu, lam, lam, 0, 0, 0],
                        [lam, lam+2*mu, lam, 0, 0, 0],
                        [lam, lam, lam+2*mu, 0, 0, 0],
                        [0, 0, 0, mu, 0, 0],
                        [0, 0, 0, 0, mu, 0],
                        [0, 0, 0, 0, 0, mu]
                        ])
    return etan, sigma, xq
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
def itgzone(xyz, le, nout):
    """
    Check element connectivity and calculate total volume

    Returns: 
        Volume: total volume of the mesh
    """
    eps = 1e-7
    ne = np.shape(le)[0]
    volume = 0

    for i in range(1, ne + 1):
        elxy = xyz[le[i-1]-1]
        _, _, det = shapel(np.array([0, 0, 0]), elxy)
        dvol = 8*det
        if dvol < eps:
            print(f"Negative Jacobian in element {i}")
            print(f"Node coordinates are {elxy}")
            raise ValueError(" Negative Jacobian")
        volume += dvol
    return volume
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
def mooney(F, A10, A01, K, LTAN):
    """ 
    Calculate 2nd PK stress and material stiffness for Mooney-Rivlin hyperelastic material.

    Inputs:
            F : Deformation gradient [3,3]
            A10, A01, K : Material constants for Mooney-Rivlin Material model
            LTAN : 0=Calculate stress alone; 1=Calcualte stress+material stiffness

    Outputs:
            Stress : PK II stress [S11, S22, S33, S12, S23, S13]
            D : Material Stiffness [6,6]          
    """
    # Some constants
    x12 = 1/2
    x13 = 1/3
    x23 = 2/3
    x43 = 4/3
    x53 = 5/3
    x89 = 8/9

    # Calculate Right Cauchy Green Deformation tensor and translate to Voigt notation
    C = F.T @ F
    C1 = C[0, 0]
    C2 = C[1, 1]
    C3 = C[2, 2]
    C4 = C[0, 1]
    C5 = C[1, 2]
    C6 = C[0, 2]

    # Calcualte Invariants
    I1 = C1 + C2 + C3
    I2 = C1*C2 + C1*C3 + C2*C3 - C4*C4 - C5*C5 - C6*C6
    I3 = np.linalg.det(C)

    # Calculate Derivatives of I wrt E
    I1E = 2 * np.array([1, 1, 1, 0, 0, 0])
    I2E = 2 * np.array([C2+C3, C3+C1, C1+C2, -C4, -C5, -C6])
    I3E = 2 * np.array([C2*C3-C5*C5, C1*C3-C6*C6, C1*C2 -
                       C4*C4, C5*C6-C3*C4, C6*C4-C1*C5, C4*C5-C2*C6])

    # Some variables
    w1 = I3**(-x13)
    w2 = x13*I1*I3**(-x43)
    w3 = I3**(-x23)
    w4 = x23*I2*I3**(-x53)
    w5 = x12*I3**(-x12)

    # Derivative of Reduced Invariants J wrt E
    J1E = w1*I1E - w2*I3E
    J2E = w3*I2E - w4*I3E
    J3E = w5*I3E

    # Calculate PK2 stress now
    J3 = np.sqrt(I3)
    Stress = A10*J1E + A01*J2E + K*(J3-1)*J3E

    # For Material stiffness
    D = np.zeros(shape=(6, 6))
    if LTAN:    # Asked to compute material stiffness D
        # dI_dEE
        I2EE = np.array([[0, 4, 4, 0, 0, 0],
                         [4, 0, 4, 0, 0, 0],
                         [4, 4, 0, 0, 0, 0],
                         [0, 0, 0, -2, 0, 0],
                         [0, 0, 0, 0, -2, 0],
                         [0, 0, 0, 0, 0, -2]])
        I3EE = np.array([[0, 4*C3, 4*C2, 0, -4*C5, 0],
                         [4*C3, 0, 4*C1, 0, -4*C4, 0, 0],
                         [0, 0, -4*C4, -2*C4, -2*C3, 2*C6, 2*C5],
                         [-4*C5, 0, 0, 2*C6, -2*C1, 2*C4],
                         [0, -4*C6, 0, 2*C5, 2*C4, -2*C2]])

        # Some variables
        w1 = x23*I3**(-x12)
        w2 = x89*I1*I3**(-x43)
        w3 = x13*I1*I3**(-x43)
        w4 = x43*I3**(-x12)
        w5 = x89*I2*I3**(-x53)
        w6 = I3**(-x23)
        w7 = x23*I2*I3**(-x53)
        w8 = I3**(-x12)
        w9 = x12*I3**(-x12)

        # dJ_dEE
        J1EE = -w1*(J1E@J3E.T + J3E@J1E.T) + w2*(J3E@J3E.T) - w3*I3EE
        J2EE = -w4*(J2E@J3E.T + J3E@J2E.T) + w5*(J3E@J3E.T) + w6*I2EE - w7*I3EE
        J3EE = -w8*(J3E@J3E.T) + w9*I3EE
        D = A10*J1EE + A01*J2EE + K*(J3E@J3E.T) + K*(J3-1)*J3EE
    return Stress, D
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
def cauchy(F, S):
    """ 
    Convert PK2 stress into Cauchy stress.
    Inputs:
            F : Deformation gradient, [3,3]
            S : PK2 Stress, [6,1]
    """
    # Translate S into 3x3 matrix
    PK = np.array([[S[0], S[3], S[5]],
                   [S[3], S[1], S[4]],
                   S[5], S[4], S[2]])
    detf = np.linalg.det(F)
    PKF = PK@F.T
    ST = F@PKF/detf
    # Translate to voigt notation
    cauchyStress = np.array(
        [ST[0, 0], ST[1, 1], ST[2, 2], ST[0, 1], ST[1, 2], ST[0, 3]])
    return cauchyStress
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
def hyper3d(prop, UPDATE, LTAN, ne, ndof, xyz, le, disptd, force, gkf, sigma):
    """
    Compute global stiffness matrix and residual force for hyperleastic material models.
    """
    # Integration points and weights
    XG = np.array([-0.577350, 0.577350])
    WGT = np.array([1.0, 1.0])

    # Index for history variables (eaxh integration points)
    intn = 0
    # -- Loop over elements, this is the main loop to compute K and F
    for ie in range(1, ne+1):
        # Nodal coordinates and incremental displacements
        elxy = xyz[le[ie-1]-1]
        # Local to global mapping
        # Each entry in idof corresponds to its actual number in global dofs
        idof = np.zeros(24)
        for i, v in enumerate(le[ie-1]-1):
            idof[ndof*i] = ndof*v
            idof[ndof*i + 1] = ndof*v + 1
            idof[ndof*i + 2] = ndof*v + 2
        dsp = disptd[idof].reshape(ndof, 8, order='F')

        # -- Loop over integration points
        for lx in range(2):
            for ly in range(2):
                for lz in range(2):
                    E1, E2, E3 = XG[lx], XG[ly], XG[lz]
                    intn += 1  # Keep track of the intergration point index
                    # Calcualte determinant and shape function derivative at this integration point
                    _, shpd, det = shapel(np.array([E1, E2, E3]), elxy)
                    FAC = WGT[lx]*WGT[ly]*WGT[lz]*det   # Just a variable
                    # Calculate Deformation gradient F = grad0U + I
                    F = dsp@shpd.T + np.eye(3)
                    # Compute PK2 stress and tangent stiffness
                    stress, dtan = mooney(F, prop[0], prop[1], prop[2], LTAN)
                    # Update plastic variable
                    if UPDATE:
                        stress = cauchy(F, stress)
                        sigma[:, intn-1] = stress.copy()
                    # -- Add residual force and tangent stiffness matrix
                    bn = np.zeros((6, 24))
                    bg = np.zeros((9, 24))
                    # Add entries
                    for i in range(8):
                        col = np.arange(i*ndof, i*ndof + 2 + 1)
                        bn[:, col] = np.array([
                                              [shpd[0, i]*F[0, 0], shpd[0, i]
                                                  * F[1, 0], shpd[0, i]*F[2, 0]],
                                              [shpd[1, i]*F[0, 1], shpd[1, i]
                                                  * F[1, 1], shpd[1, i]*F[2, 1]],
                                              [shpd[2, i]*F[0, 2], shpd[2, i]
                                                  * F[1, 2], shpd[2, i]*F[2, 2]],
                                              [shpd[0, i]*F[0, 1] + shpd[1, i]*F[0, 0], shpd[0, i]*F[1, 1] +
                                                  shpd[1, i]*F[1, 0], shpd[0, i]*F[2, 1] + shpd[1, i]*F[2, 0]],
                                              [shpd[1, i]*F[0, 2] + shpd[2, i]*F[0, 1], shpd[1, i]*F[1, 3] +
                                                  shpd[2, i]*F[1, 1], shpd[1, i]*F[2, 2] + shpd[2, i]*F[2, 1]],
                                              [shpd[0, i]*F[0, 2] + shpd[2, i]*F[0, 0], shpd[0, i]*F[1, 2] +
                                                  shpd[2, i]*F[1, 0], shpd[0, i]*F[2, 2] + shpd[2, i]*F[2, 0]]
                                              ])
                        bg[:, col] = np.array([
                            [shpd[0, i], 0, 0],
                            [shpd[1, i], 0, 0],
                            [shpd[2, 0], 0, 0],
                            [0, shpd[0, i], 0],
                            [0, shpd[1, i], 0],
                            [0, shpd[2, i], 0],
                            [0, 0, shpd[0, i]],
                            [0, 0, shpd[1, i]],
                            [0, 0, shpd[2, i]]])
                        # Residual forces
                        force[idof] -= FAC*bn.T@stress
                        # Tangent stiffness
                        if LTAN:
                            # Expand stress to matrix form
                            sig = np.array([[stress[0], stress[3], stress[5]],
                                            [stress[3], stress[1], stress[4]],
                                            [stress[5], stress[4], stress[2]]])
                            shead = np.kron(np.eye(3), sig)
                            ekf = bn.T@dtan@bn + bg.T@shead@bg
                            for i in range(24):
                                for j in range(24):
                                    gkf[idof[i], idof[j]] += FAC*ekf
    return force, gkf, sigma
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
def output(FLG, ITER, RESN, TIME, DELTA):
    """Print convergence history."""
    if FLG == 1:
        if ITER > 2:
            print(rf"Iter: {ITER}, Residual: {RESN} ")
        else:
            print(
                rf"Time: {np.round(TIME, 3)}, Time Step: {np.round(DELTA, 6)}, Iter: {ITER}, Residual: {RESN}")
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
def prout(OUTFILE, TIME, NUMNP, NE, NDOF, sigma, disptd):
    """Print converged displacements and stresses"""

    OUTFILE.write(
        f"\r\n\r\nTIME = {TIME:11.3e}\r\n\r\nNodal Displacements\r\n")
    OUTFILE.write(f"\r\n Node          U1          U2          U3")
    # These are the displacements at each node
    for i in np.arange(1, NUMNP+1):
        ii = NDOF*(i-1)
        OUTFILE.write(
            f"\r\n{i:5d} {disptd[ii]:11.3e} {disptd[ii+1]:11.3e} {disptd[ii+2]:11.3e}")

    OUTFILE.write(f"\r\n\r\nElement Stress\r\n")
    OUTFILE.write(
        f"\r\n        S11        S22        S33        S12        S23        S13")
    # These are the stresses at each integration points
    for i in np.arange(1, NE+1):
        OUTFILE.write(f"\r\nElement {i:5d}")
        for j in np.arange(0, 8*i):
            OUTFILE.write(
                f"\r\n{sigma[0, j]:11.3e} {sigma[1, j]:11.3e} {sigma[2, j]:11.3e} {sigma[3, j]:11.3e} {sigma[4, j]:11.3e} {sigma[5, j]:11.3e}")

    OUTFILE.write(f"\r\n\r\n")
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
def debug(arg=None):
    """Utility function to pause at the location where this is called."""
    input(rf"Paused at flag: {arg}")
# -------------------------------------------------------------------------


# ----------------------------------------------------------------------------
def nlfea(itra, tol, atol, ntol, tims, nout, MID, prop, extforce, sdispt, xyz, le):
    r""" Main program for Hyperelastic/elastoplastic analysis.

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
    # open file to store output.
    outfile = open(nout, "w")

    numnp, ndof = np.shape(xyz)
    ne = np.shape(le)[0]
    neq = ndof*numnp
    disptd = np.zeros(neq)  # Nodal displacement
    dispdd = np.zeros(neq)  # Nodal increment

    if MID >= 0:
        # Initialize material properties
        etan, sigma, xq = plset(prop, MID, ne)
    _ = itgzone(xyz, le, nout)    # Check element connectivity

    # Load increments [Start, End, Increment, InitialLoad, FinalLoad]
    nload = np.shape(tims)[0]
    iload = 1   # First load increment
    timef = tims[iload-1, 0]    # Starting time
    timei = tims[iload-1, 1]    # Ending time
    delta = tims[iload-1, 2]    # Time increment
    cur1 = tims[iload-1, 3]     # Start load factor
    cur2 = tims[iload-1, 4]     # End load factor
    delta0 = delta      # Saved time increment
    time = timef    # starting time
    tdelta = timei - timef  # Total time interval for load step
    itol = 1    # Bisection level
    tary = np.zeros(ntol)   # Time stamps for bisections

    # -- Load increment Loop
    # -----------------------------------------------------------------------
    istep = -1
    # First loop [10] is for load steps and load increments.
    # The loop has NLOAD load steps.Each load step is composed of multiple load increments. See Section 2.2.4 load increment force method.
    # The total load applied is divided by number of increments
    FLAG10 = 1
    while(FLAG10 == 1):  # Previous solution has converged
        FLAG10 = 0
        # Store previously converged displacement for the sake of bisection
        cdisp = disptd.copy()
        if itol == 1:  # No bisection
            delta = delta0
            tary[itol-1] = time + delta
        else:   # Recover previous bisection
            itol -= 1   # Reduce bisection level
            delta = tary[itol-1] - tary[itol]   # New time increment
            tary[itol] = 0  # Empty converged bisection level
            istep = istep - 1   # Decrease load increment
        time0 = time    # Save current time
        # Update stress and history variables
        UPDATE = True
        LTAN = False
        gkf = np.zeros((neq, neq))  # make it sparse
        force = np.zeros(neq)
        if MID == 0:
            force, gkf, sigma = elast3d(
                etan, UPDATE, LTAN, ne, ndof, xyz, le, disptd, force, gkf, sigma)
        elif MID > 0:
            raise NotImplementedError("Plast3d not available")
        elif MID < 0:
            raise hyper3d(prop, UPDATE, LTAN, ne, ndof, xyz, le)
        else:
            raise NotImplementedError("Wrong material")

        # Print Results
        if istep >= 0:
            prout(outfile, time, numnp, ne, ndof, sigma, disptd)

        time += delta    # Increase time
        istep += 1

        # Check time and control bisection
        # Second loop [11] is for bisection, if convergence cannot be obtained for current load.
        # If convergence iteration fails, load increment is halved. Then loop 11 is repeated from previously converged point.
        # There is a maximum number of bisection that can be done before giving up.
        # For bisection, the previously converged displacement is stored in CDISP.
        # This is because the displacement field DISPTD is updated on the go, and only reverted back using previous value in CDISP if bisection is needed.
        FLAG11 = 1
        while FLAG11:       # Bisection loop starts
            FLAG11 = 0
            if (time - timei) > 1e-10:  # Time passed the end time
                if (timei + delta - time) > 1e-10:  # One more at the end time
                    delta = timei + delta-time  # Time increment to end
                    delta0 = delta  # Saved time increment
                    time = timei  # current time is the end
                else:
                    iload += 1   # Progress to next step
                    if (iload > nload):  # Finished final load step
                        FLAG10 = 0  # Stop the program
                        break
                    else:   # Next load step
                        time -= delta
                        delta = tims[iload-1, 2]
                        delta0 = delta
                        time = time + delta
                        timef = tims[iload-1, 0]
                        timei = tims[iload-1, 1]
                        tdelta = timei - timef
                        cur1 = tims[iload-1, 3]
                        cur2 = tims[iload-1, 4]

            # Load factor and prescribed displacements
            factor = cur1 + (time - timef)/tdelta*(cur2 - cur1)
            sdisp = delta*sdispt[:, 2]/tdelta*(cur2-cur1)

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
                    elast3d(etan, UPDATE, LTAN, ne, ndof, xyz,
                            le, disptd, force, gkf, sigma)
                # Increase external force. First identify the correct global dof at which these loads are applied,i.e, ndof*node + dof.
                # Then add the current load increment to the position.
                loc = ndof*(extforce[:, 0]-1) + (extforce[:, 1] - 1)
                # Make sure the indices are integers. Look for better implementation.
                loc = np.array([int(iloc) for iloc in loc], dtype=np.int64)
                force[loc] += factor*extforce[:, 2]
                # Prescribed displacement BC
                ndisp = np.shape(sdispt)[0]
                fixeddof = ndof*(sdispt[:, 0] - 1) + sdispt[:, 1] - 1
                gkf[fixeddof] = np.zeros([ndisp, neq])

                Ieye = np.eye(ndisp)
                for i in range(len(fixeddof)):
                    for j in range(len(fixeddof)):
                        gkf[fixeddof[i], fixeddof[j]] = prop[0] * \
                            Ieye[i, j]

                force[fixeddof] = 0
                if iter == 1:
                    force[fixeddof] = prop[0] * sdisp[:]

                # Check convergence
                if iter > 1:
                    fixeddof = ndof*(sdispt[:, 0] - 1) + sdispt[:, 1] - 1
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
                            delta = delta/2
                            time = time0 + delta
                            tary[itol-1] = time
                            disptd = cdisp.copy()
                            print(
                                rf"Not converged. Bisecting load increment {itol}")
                        else:
                            raise RuntimeError(
                                "Maximum number of bisection reached without convergence. Terminating now...")

                        # If here, no runtime error raised. Proceed for the NR iterations
                        FLAG11 = 1
                        FLAG20 = 1
                        break
                # Solve the system of equation
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
    XYZ = np.array([[0, 0, 0],
                    [1, 0, 0],
                    [1, 1, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 1],
                    [1, 1, 1],
                    [0, 1, 1]])

    # Element connectivity
    LE = np.array([[1,  2,  3,  4,  5,  6,  7,  8]])

    # External forces [Node, DOF, Value]
    EXTFORCE = np.array([[5, 3, 10e3],
                         [6, 3, 10e3],
                         [7, 3, 10e3],
                         [8, 3, 10e3]])

    # Prescribed displacement [Node, DOF, Value]
    SDISPT = np.array([[1, 1, 0],
                       [1, 2, 0],
                       [1, 3, 0],
                       [2, 2, 0],
                       [2, 3, 0],
                       [3, 3, 0],
                       [4, 1, 0],
                       [4, 3, 0]])

    # Material Properties
    # MID = 0(Linear elastic) PROP: [lambda, mu]
    MID = 0
    E = 2e11
    NU = 0.3
    LAMBDA = E*NU/((1 + NU) * (1 - 2*NU))
    MU = E/(2 * (1 + NU))
    PROP = np.array([1.1538e6, 7.6923e5])

    # Load increments [Start, End, Increment, InitialFactor, FinalFactor]
    TIMS = np.array([[0, 0.5, 0.1, 0, 0.5],
                    [0.5, 1.0, 0.1, 0.5, 1]])

    # Set program parameter
    ITRA = 30
    ATOL = 1e5
    NTOL = 6
    TOL = 1E-6

    # Call main function
    NOUT = "output.out"
    out = nlfea(ITRA, TOL, ATOL, NTOL, TIMS, NOUT,
                MID, PROP, EXTFORCE, SDISPT, XYZ, LE)
# -------------------------------------------------------------------------
