import numpy as np
from nlfem.shapefunctions import hex3D
from nlfem.hardening import combHard, combHardTan
from nlfem.rotatedStress import rotatedStress


def plast3D(
    MID,
    prop,
    UPDATE,
    LTAN,
    ETAN,
    ne,
    ndof,
    xyz,
    le,
    disptd,
    dispdd,
    force,
    gkf,
    sigma,
    xq,
):
    """
    Compute global stiffness matrix and residual force for plastic material models.
    It keeps track of history dependent variables and stores them in global arrays
    after solution converges at given load increment.

    Inputs
    ================
    - MID : int
        material model. [1: infinitesimal elastoplastic , 2: infinitesimal elastoplasticity with finite rotation , 31: finite deformation elastoplastic ]
    - prop : [lambda, mu, beta, H, Y0]
        material constants for combined linear isotropic/kinematic hardening
        Here, beta = combined hardening parameter, H = plastic modulus, Y0 = Initial Yield stress
    - UPDATE : bool
        whether to store (True) stresses and history variables in global sigma and xq
    - LTAN : bool
        whether to calculate tangent stiffness matrix and store in global array gkf.
        The residual force is always computed.
    """
    # -- Integration points and weights
    XG = np.array([-0.577350, 0.577350])
    WGT = np.array([1.0, 1.0])

    # -- Index for history variables (each integration points)
    intn = 0

    # -- Loop over elements, this is the main loop to compute K and F
    for ie in range(1, ne + 1):
        # -- Nodal coordinates and incremental displacements
        elxy = xyz[le[ie - 1] - 1]
        # -- Local to global mapping
        # Each entry in idof corresponds to its actual number in global dofs
        idof = np.zeros(24, dtype=np.int32)
        for i, v in enumerate(le[ie - 1] - 1):
            idof[ndof * i] = ndof * v
            idof[ndof * i + 1] = ndof * v + 1
            idof[ndof * i + 2] = ndof * v + 2

        dsp = disptd[idof].reshape(ndof, 8, order="F")
        dspd = dispdd(idof).reshape(ndof, 8, order="F")

        # -- Loop over integration points
        for lx in np.arange(2):
            for ly in np.arange(2):
                for lz in np.arange(2):
                    E1, E2, E3 = XG[lx], XG[ly], XG[lz]
                    intn += 1  # Keep track of the intergration point index

                    # -- Calcualte determinant and shape function derivative at this integration point
                    _, shpd, det = hex3D(np.array([E1, E2, E3]), elxy)
                    FAC = WGT[lx] * WGT[ly] * WGT[lz] * det  # Just a variable

                    # -- Previous converged history variables
                    if MID > 30:
                        nalpha = 3
                        stressn = sigma[6:12, intn - 1]
                    else:
                        nalpha = 6
                        stressn = sigma[0:6, intn - 1]

                    alphan = xq[0:nalpha, intn - 1]
                    epn = xq[nalpha, intn - 1]

                    # Strain increment
                    if (MID == 2) or (MID == 31):
                        # -- Calculate Deformation gradient F = grad0U + I
                        F = dsp @ shpd.T + np.eye(3)

                        # At each integration point, the derivatives
                        # of the shape functions is wrt undeformed configuration.
                        # For MID = 2 or 31, we use updated Lagrangian formulation,
                        # so the material derivatives are converted into
                        # spatial derivatives by multiolying with inverse of F
                        shpd = np.linalg.inv(F).T @ shpd
                    deps = dspd @ shpd.T
                    ddeps = np.array(
                        [
                            deps[0, 0],
                            deps[1, 1],
                            deps[2, 2],
                            deps[0, 1] + deps[1, 0],
                            deps[1, 2] + deps[2, 1],
                            deps[0, 2] + deps[2, 0],
                        ]
                    )

                    # -- Compute stress, back stress and effective plastic strain
                    if MID == 1:
                        # -- Infinitesimal plasticity
                        stress, alpha, ep = combHard(
                            prop, ETAN, ddeps, stressn, alphan, epn
                        )
                    elif MID == 2:
                        # -- Plasticity with fiinite rotation
                        FAC = FAC * np.linalg.det(F)
                        stressn, alphan = rotatedStress(deps, stressn, alphan)
                        stress, alpha, ep = combHard(
                            prop, ETAN, ddeps, stressn, alphan, epn
                        )
                    elif MID == 31:
                        raise NotImplementedError(
                            "Finite strain elastoplasticity has not been implemented yet"
                        )

                    # -- Update plastic variable
                    if UPDATE:
                        # stress = cauchy(F, stress)
                        sigma[0:6, intn - 1] = stress.copy()
                        xq[:, intn - 1] = np.hstack((alpha, ep))
                        if MID > 30:
                            raise NotImplementedError(
                                "Finite strain elastoplasticity not done"
                            )
                            # sigma[6:12, intn - 1] = B.copy()

                    # -- Add residual force and tangent stiffness matrix
                    bn = np.zeros((6, 24))
                    bg = np.zeros((9, 24))

                    # -- Add entries
                    for i in range(8):

                        col = np.arange(i * ndof, i * ndof + ndof, dtype=np.int32)
                        bn[:, col] = np.array(
                            [
                                [shpd[0, i], 0, 0],
                                [0, shpd[1, i], 0],
                                [0, 0, shpd[2, i]],
                                [shpd[1, i], shpd[0, i], 0],
                                [0, shpd[2, i], shpd[1, i]],
                                [shpd[2, i], 0, shpd[0, i]],
                            ]
                        )

                        bg[:, col] = np.array(
                            [
                                [shpd[0, i], 0, 0],
                                [shpd[1, i], 0, 0],
                                [shpd[2, 0], 0, 0],
                                [0, shpd[0, i], 0],
                                [0, shpd[1, i], 0],
                                [0, shpd[2, i], 0],
                                [0, 0, shpd[0, i]],
                                [0, 0, shpd[1, i]],
                                [0, 0, shpd[2, i]],
                            ]
                        )

                    # -- Residual forces
                    force[idof] -= FAC * bn.T @ stress

                    # -- Tangent stiffness
                    if LTAN:
                        if MID == 1:
                            DTAN = combHardTan(prop, ETAN, ddeps, stressn, alphan, epn)
                        elif MID == 2:
                            dtan = combHardTan(prop, ETAN, ddeps, stressn, alphan, epn)

                            CTAN = np.array([
                                            [-stress[0], stress[0], stress[0], -stress[3], 0, -stress[5]],
                                            [stress[1], -stress[1], stress[1], -stress[3], -stress[4], 0],
                                            [stress[2], stress[2], -stress[2], 0, -stress[4], -stress[5]],
                                            [-stress[3], -stress[3], 0, -0.5 * (stress[0] + stress[1]),  -0.5 * stress[5], -0.5*stress[4]],
                                            [0, -stress[4], -stress[4], -0.5*stress[5], -0.5*(stress[1] + stress[2]), -0.5*stress[3]],
                                            [-stress[5], 0, -stress[5], -0.5*stress[4], -0.5*stress[3], -0.5*(stress[0] + stress[2])]
                            ])

                            # Expand stress to matrix form
                            sig = np.array(
                                [
                                    [stress[0], stress[3], stress[5]],
                                    [stress[3], stress[1], stress[4]],
                                    [stress[5], stress[4], stress[2]],
                                ]
                            )
                            shead = np.kron(np.eye(3), sig)
                            ekf = bn.T @ dtan @ bn + bg.T @ shead @ bg
                        elif MID == 31:
                            raise NotImplementedError("no finite strain elastoplasticity")
                        
                        
                        for i in range(24):
                            for j in range(24):
                                gkf[idof[i], idof[j]] += FAC * ekf[i, j]

    return force, gkf, sigma