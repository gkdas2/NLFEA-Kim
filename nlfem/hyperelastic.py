import numpy as np
from nlfem.shapefunctions import hex3D
from nlfem.mooney import mooney


def hyper3d(prop, UPDATE, LTAN, ne, ndof, xyz, le, disptd, force, gkf, sigma):
    """
    Compute Cauchy stress, global stiffness matrix and residual force for hyperleastic material models.
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

        # -- Loop over integration points
        for lx in np.arange(2):
            for ly in np.arange(2):
                for lz in np.arange(2):
                    E1, E2, E3 = XG[lx], XG[ly], XG[lz]
                    intn += 1  # Keep track of the intergration point index

                    # -- Calcualte determinant and shape function derivative at this integration point
                    _, shpd, det = hex3D(np.array([E1, E2, E3]), elxy)
                    FAC = WGT[lx] * WGT[ly] * WGT[lz] * det  # Just a variable

                    # -- Calculate Deformation gradient F = grad0U + I
                    F = dsp @ shpd.T + np.eye(3)

                    # -- Compute PK2 stress and tangent stiffness
                    stress, dtan = mooney(F, prop[0], prop[1], prop[2], LTAN)

                    # -- Update plastic variable
                    if UPDATE:
                        #stress = cauchy(F, stress)
                        sigma[:, intn - 1] = stress.copy()

                    #print(f"ie:{ie}, intn: {intn}, sig: {stress}")

                    # -- Add residual force and tangent stiffness matrix
                    bn = np.zeros((6, 24))
                    bg = np.zeros((9, 24))

                    # -- Add entries
                    for i in range(8):

                        col = np.arange(i * ndof, i * ndof + ndof, dtype=np.int32)
                        bn[:, col] = np.array(
                            [
                                [
                                    shpd[0, i] * F[0, 0],
                                    shpd[0, i] * F[1, 0],
                                    shpd[0, i] * F[2, 0],
                                ],
                                [
                                    shpd[1, i] * F[0, 1],
                                    shpd[1, i] * F[1, 1],
                                    shpd[1, i] * F[2, 1],
                                ],
                                [
                                    shpd[2, i] * F[0, 2],
                                    shpd[2, i] * F[1, 2],
                                    shpd[2, i] * F[2, 2],
                                ],
                                [
                                    shpd[0, i] * F[0, 1] + shpd[1, i] * F[0, 0],
                                    shpd[0, i] * F[1, 1] + shpd[1, i] * F[1, 0],
                                    shpd[0, i] * F[2, 1] + shpd[1, i] * F[2, 0],
                                ],
                                [
                                    shpd[1, i] * F[0, 2] + shpd[2, i] * F[0, 1],
                                    shpd[1, i] * F[1, 2] + shpd[2, i] * F[1, 1],
                                    shpd[1, i] * F[2, 2] + shpd[2, i] * F[2, 1],
                                ],
                                [
                                    shpd[0, i] * F[0, 2] + shpd[2, i] * F[0, 0],
                                    shpd[0, i] * F[1, 2] + shpd[2, i] * F[1, 0],
                                    shpd[0, i] * F[2, 2] + shpd[2, i] * F[2, 0],
                                ],
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
                        for i in range(24):
                            for j in range(24):
                                gkf[idof[i], idof[j]] += FAC * ekf[i, j]

    return force, gkf, sigma


def cauchy(F, S):
    """
    Convert PK2 stress into Cauchy stress.
    Inputs:
            F : Deformation gradient, [3,3]
            S : PK2 Stress, [6,1]
    """
    # -- Translate S into 3x3 matrix
    PK = np.array([[S[0], S[3], S[5]], [S[3], S[1], S[4]], [S[5], S[4], S[2]]])
    detf = np.linalg.det(F)
    PKF = PK @ F.T
    ST = F @ PKF / detf
    # -- Translate to voigt notation
    cauchyStress = np.array(
        [ST[0, 0], ST[1, 1], ST[2, 2], ST[0, 1], ST[1, 2], ST[0, 2]]
    )
    return cauchyStress


# -------------------------------------------------------------------------
