import numpy as np
from nlfem.shapefunctions import shapel

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
    # -- Integration points and weights
    xg = np.array([-0.57735026918, 0.57735026918])
    wgt = np.array([1.0, 1.0])
    # -- Stress storage index (No of integration points)
    intn = 0

    # --------------------------------------------------------------------------
    # -- Loop over elements to compute K and F
    for ie in np.arange(1, ne + 1):
        # Nodal coordinates and incremental displacements
        elxy = xyz[le[ie - 1] - 1]
        # -- Each entry in idof corresponds to its actual number in global dofs
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
                    e1 = xg[lx]
                    e2 = xg[ly]
                    e3 = xg[lz]
                    intn += 1
                    # -- Determinant and shape function derivative
                    _, shpd, det = shapel(np.array([e1, e2, e3]), elxy)
                    fac = wgt[lx] * wgt[ly] * wgt[lz] * det
                    # -- Strain
                    # gradient of u, r=eqn 1.139
                    deps = dsp @ shpd.T
                    # Strain in Voight notation (note y12 = 2e12)
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
                    # -- Stress
                    # stress = D*strain using voigt notation
                    stress = etan @ ddeps

                    # -- Update stress(store stress in this big global array sigma, that acculumates stresses at all integration points) for all elements
                    if UPDATE:
                        sigma[:, intn - 1] = stress.copy()

                    # -- Add residual force and stiffness matrix
                    # -- Assemble the B matrix = dN_dx_. Eqn 1.140
                    bm = np.zeros((6, 24))
                    for i in range(8):
                        col = np.arange(i * ndof, i * ndof + ndof)
                        bm[:, col] = np.array(
                            [
                                [shpd[0, i], 0, 0],
                                [0, shpd[1, i], 0],
                                [0, 0, shpd[2, i]],
                                [shpd[1, i], shpd[0, i], 0],
                                [0, shpd[2, i], shpd[1, i]],
                                [shpd[2, i], 0, shpd[0, i]],
                            ]
                        )
                    # -- Residual force
                    # -- internal force  = integral(B.T * sigma )
                    force[idof] = force[idof] - fac * bm.T @ stress

                    # -- Compute element tangent stiffness and iteratively add to global tangent stiffness matrix
                    if LTAN:
                        ekf = bm.T @ etan @ bm  # ke_intergationPoint  = B.T*D*B
                        for i in range(24):
                            for j in range(24):
                                gkf[idof[i], idof[j]] += fac * ekf[i, j]
    return force, gkf, sigma
