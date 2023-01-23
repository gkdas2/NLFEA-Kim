import numpy as np

# -------------------------------------------------------------------------
def shapel(xi, elxy):
    """Compute shape function, derivatives, and determinant of hex element.

    Inputs:
            xi: (xi, eta, zi) in the reference coordinates is the location where shape function and derivative are to be evaluated.
            elxy: 8x3 matrix. It contains nodal coordinates of eight nodes of the element.

    Returns:
            sf: 8x1 array. Shape functions
            gdsf: 3x8 matrix. Derivatives of the shape function, dN_dx
            det: Scalar. Determinant of the Jacobian of the mapping.
    """
    # -- Isoparametric element coordinates
    xnode = np.array(
        [
            [-1, 1, 1, -1, -1, 1, 1, -1],
            [-1, -1, 1, 1, -1, -1, 1, 1],
            [-1, -1, -1, -1, 1, 1, 1, 1],
        ]
    )
    quar = 0.125
    # -- Shape functions
    sf = np.zeros(8)
    # -- Derivative of shape function
    dsf = np.zeros((3, 8))

    for i in range(8):
        xp, yp, zp = xnode[:, i]
        # xi_i, eta_i, zi_i
        # isoparametric shape functons. Eqn 1.136
        xi0 = np.array([1 + xi[0] * xp, 1 + xi[1] * yp, 1 + xi[2] * zp])
        sf[i] = quar * xi0[0] * xi0[1] * xi0[2]  # N(xi_)
        dsf[0, i] = quar * xp * xi0[1] * xi0[2]  # dN_dxi
        dsf[1, i] = quar * yp * xi0[0] * xi0[2]  # dN_deta
        dsf[2, i] = quar * zp * xi0[0] * xi0[1]  # dN_dzi

    # jacobian dx_dxi
    gj = dsf @ elxy
    det = np.linalg.det(gj)
    gjinv = np.linalg.inv(gj)
    # DN_dx Eqn 1.138
    gdsf = gjinv @ dsf
    return sf, gdsf, det


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
    n = 8 * ne  # 8 integration points per element

    if MID > 30:
        sigma = np.zeros((12, n))
        xq = np.zeros((4, n))
        sigma[6:9, :] = 1
        etan = np.array(
            [
                [lam + 2 * mu, lam, lam],
                [lam, lam + 2 * mu, lam],
                [lam, lam, lam + 2 * mu],
            ]
        )
    else:
        sigma = np.zeros((6, n))
        xq = np.zeros((7, n))
        etan = np.array(
            [
                [lam + 2 * mu, lam, lam, 0, 0, 0],
                [lam, lam + 2 * mu, lam, 0, 0, 0],
                [lam, lam, lam + 2 * mu, 0, 0, 0],
                [0, 0, 0, mu, 0, 0],
                [0, 0, 0, 0, mu, 0],
                [0, 0, 0, 0, 0, mu],
            ]
        )
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
        elxy = xyz[le[i - 1] - 1]
        _, _, det = shapel(np.array([0, 0, 0]), elxy)
        dvol = 8 * det
        if dvol < eps:
            print(f"Negative Jacobian in element {i}")
            print(f"Node coordinates are {elxy}")
            raise ValueError(" Negative Jacobian")
        volume += dvol
    return volume
