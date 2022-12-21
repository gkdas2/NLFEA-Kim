import numpy as np


def shapel(xi, elxy):
    r""" Compute shape function, derivatives, and determinant of hex element. 

    Inputs: 

            xi: (xi, eta, zi) in the reference coordinates is the location where shape function and derivative are to be evaluated.

            elxy: 3x8 matrix. It contains nodal coordinates of eight nodes of the element.

    Returns:

            sf: 8x1 array. Shape functions

            gdsf: 8x3 matrix. Derivatives of the shape function, dN_dx

            det: Scalar. Determinant of the Jacobian of the mapping.            
    """

    # Isoparametric element coordinates
    xnode = np.array([[-1,  1,  1, -1, -1,  1,  1, -1],
                      [-1, -1,  1,  1, -1, -1,  1,  1],
                      [-1, -1, -1, -1,  1,  1,  1,  1]])

    quar = 0.125

    # Shape functions
    sf = np.zeros((8, 1))

    # Derivative of shape function
    dsf = np.zeros((3, 8))

    for i in range(8):
        xp, yp, zp = xnode[:, i]    # xi_i, eta_i, zi_i

        # isoparametric shape functons. Eqn 1.136
        xi0 = np.array([1+xi(0)*xp, 1+xi(1)*yp, 1+xi(2)*zp])
        sf[i] = quar*xi0[0]*xi0[1]*xi0[2]
        dsf[0, i] = quar*xp*xi0[1]*xi0[2]
        dsf[1, i] = quar*yp*xi0[0]*xi0[2]
        dsf[2, i] = quar*zp*xi0[0]*xi0[1]

    gj = dsf@elxy   # jacobian dx_dxi
    det = np.linalg.det(gj)
    gjinv = np.linalg.inv(gj)
    gdsf = gjinv@dsf  # DN_dx Eqn 1.138
    
    return sf, gdsf, det
