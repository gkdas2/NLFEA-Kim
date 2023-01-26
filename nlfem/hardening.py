import numpy as np


def combHard(mp, D, deps, stressN, alphaN, epN):
    """Update stress and backstress for Linear Combined Isotropic/Kinematic Hardening.

    Inputs:
        mp: [lambda, mu, beta, H, Y0] are the material properties
        D: elastic stiffness matrix
        deps: current strain increment, NOT total strain
        stressN: [s11, s22, s33, t12, t23, t13] at previous load increment 
        alphaN: [a11, a22, a33, a12, a23, a13] at previous load increment
        epN: plastic strain at previous load increment
    """

    Iden = np.array([1, 1, 1, 0, 0, 0])

    # Some constants
    two3 = 2 / 3
    stwo3 = np.sqrt(two3)

    # Unpack material properties
    mu = mp[1]
    beta = mp[2]
    H = mp[3]
    Y0 = mp[4]

    # Tolerance for yield
    ftol = Y0 * 1e-6

    # Trial stress
    stresstr = stressN + D @ deps

    # Trace of trial stress
    I1 = np.sum(stresstr[0:3])

    # Deviatoric stress
    strDeviatoric = stresstr - I1 * Iden / 3
    
    # Shifted stress and norm
    eta = strDeviatoric - alphaN
    etat = np.sqrt(
        eta[0] ** 2
        + eta[1] ** 2
        + eta[2] ** 2
        + 2 * (eta[3] ** 2 + eta[4] ** 2 + eta[5] ** 2)
    )

    # Trial yield function
    fyld = etat - stwo3 * (Y0 + (1 - beta) * H * epN)

    # Test for yield. If still elastic, take values and exit. If not, proceed for correction
    if fyld < ftol:
        stress = stresstr
        alpha = alphaN
        ep = epN
        return stress, alpha, ep

    # Plastic consistency parameter
    gamma = fyld / (2 * mu + two3 * H)
    # Updated effective plastic strain
    ep = epN + gamma * stwo3
    # unit vector normal to f
    N = eta / etat
    # updated stress
    stress = stresstr - 2 * mu * gamma * N
    # updated back stress
    alpha = alphaN + two3 * beta * H * gamma * N

    return stress, alpha, ep


def combHardTan(mp, D, deps, stressN, alphaN, epN):
    """Tangent stiffness for Linear Combined Isotropic/Kinematic Hardening.
    This is similar to combHard function, but computes tangent alone.

    Inputs:
        mp: [lambda, mu, beta, H, Y0] are the material properties
        D: elastic stiffness matrix
        stressN: [s11, s22, s33, t12, t23, t13]
        alphaN: [a11, a22, a33, a12, a23, a13]
    """

    Iden = np.array([1, 1, 1, 0, 0, 0])

    # Some constants
    two3 = 2 / 3
    stwo3 = np.sqrt(two3)

    # Unpack material properties
    mu = mp[1]
    beta = mp[2]
    H = mp[3]
    Y0 = mp[4]

    # Tolerance for yield
    ftol = Y0 * 1e-6

    # Trial stress
    stresstr = stressN + D @ deps

    # Trace of trial stress
    I1 = np.sum(stresstr[0:3])

    # Deviatoric stress
    strDeviatoric = stresstr - I1 * Iden / 3

    # Shifted stress and norm
    eta = strDeviatoric - alphaN
    etat = np.sqrt(
        eta[0] ** 2
        + eta[1] ** 2
        + eta[2] ** 2
        + 2 * (eta[3] ** 2 + eta[4] ** 2 + eta[5] ** 2)
    )

    # Trial yield function
    fyld = etat - stwo3 * (Y0 + (1 - beta) * H * epN)

    # Test for yield. If still elastic, take values and exit. If not, proceed for correction
    if fyld < ftol:
        Dtan = D
        return Dtan

    # Plastic consistency parameter
    gamma = fyld / (2 * mu + two3 * H)

    # unit vector normal to f
    N = eta / etat

    # coefficients
    var1 = 4 * mu**2 / (2 * mu + two3 * H)
    var2 = 4 * mu**2 * gamma / etat

    # Tangent stiffness
    Dtan = D - (var1 - var2) * N @ N.T + var2 * Iden @ Iden.T / 3
    # contribution from 4-th order I tensor
    Dtan[0, 0] -= var2
    Dtan[1, 1] -= var2
    Dtan[2, 2] -= var2
    Dtan[3, 3] -= 0.5 * var2
    Dtan[4, 4] -= 0.5 * var2
    Dtan[5, 5] -= 0.5 * var2

    return Dtan
