import numpy as np


def convertToTensor3(S):
    return np.array([[S[0], S[3], S[5]], [S[3], S[1], S[4]], [S[5], S[4], S[2]]])


def convertToVoigt3(S):
    return np.array([S[0, 0], S[1, 1], S[2, 2], S[0, 1], S[1, 2], S[0, 2]])


def rotatedStress(L, S, A):
    """Rotate stress and back stress to the rotation-free configuration.

    Inputs:
        L: [3,3] - velocity gradient ddelu/d(n+1)x = [dui/dxj]
        S: [6,1] - stress
        A: [6,1] - basck stress

    Returns:
        Rotation-free S and A
    """

    # convert to a 3x3 tensor
    strTensor = convertToTensor3(S)
    alpTensor = convertToTensor3(A)

    R = L @ (np.linalg.inv(np.eye(3) - 0.5 * L)) # Midpoint configuration ddelu/d(n+1/2)x
    W = 0.5 * (R - R.T)

    R = np.eye(3) + np.linalg.inv(np.eye(3) - 0.5 * W) @ W

    strTensor = R @ strTensor @ R.T
    alpTensor = R @ alpTensor @ R.T

    stress = convertToVoigt3(strTensor)
    alpha = convertToVoigt3(alpTensor)

    return stress, alpha
