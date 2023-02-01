import numpy as np

def convertToTensor3(S: np.ndarray) -> np.ndarray:
    return np.array([[S[0], S[3], S[5]], [S[3], S[1], S[4]], [S[5], S[4], S[2]]])


def convertToVoigt3(S: np.ndarray) -> np.ndarray:
    return np.array([S[0, 0], S[1, 1], S[2, 2], S[0, 1], S[1, 2], S[0, 2]])

def mulPlast(mp, D, L, b, alpha, ep):
    """
    Perform return-mapping for multiplicative plasticity with linear combined hardening
    to get stress and plastic variables.

    Inputs:
    =================
    mp : [lambda, mu, beta, H, Y0]
        Material Properties
    D : [3, 3]
        Elasticity matrix between principal stress and log principal stretch
    L : [3, 3]
        Velocity gradient [dui/dxj]
    b : [6, 1]    
        elastic left Cauchy Green deformation tensor
    alpha : [3, 1]
        principal back stress
    ep : float or int
        effective plastic strain       

    Returns:
    ===================
    stress : 
        Kirchoff stress

    """

    #-- Constants
    EPS = 1e-12
    Iden = np.array([1, 1, 1])
    two3 = 2/3
    stwo3 = np.sqrt(two3) 

    #-- Unpack material properties
    mu, beta, H, Y0 = mp[1:]

    # tolerance for yield
    ftol = 1e-6

    # Calculate incremental deformation gradient using velocity gradient
    R = np.linalg.inv(np.eye(3) - L)

    # Update elastic C-G deformation tensor using incremental defirmation gradient
    bm = convertToTensor3(b)
    bm = R@bm@R.T

    # Calculate the eigenvalues(principle stretches) and eigenvectors (principle directions)
    # of updated left C-G deformation tensor.
    V, P = np.linalg.eig(bm)

    b = convertToVoigt3(bm)

    # Assemble eigenvector matrices
    M = np.zeros((6, 3))
    M[0] = V[0]**2
    M[1] = V[1]**2
    M[2] = V[2]**2
    M[3] = V[0] * V[1]
    M[4] = V[1] * V[2]
    M[5] = V[0] * V[2]
    
    # Principle stretch
    eigen = np.array([P[0, 0], P[1,1], P[2, 2]])
    
    
    
    
    


    # Using log principle stretches, find principal Kirchoff stresses
    # using return mapping
    
    
    # Update stress and plastic variables using plastic consistency paramenters


