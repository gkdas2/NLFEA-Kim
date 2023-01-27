import numpy as np


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
    # -- Some constants
    x12 = 1 / 2
    x13 = 1 / 3
    x23 = 2 / 3
    x43 = 4 / 3
    x53 = 5 / 3
    x89 = 8 / 9

    # -- Calculate Right Cauchy Green Deformation tensor and translate to Voigt notation
    C = F.T @ F
    C1 = C[0, 0]
    C2 = C[1, 1]
    C3 = C[2, 2]
    C4 = C[0, 1]
    C5 = C[1, 2]
    C6 = C[0, 2]

    # -- Calcualte Invariants
    I1 = C1 + C2 + C3
    I2 = C1 * C2 + C1 * C3 + C2 * C3 - C4 * C4 - C5 * C5 - C6 * C6
    I3 = np.linalg.det(C)
    
    
    # -- Calculate Derivatives of I wrt E
    I1E = 2 * np.array([1, 1, 1, 0, 0, 0])
    I2E = 2 * np.array([C2 + C3, C3 + C1, C1 + C2, -C4, -C5, -C6])
    I3E = 2 * np.array(
        [
            C2 * C3 - C5 * C5,
            C1 * C3 - C6 * C6,
            C1 * C2 - C4 * C4,
            C5 * C6 - C3 * C4,
            C6 * C4 - C1 * C5,
            C4 * C5 - C2 * C6,
        ]
    )

    # -- Some variables
    w1 = I3 ** (-x13)
    w2 = x13 * I1 * I3 ** (-x43)
    w3 = I3 ** (-x23)
    w4 = x23 * I2 * I3 ** (-x53)
    w5 = x12 * I3 ** (-x12)

    # -- Derivative of Reduced Invariants J wrt E
    J1E = w1 * I1E - w2 * I3E
    J2E = w3 * I2E - w4 * I3E
    J3E = w5 * I3E

    # -- Calculate PK2 stress now
    J3 = np.sqrt(I3)
    J3M1 = J3 - 1.0
    Stress = A10 * J1E + A01 * J2E + K * J3M1 * J3E

    # -- For Material stiffness
    D = np.zeros(shape=(6, 6))
    if LTAN:  # Asked to compute material stiffness D
        # dI_dEE
        I2EE = np.array(
            [
                [0, 4, 4, 0, 0, 0],
                [4, 0, 4, 0, 0, 0],
                [4, 4, 0, 0, 0, 0],
                [0, 0, 0, -2, 0, 0],
                [0, 0, 0, 0, -2, 0],
                [0, 0, 0, 0, 0, -2],
            ]
        )

        I3EE = np.array(
            [
                [0, 4 * C3, 4 * C2, 0, -4 * C5, 0],
                [4 * C3, 0, 4 * C1, 0, 0, -4 * C6],
                [4 * C2, 4 * C1, 0, -4 * C4, 0, 0],
                [0, 0, -4 * C4, -2 * C3, 2 * C6, 2 * C5],
                [-4 * C5, 0, 0, 2 * C6, -2 * C1, 2 * C4],
                [0, -4 * C6, 0, 2 * C5, 2 * C4, -2 * C2],
            ]
        )

        # Some variables
        w1 = x23 * I3 ** (-x12)
        w2 = x89 * I1 * I3 ** (-x43)
        w3 = x13 * I1 * I3 ** (-x43)
        w4 = x43 * I3 ** (-x12)
        w5 = x89 * I2 * I3 ** (-x53)
        w6 = I3 ** (-x23)
        w7 = x23 * I2 * I3 ** (-x53)
        w8 = I3 ** (-x12)
        w9 = x12 * I3 ** (-x12)

        # dJ_dEE
        #J1EE = -w1 * (J1E @ J3E.T + J3E @ J1E.T) + w2 * (J3E @ J3E.T) - w3 * I3EE
        J1EE = -w1 * ( np.outer(J1E, J3E) + np.outer(J3E, J1E)) + w2 * np.outer(J3E, J3E) - w3 * I3EE
        J2EE = (
            -w4 * (np.outer(J2E, J3E) + np.outer(J3E,J2E))
            + w5 * np.outer(J3E, J3E)
            + w6 * I2EE
            - w7 * I3EE
        )
        J3EE = -w8 * np.outer(J3E, J3E) + w9 * I3EE
        D = A10 * J1EE + A01 * J2EE + K * np.outer(J3E, J3E) + K * (J3 - 1) * J3EE
    return Stress, D
