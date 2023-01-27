import numpy as np
from nlfem.nonlinear_static import nlfea

# Hyperelastic tension example 1 element

XYZ = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ]
)

# Element connectivity
LE = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])

# External forces [Node, DOF, Value]
EXTFORCE = np.array([])

# Prescribed displacement [Node, DOF, Value]
SDISPT = np.array(
    [
        [1, 1, 0],
        [4, 1, 0],
        [5, 1, 0],
        [8, 1, 0],
        [1, 2, 0],
        [2, 2, 0],
        [5, 2, 0],
        [6, 2, 0],
        [1, 3, 0],
        [2, 3, 0],
        [3, 3, 0],
        [4, 3, 0],
        [2, 1, 5],
        [3, 1, 5],
        [6, 1, 5],
        [7, 1, 5],
    ]
)

# Contact Elements
LEC = []

# Material Properties
# MID = 0(Linear elastic) PROP: [lambda, mu]
# Material properties
# MID:0(Linear elastic)
#        PROP=[LAMBDA NU]
# MID=0;
#PROP=[1.1538E6, 7.6923E5];
#     1(Linear hardening) 2(Saturated hardening)
#     31(Linear hardening finite deformation)
#     32(Saturated hardening finite deformation)
#        PROP=[LAMDA MU BETA H Y0 CE1 FTOL]
#MID=1;
#PROP=[110.747E9   80.1938E9   0.0    10.E3  30.0E3  1.0  0.0001];
#     -N(Hyperelasticity with N parameters)
#        PROP=[A10 A01 D]
MID = -1
#E = 2e11
#NU = 0.3
#LAMBDA = E * NU / ((1 + NU) * (1 - 2 * NU))
#MU = E / (2 * (1 + NU))
PROP = np.array([80, 20, 1e7])

# Load increments [Start, End, Increment, InitialFactor, FinalFactor]
TIMS = np.array([[0, 1.0, 0.05, 0.0, 1.0]])

# Set program parameter
ITRA = 30
ATOL = 1e5
NTOL = 6
TOL = 1e-6

# Call main function
NOUT = "output.out"
out = nlfea(ITRA, TOL, ATOL, NTOL, TIMS, NOUT,
            MID, PROP, EXTFORCE, SDISPT, XYZ, LE)
# -------------------------------------------------------------------------
