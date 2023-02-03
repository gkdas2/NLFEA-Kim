# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 03:30:40 2023

@author: lordl
"""

import numpy as np
from nlfem.nonlinear_static import nlfea

XYZ = np.array([[0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1]])

# Element connectivity
LE = np.array([[1,  2,  3,  4,  5,  6,  7,  8]])

# External forces [Node, DOF, Value]
EXTFORCE = np.array([[5, 3, 10e3],
                     [6, 3, 10e3],
                     [7, 3, 10e3],
                     [8, 3, 10e3]])

# Prescribed displacement [Node, DOF, Value]
SDISPT = np.array([[1, 1, 0],
                   [1, 2, 0],
                   [1, 3, 0],
                   [2, 2, 0],
                   [2, 3, 0],
                   [3, 3, 0],
                   [4, 1, 0],
                   [4, 3, 0]])

# Material Properties
# MID = 0(Linear elastic) PROP: [lambda, mu]
MID = 1
E = 2e11
NU = 0.3
LAMBDA = E*NU/((1 + NU) * (1 - 2*NU))
MU = E/(2 * (1 + NU))
PROP = np.array([110.747E3, 80.1938, 0.0, 10E3, 50E3, 1, 0.0001])

# Load increments [Start, End, Increment, InitialFactor, FinalFactor]
TIMS = np.array([[0, 0.5, 0.1, 0, 0.5],
                 [0.5, 1.0, 0.1, 0.5, 1]])

# Set program parameter
ITRA = 30
ATOL = 1e5
NTOL = 6
TOL = 1E-6

# Call main function
NOUT = "output.out"
out = nlfea(ITRA, TOL, ATOL, NTOL, TIMS, NOUT,
            MID, PROP, EXTFORCE, SDISPT, XYZ, LE)
# -------------------------------------------------------------------------
