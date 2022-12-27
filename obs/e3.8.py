import numpy as np
import matplotlib.pyplot as plt

"""
Example 3.8 Uniaxial bar (total Lagrangian formulation)
"""

tol = 1e-5

iter = 0

E = 200

# Initial displacement u = 0
u = 0      

# Force. We will apply all force at once, no incremental force here
f = 100

# First, compute current state for given displacement u
strain = u + 0.5 * u**2
stress = E*strain
P = stress*(1+u)    # Internal force
R = f-P

conv = R**2/(1+f**2)

print(rf" iter: {iter}, u: {u}, conv: {conv}, stress : {stress}, strain: {strain}")

# Now compute delu and update current displacement since convergence is not met for u_Initial
while conv > tol and iter < 20:
    
    Kt = E*(1+u)**2 + stress
    delu = R/Kt
    
    u = u + delu
    
    # For this new displacemnt check state for residual
    strain = u + 0.5 * u**2
    stress = E*strain
    P = stress*(1+u)
    R = f-P
    
    conv = R**2/(1+f**2)
    
    iter += 1
    print(rf" iter: {iter}, u: {u}, conv: {conv}, stress : {stress}, strain: {strain}")
