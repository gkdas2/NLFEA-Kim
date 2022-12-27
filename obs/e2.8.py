import numpy as np
import matplotlib.pyplot as plt

"""
Example 2.8 Displacement controlled procedure
"""

tol = 1e-5
conv = 0
u1 = 0
u1old = u1


# Displacement increment loop
for i in range(1, 10):
    u2 = 0.1*i
    P = 300*u1**2 + 400*u1*u2 - 200*u2**2 + 150*u1 - 100*u2
    
    R = 0-P
    conv = R**2
    
    # Convergence loop for each step
    iter = 0
    while conv > tol and iter < 20:
        Kt = 600*u1 + 400*u2 + 150
        delu1 = R/Kt
        u1 = u1old + delu1
        P = P = 300*u1**2 + 400*u1*u2 - 200*u2**2 + 150*u1 - 100*u2
        R = 0-P
        
        conv = R**2
        u1old = u1
        iter += 1

    F = 200*u1**2 - 400*u1*u2 + 200*u2**2 - 100*u1 + 100*u2 

    plt.plot(F, u1, '-o')
    plt.plot(F, u2, '-s')
    
    print(rf" iter: {i}, u1: {u1}, u2: {u2}, F: {F} ")

plt.savefig('e2.8.png')