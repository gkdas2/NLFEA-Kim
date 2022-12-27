import numpy as np
import matplotlib.pyplot as plt

"""
Example 2.3 Two nonlinear springs (Newton-Raphson method)
"""

tol = 1e-5
iter = 0
c = 0

u = np.array([0, 0])
uold = u.copy()
f = np.array([0, 100])

P = np.array([300*u[0]**2 + 400*u[0]*u[1] - 200*u[1]**2 + 150*u[0] - 100*u[1],
                        200*u[0]**2 - 400*u[0]*u[1] + 200*u[1]**2 - 100*u[0] + 100*u[1]])

R = f - P

conv = (R[0]**2 + R[1]**2) / (1 + f[0]**2 + f[1]**2)
c = np.abs(0.9 - u[1]) / np.abs(0.9 - uold[1])**2

print(rf" iter: {iter}, u1: {u[0]}, u2: {u[1]}, conv: {conv}, c: {c} ")

while conv > tol and iter < 20:
    
    Kt = np.array([[600*u[0] + 400*u[1] + 150, 400*(u[0] - u[1]) - 100],
                         [400*(u[0] - u[1]) - 100, 400*(u[1] - u[0]) + 100]])
    delu = np.linalg.solve(Kt, R)
    u = uold + delu
    
    P = np.array([300*u[0]**2 + 400*u[0]*u[1] - 200*u[1]**2 + 150*u[0] - 100*u[1],
                        200*u[0]**2 - 400*u[0]*u[1] + 200*u[1]**2 - 100*u[0] + 100*u[1]])  
    R = f - P
    conv = (R[0]**2 + R[1]**2) / (1 + f[0]**2 + f[1]**2)
    c = np.abs(0.9 - u[1]) / np.abs(0.9 - uold[1])**2
    uold = u.copy()
    iter += 1
    print(rf" iter: {iter}, u1: {u[0]}, u2: {u[1]}, conv: {conv}, c: {c} ")
