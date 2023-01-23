import numpy as np

# -------------------------------------------------------------------------
def output(FLG, ITER, RESN, TIME, DELTA):
    """Print convergence history on screen."""
    if FLG == 1:
        if ITER > 2:
            print(rf"Iter: {ITER}, Residual: {RESN} ")
        else:
            print(
                rf"Time: {np.round(TIME, 3)}, Time Step: {np.round(DELTA, 6)}, Iter: {ITER}, Residual: {RESN}"
            )


# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
def prout(OUTFILE, TIME, NUMNP, NE, NDOF, sigma, disptd):
    """Print converged displacements and stresses to file."""

    OUTFILE.write(f"\r\n\r\nTIME = {TIME:11.3e}\r\n\r\nNodal Displacements\r\n")
    OUTFILE.write(f"\r\n Node          U1          U2          U3")
    # --These are the displacements at each node
    for i in np.arange(1, NUMNP + 1):
        ii = NDOF * (i - 1)
        OUTFILE.write(
            f"\r\n{i:5d} {disptd[ii]:11.3e} {disptd[ii+1]:11.3e} {disptd[ii+2]:11.3e}"
        )

    OUTFILE.write(f"\r\n\r\nElement Stress\r\n")
    OUTFILE.write(
        f"\r\n        S11        S22        S33        S12        S23        S13        vonMises"
    )
    # -- These are the stresses at each integration points
    for i in np.arange(1, NE + 1):
        OUTFILE.write(f"\r\nElement {i:5d}")
        for j in np.arange(8 * (i - 1), 8 * i):
            vm = np.sqrt(
                0.5
                * (
                    (sigma[0, j] - sigma[1, j]) ** 2
                    + (sigma[1, j] - sigma[2, j]) ** 2
                    + (sigma[0, j] - sigma[2, j]) ** 2
                )
                + 3 * (sigma[3, j] ** 2 + sigma[4, j] ** 2 + sigma[5, j] ** 2)
            )
            OUTFILE.write(
                f"\r\n{sigma[0, j]:11.3e} {sigma[1, j]:11.3e} {sigma[2, j]:11.3e} {sigma[3, j]:11.3e} {sigma[4, j]:11.3e} {sigma[5, j]:11.3e} {vm:11.3e}"
            )

    OUTFILE.write(f"\r\n\r\n")


# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
def debug(arg=None):
    """Utility function to pause at the location where this is called."""
    input(rf"Paused at flag: {arg}")


# -------------------------------------------------------------------------
