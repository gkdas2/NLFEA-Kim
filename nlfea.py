import numpy as np

##


def shapel(xi, elxy):
    r""" Compute shape function, derivatives, and determinant of hex element. 

    Inputs: 

            xi: (xi, eta, zi) in the reference coordinates is the location where shape function and derivative are to be evaluated.

            elxy: 8x3 matrix. It contains nodal coordinates of eight nodes of the element.

    Returns:

            sf: 8x1 array. Shape functions

            gdsf: 8x3 matrix. Derivatives of the shape function, dN_dx

            det: Scalar. Determinant of the Jacobian of the mapping.            
    """

    # Isoparametric element coordinates
    xnode = np.array([[-1,  1,  1, -1, -1,  1,  1, -1],
                      [-1, -1,  1,  1, -1, -1,  1,  1],
                      [-1, -1, -1, -1,  1,  1,  1,  1]])

    quar = 0.125

    # Shape functions
    sf = np.zeros(8)

    # Derivative of shape function
    dsf = np.zeros((3, 8))

    for i in range(8):
        xp, yp, zp = xnode[:, i]    # xi_i, eta_i, zi_i

        # isoparametric shape functons. Eqn 1.136
        xi0 = np.array([1+xi(0)*xp, 1+xi(1)*yp, 1+xi(2)*zp])
        sf[i] = quar*xi0[0]*xi0[1]*xi0[2]   # N(xi_)
        dsf[0, i] = quar*xp*xi0[1]*xi0[2]   # dN_dxi
        dsf[1, i] = quar*yp*xi0[0]*xi0[2]   # dN_deta
        dsf[2, i] = quar*zp*xi0[0]*xi0[1]   # dN_dzi

    gj = dsf@elxy   # jacobian dx_dxi
    det = np.linalg.det(gj)
    gjinv = np.linalg.inv(gj)
    gdsf = gjinv@dsf  # DN_dx Eqn 1.138

    return sf, gdsf, det

##


def elast3d(etan, UPDATE, LTAN, ne, ndof, xyz, le, disptd, force, gkf, sigma):
    """
    Main program computing global stiffness residual force for linear elastic material model

    Inputs:
        etan: Size: [6,6]. Elastic stiffness matrix D for voigt notation. Eq 1.81

        UPDATE: Logical. True: save stress value

        LTAN: logical. True: calculate global stiffness matrix

        ne: integer. Total number of elements

        ndof: integer. Dimension of problem (3)

        xyz: [nnode, 3]. Coordinates of all nodes

        le: [ne, 8]. Element connectivity   

        sigma: stress at each integration point (updated)

        force: residual force array. 
    Calls:
        shapel() at each integration points    
    """

    # Integration points and weights
    xg = np.array([-0.57735026918, 0.57735026918])
    wgt = np.array([1.0, 1.0])

    # Stress storage index (No of integration points)
    intn = 0
    # Loop over elements to compute K and F
    for ie in np.arange(ne):
        # Nodal coordinates and incremental displacements

        elxy = xyz[le[ie]-1]

        # Local to global mapping
        idof = np.zeros(24)
        for i in range(8):
            ii = i*ndof
            idof[ii:ii+ndof] = np.arange((le[ie, i]-1)
                                         * ndof, (le[ie, i]-1)*ndof + ndof)

        dsp = disptd[idof].reshape(ndof, 8)

        # Loop over integration points
        for lx in np.arange(2):
            for ly in np.arange(2):
                for lz in np.arange(2):
                    e1 = xg[lx]
                    e2 = xg[ly]
                    e3 = xg[lz]
                    intn += 1

                    # Determinant and shape function derivative
                    _, shpd, det = shapel(np.array([e1, e2, e3]), elxy)
                    fac = wgt[lx]*wgt[ly]*wgt[lz]*det

                    # Strain
                    deps = dsp@shpd.T  # gradient of u, r=eqn 1.139
                    # Strain in Voight notation (note y12 = 2e12)
                    ddeps = np.array([deps[1, 1], deps[2, 2], deps[3, 3], deps[1, 2] +
                                     deps[2, 1], deps[2, 3]+deps[3, 2], ], deps[1, 3]+deps[3, 1])

                    # Stress
                    stress = etan@ddeps  # stress = D*strain using voigt notation

                    # Update stress(store stress in this big global array sigma, that acculumates stresses at all integration points) for all elements
                    if UPDATE:
                        sigma[:, intn-1] = stress.copy()

                    # Add residual force and stiffness matrix
                    
                    # Assemble the B matrix = dN_dx_. Eqn 1.140
                    bm = np.zeros((6, 24))
                    for i in range(8):
                        col = np.arange(i*ndof,   i*ndof + ndof)
                        bm[:, col] = np.array([[shpd[0, i, ], 0, 0],
                                               [0, shpd[1, i], 0],
                                               [0, 0, shpd[2, i]],
                                               [0, shpd[2, i], shpd[1, i]],
                                               [shpd[2, i], 0, shpd[0, i]]])
                    # Residual force
                    # internal force  = integral(B.T * sigma )
                    force[idof] = force[idof] - fac*bm.T@stress

                    # Compute element tangent stiffness and iteratively add to global tangent stiffness matrix
                    if LTAN:
                        ekf = bm.T@etan@bm  # ke_intergationPoint  = B.T*D*B

                        for i in range(24):
                            for j in range(24):
                                gkf[idof[i], idof[j]] += fac*ekf

##


def plset(prop, MID, ne):
    """
    Initialize history variables and elastic stiffness matrix.

    XQ : 1-6 = Back stress alpha, 7 = Effective plastic strain

    SIGMA : Stress for rate-form plasticity

            Left Cauchy-Green tensor XB for multiplicative plasticity

            ETAN : Elastic stiffness matrix
    """

    lam = prop[1]
    mu = prop[2]

    n = 8*ne

    if MID > 30:
        sigma = np.zeros((12, n))
        xq = np.zeros((4, n))
        sigma[6:9, :] = 1
        etan = np.array([[lam + 2*mu, lam, lam],
                         [lam, lam+2*mu, lam],
                         [lam, lam, lam+2*mu]])

    else:
        sigma = np.zeros((6, n))
        xq = np.zeros((7, n))
        etan = np.array([
                        [lam+2*mu, lam, lam, 0, 0, 0],
                        [lam, lam+2*mu, lam, 0, 0, 0],
                        [lam, lam, lam+2*mu, 0, 0, 0],
                        [0, 0, 0, mu, 0, 0],
                        [0, 0, 0, 0, mu, 0],
                        [0, 0, 0, 0, 0, mu]
                        ])

    return etan, sigma, xq

##


def itgzone(xyz, le, nout):
    """
    Check element connectivity and calculate total volume

    Returns: 
        Volume: total volume of the mesh
    """
    eps = 1e-7
    ne = np.shape(le)[0]
    volume = 0

    for i in range(ne):
        elxy = xyz[le[0]-1]
        _, _, det = shapel(np.array([0, 0, 0]), elxy)
        dvol = 8*det
        if dvol < eps:
            print(f"Negative Jacobian in element {i}")
            print(f"Node coordinates are {elxy}")
            raise ValueError(" Negative Jacobian")

        volume += dvol

    return volume


##
def nlfea(itra, tol, atol, ntol, tims, nout, MID, prop, extforce, sdispt, xyz, le, force, gkf):
    """
    Main program for Hyperelastic/elastoplastic analysis
    """

    numnp, ndof = np.shape(xyz)

    ne = np.shape(le)[0]

    neq = ndof*numnp

    disptd = np.zeros(neq)  # Nodal displacement
    dispdd = np.zeros(neq)  # Nodal increment

    if MID >= 0:
        etan = plset(prop, MID, ne)  # Initialize material properties

    _ = itgzone(xyz, le, nout)    # Check element connectivity

    # Load increments [Start, End, Increment, InitialLoad, FinalLoad]
    nload = np.shape(tims)[1]
    iload = 1   # First load increment
    timef = tims[0, iload-1]    # Starting time
    timei = tims[1, iload-1]    # Ending time
    delta = tims[2, iload-1]    # Time increment
    cur1 = tims[3, iload-1]     # Start load factor
    cur2 = tims[4, iload-1]     # End load factor

    delta0 = delta      # Saved time increment
    time = timef    # starting time
    tdelta = timei - timef  # Time interval for load step
    itol = 1    # Bisection level
    tary = np.zeros(ntol)   # Time stamps for bisections

    # -- Load increment Loop

    istep = -1
    FLAG10 = 1

    while(FLAG10 == 1):  # Solution has converged
        FLAG10 = 0
        FLAG11 = 1
        FLAG20 = 1

        cdisp = disptd.copy()   # Store converged displacement

        if itol == 1:  # No bisection
            delta = delta0
            tary[itol-1] = time + delta
        else:   # Recpver previous bisection
            itol -= 1   # Reduce bisection level
            delta = tary[itol-1] - tary[itol]   # New time increment
            tary[itol] = 0  # Empty converge dbisection level
            istep = istep - 1   # Decrease load increment

        time0 = time    # Save current time

        # Update stress and history variables

        UPDATE = True
        LTAN = False

        if MID == 0:
            elast3d()
