import numpy as np


def elast3d(etan, update, ltan, ne, ndof, xyz, le, disptd, force, gkf, sigma):
    """
    Main program computing global stiffness residual force for linear elastic material model

    Inputs:
        etan: Size: [6,6]. Elastic stiffness matrix D for voigt notation. Eq 1.81
        
        update: Logical. True: save stress value
        
        ltan: logical. True: calculate global stiffness matrix
        
        ne: integer. Total number of elements
        
        ndof: integer. Dimension of problem (3)
        
        xyz: [nnode, 3]. Coordinates of all nodes
        
        le: [ne, 8]. Element connectivity   
    
        sigma: stress at each integration point (updated)
        
        force: residual force array. 
    Calls:
        shapel()    
    """

    # Integration points and weights
    xg = np.array([-0.57735026918, 0.57735026918])
    wgt = np.array([1.0, 1.0])
    
    # Loop over elements to compute K and F
    for i in np.arange(ne):
        # Nodal coordinates and incremental displacements
        
        elxy = np.array([xyz[ii-1] for ii in le[i]])
        
        print(elxy.shape)
        
        
        
        
        
        
        
        

if __name__ == "__main__":

    # Nodal coordinates
    xyz = np.array([[0,0,0],
                    [1,0,0],
                    [1,1,0],
                    [0,1,0],
                    [0,0,1],
                    [1,0,1],
                    [1,1,1],
                    [0,1,1]])
    
    # Element connectivity
    le = np.array([[1,2,3,4,5,6,7,8]])
    
    # External forces [Node, DOF, Value]
    extforce = np.array([[5,3,10e3],
                         [6,3,10e3],
                         [7,3,10e3],
                         [8,3,10e3]])
    
    # Material properties
    # MID = 0 (Linear elastic) Prop = [lambda nu]
    mid = 0
    prop = np.array([110.747e3, 80.1938e3])
    
    # Load increments [Start, End, Increment, InitialFactor, FinalFactor]
    tims = np.array([0.0, 1.0, 1, 0.0, 1.0])
    
    # Set program parameters
    itra = 30
    atol = 1.0e5
    ntol = 6
    tol = 1e-6
    
    # Call main function
        
    etan = 0
    update = 0
    ltan = 0
    ne = np.shape(le)[0]
    ndof = np.shape(xyz)[1]
    disptd = 0
    force = 0
    sigma = 0
    gkf = 0
    elast3d(etan, update, ltan, ne, ndof, xyz, le, disptd, force, gkf, sigma)        