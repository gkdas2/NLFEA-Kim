from asyncore import write
import string
import numpy as np

def connectivity2D(numElementsX,numElementsY):
    """Generate connectivity matrix for 2D rectangular mesh."""
    numElements = numElementsX * numElementsY # Total number of elements
    numNodes = (numElementsX+1)*(numElementsY+1) #Total number of Elements
    nodeIndex= np.arange(1,numNodes+1, dtype = int) #Index of all nodes

    mat = nodeIndex.reshape((numElementsY +1, numElementsX + 1))
    map = np.zeros([numElements, 4], dtype = int)

    counter = 0
    for ii in range(numElementsY):
        for jj in range(numElementsX):
            map[counter, 0] = mat[ii, jj]
            map[counter, 1] = mat[ii, jj +1]
            map[counter, 2] = mat[ii + 1, jj +1]
            map[counter, 3] = mat[ii + 1, jj]
            counter+=1

    return mat, map

def connectivity3D(numElementsX, numElementsY, numElementsZ):
    """Generate connectivity matrix for 3D hexahedral mesh."""
    numElements = numElementsX * numElementsY * \
        numElementsZ  # Total number of elements
    numNodes = (numElementsX+1)*(numElementsY+1) * \
        (numElementsZ + 1)  # Total number of Elements
    nodeIndex = np.arange(1, numNodes+1, dtype=np.int64)  # Index of all nodes

    mat = nodeIndex.reshape(
        (numElementsZ + 1, numElementsY + 1, numElementsX + 1))

    map = np.zeros((numElements, 8), dtype=np.int64)

    counter = 0
    for kk in np.arange(numElementsZ):
        for ii in np.arange(numElementsY):
            for jj in np.arange(numElementsX):
                map[counter, 0] = mat[kk, ii, jj]
                map[counter, 1] = mat[kk, ii, jj + 1]
                map[counter, 2] = mat[kk, ii + 1, jj + 1]
                map[counter, 3] = mat[kk, ii + 1, jj]
                map[counter, 4] = mat[kk+1, ii, jj]
                map[counter, 5] = mat[kk+1, ii, jj + 1]
                map[counter, 6] = mat[kk+1, ii + 1, jj + 1]
                map[counter, 7] = mat[kk+1, ii + 1, jj]

                counter += 1

    return mat, map

def generate2Dmesh(totalLengthx, totalHeight, numElementsX, numElementsY):
    """Gives 2D coordinates and element connectivity for rectangular mesh."""
    numElements = numElementsX * numElementsY # Total number of elements
    numNodes = (numElementsX+1)*(numElementsY+1) #Total number of Elements
    #dofIndex= np.arange(0,2*numNodes, dtype = int) #Index of all dofs  
    eleLength=totalLengthx/numElementsX;        #Individual Element length in X direction 
    eleWidth=totalHeight/numElementsY          #Individual Element lenght in Y direction
    elIndex = np.arange(1, numElements+1)
    print('Generating node coordinates...')
    x=np.arange(0, totalLengthx + eleLength, eleLength, dtype= float)
    y=np.arange(0, totalHeight + eleWidth, eleWidth, dtype = float)
    XX, YY  = np.meshgrid(x, y)
    YY = YY.flatten()
    XX = XX.flatten()
    COORD = np.zeros([numNodes, 2], dtype = float)
    COORD[:,0] = XX
    COORD[:,1] = YY
    print('Generating node coordinates... Done!')

    print('Generating 2D node arrangements...')
    mat, map = connectivity2D(numElementsX,numElementsY)
    print('Generating 2D node arrangements... Done!')

    return mat, map, COORD, numNodes, numElements, eleLength, eleWidth, elIndex

def generate3Dmesh(totalLengthx, totalLengthy,  totalLengthz, numElementsX, numElementsY, numElementsZ):
    """Gives 3D coordinates and element connectivity for hexahendral mesh."""
    numElements = numElementsX * numElementsY * \
        numElementsZ  # Total number of elements
    numNodes = (numElementsX+1)*(numElementsY+1) * \
        (numElementsZ + 1)  # Total number of Elements
    nodeIndex = np.arange(1, numNodes+1, dtype=np.int64)  # Index of all nodes

    # Individual Element length in X direction
    eleLengthX = totalLengthx/numElementsX
    # Individual Element lenght in Y direction
    eleLengthY = totalLengthy/numElementsY
    eleLengthZ = totalLengthz/numElementsZ

    elIndex = np.arange(1, numElements+1, dtype=np.int64)

    print('Generating node coordinates...')
    x = np.arange(0, totalLengthx + eleLengthX, eleLengthX, dtype=np.float64)
    y = np.arange(0, totalLengthy + eleLengthY, eleLengthY, dtype=np.float64)
    z = np.arange(0, totalLengthz + eleLengthZ, eleLengthZ, dtype=np.float64)
    YY, ZZ, XX = np.meshgrid(y, z, x)
    ZZ = ZZ.flatten()
    YY = YY.flatten()
    XX = XX.flatten()
    COORD = np.zeros([numNodes, 3], dtype=np.float64)
    COORD[:, 0] = XX
    COORD[:, 1] = YY
    COORD[:, 2] = ZZ

    print('Generating 3D node arrangements...')
    mat, map = connectivity3D(numElementsX, numElementsY, numElementsZ)

    return mat, map, COORD, numNodes, numElements, eleLengthX, eleLengthY, eleLengthZ, nodeIndex, elIndex

"""
def plot2Dmesh(map, COORD, rho, nElements, h, dof = 2, alpha = 0.5, figname = "plot2dmesh.pdf"):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    from matplotlib import cm
    print("Begin plotting...")
    if dof == 2:
    #     f = np.array([1, 2, 3, 4])
         NPE = 4

    fig, ax  =plt.subplots()

    patch_list = []

    for nn in  np.arange(nElements):
        nl = map[nn,:]
        c = rho[nn]
        ecoord= COORD[nl[0:NPE]-1, 0:dof]
        pol = Polygon(ecoord, closed=True,  linewidth = 1, color =[1-c,1-c,1- c])# edgecolor = [0,0,0], )
        patch_list.append(pol)
    
    pc = PatchCollection(patch_list,cmap=cm.gray_r, alpha=alpha, match_original=True)
    ax.add_collection(pc)
    #pc.set_array(rho)
    plt.axis('scaled')
    #ax.axis("off")
    fig.colorbar(pc, ax =ax, fraction = 0.046, pad = 0.04, )

    plt.xlabel(r'$x$', fontsize = 16)
    plt.ylabel(r'$y$', fontsize = 16, rotation=0)
    print("Begin plotting...Done!")
    fig.canvas.draw()
    plt.pause(0.001)
"""

def write2Dmesh(filename, map, coord):
    """Writes the coordinates and mesh data in Abaqus/Calculix format."""
    Nnodes, DOFs = coord.shape
    numElements, NPE = map.shape

    file = open(filename, "w")
    file.write("**This containes mesh information")
    file.write("\n*NODE, NSET=NALL")
    #file.write("\n")
    for i in range(Nnodes):
        file.write("\n" '  ' + str(int(i+1)) + '\t , \t' + str(round(coord[i, 0], 6)) + '  \t, \t  ' + str(
            round(coord[i, 1], 6)))
    file.write("\n \n")

    file.write("*ELEMENT, TYPE=CPS4, ELSET=EALL")
    for i in range(numElements):
        file.write("\n" + str(int(i+1)) + ' , ' + str(map[i, 0]) + ' , ' + str(map[i, 1]) + ' , ' + str(
            map[i, 2]) + ' , ' + str(map[i, 3]))
    file.write("\n")
    file.close()

def write3Dmesh(filename, map, coord):
    """Writes the coordinates and mesh data in Abaqus/Calculix format"""
    # Write the updated mesh file
    Nnodes, DOFs = coord.shape
    numElements, NPE = map.shape

    file = open(filename, "w")
    file.write("  **This containes mesh information")
    file.write("\n  *NODE, NSET=NALL")
    #file.write("\n")
    for i in range(Nnodes):
        file.write("\n" '  ' + str(int(i+1)) + '\t , \t' + str(round(coord[i, 0], 6)) + '  \t, \t  ' + str(
            round(coord[i, 1], 6)) + '  , ' + str(round(coord[i, 2], 6)))
    file.write("\n \n")

    file.write("*ELEMENT, TYPE=C3D8, ELSET=EALL")
    for i in range(numElements):
        file.write("\n" + str(int(i+1)) + ' , ' + str(map[i, 0]) + ' , ' + str(map[i, 1]) + ' , ' + str(
            map[i, 2]) + ' , ' + str(map[i, 3]) + ' , ' + str(map[i, 4]) + ' , ' + str(map[i, 5]) + ' , ' + str(map[i, 6]) + ' , ' +str(map[i,7]))
    file.write("\n")
    file.close()
    
if __name__ == "__main__":
    """
    A simple script to generate 2D/3D rectangular mesh in Abaqus format
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='A simple script to generate 2D/3D rectangular mesh in Abaqus format')
    parser.add_argument('--DM', type=float, default = 2, help = "Dimension: 2 = 2D or 3 = 3D")
    parser.add_argument('--LX', type=float, default = 1, help = "Length along x axis")
    parser.add_argument('--LY', type=float, default = 1, help = "Length along y axis")
    parser.add_argument('--LZ', type=float, default = 1, help = "Length along z axis")
    parser.add_argument('--NX', type=int, default = 1, help = "Number of elements along x axis")
    parser.add_argument('--NY', type=int, default = 1, help = "Number of elements along y axis")
    parser.add_argument('--NZ', type=int, default = 1, help = "Number of elements along z axis")
    parser.add_argument('--FILE', default = "mesh.msh",  help = "Filename with extension to store mesh information")
    

    a = parser.parse_args()
    
    DIMENSION, LX, LY, LZ, NX, NY, NZ, FILENAME = a.DM, a.LX, a.LY, a.LZ, a.NX, a.NY, a.NZ, a.FILE
    
    if DIMENSION == 2:
        mat, map, COORD, numNodes, numElements, eleLength, eleWidth, elIndex = generate2Dmesh(LX, LY, NX, NY)
        
        h = 0.2
        write2Dmesh(FILENAME, map, COORD)
        print(f" 2D mesh filename {FILENAME} written!")
        rho = np.linspace(0,1, numElements)*0 + 1

    if DIMENSION == 3:
        mat, map, COORD, numNodes, numElements, eleLengthX, eleLengthY, eleLengthZ, nodeIndex, elIndex = generate3Dmesh(LX, LY, LZ, NX, NY, NZ)
        write3Dmesh(FILENAME, map, COORD)
        print(f"3D mesh filename {FILENAME} written!")