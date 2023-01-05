import numpy as np

def combHard1D(mp, deps, stressN, alphaN, epN):
    """Calculates stress, back stress, and plastic strain for a given strain increment using combined Kinematic/Isotropic Hardening in 1D
    
    Inputs:
        mp = [E=Young's Modulus, beta = hardening factor, H = Plastic Modulus, Y0 = Initial Yield Stress]
        deps = strain increment
        stressN = stress at load step N
        alphaN = back stress at load step N
        epN = plastic strain at load step N
    """
    # Unpack material properties    
    E, beta, H, Y0 = mp
    # tolerance for yield
    ftol = Y0*1e-6
    
    # trial stress
    stresstr = stressN + E*deps 
    
    # trial shifted stress
    etatr = stresstr - alphaN
    
    # trial yield function
    fyld = np.abs(etatr) - (Y0 + (1-beta)*H*epN)
    
    if fyld < ftol:
        # Material is elastic. 
        stress = stresstr
        alpha = alphaN 
        ep =  epN
    else:
        # Material is actual1y plastic, and requires return mapping
        # Update plastic strain increment, update stress, back stress and plastic strain
        dep = fyld/(E + H)
        stress = stresstr - np.sign(etatr)*E*dep
        alpha = alphaN + np.sign(etatr)*beta*H*dep
        ep = epN + dep    
    
    return stress, alpha, ep 


       