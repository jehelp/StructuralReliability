"""
Modal analysis module
"""
import numpy as np
import numpy.linalg

########################################################################
def eigPro(M_m,K_m):
    """
    Returns eigen properties (angular frequencies and eigen vectors).
    Input:
    - M_m: structural mass matrix
    - K_m: structural stiffness matrix
    Output:
    - eigVal_v: vector of eigen angular frequencies
    - eigVec_m: matrix of eigen vectors
    """
    A_m = np.linalg.inv(K_m)*M_m
    eigVal_v,eigVec_m = np.linalg.eig(A_m)
    
    i = 0
    for omeInv in eigVal_v:
        if omeInv != 0:
            eigVal_v[i] = np.real(omeInv)**(-1/2)
        i += 1

    return eigVal_v,eigVec_m

########################################################################
