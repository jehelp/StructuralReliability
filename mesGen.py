"""
Mesh generation module
pierre.jehel@centralesupelec.fr
v. 2017-12-16
"""
import numpy as np

def eltL0(strEle,nodCoo):
    """
    Define element length and orientation
    Input:
    - strEle: for each element [node I, node J, type]
    - nodCoo: nodes coordinates [X, Y]
    Output:
    - mesDat: for each element [length, cos(theta), sin(theta)]
    """
    mesDat = []
    for elt in strEle:
        nodI = np.int(elt[0])
        nodJ = np.int(elt[1])
        dX = nodCoo[nodJ][0]-nodCoo[nodI][0]
        dY = nodCoo[nodJ][1]-nodCoo[nodI][1]
        eleLen = np.sqrt(dX**2+dY**2)
        if np.abs(eleLen) > 1e-16:
            cos = dX/eleLen
            sin = dY/eleLen
        else:
            cos = 1
            sin = 0
        mesDat.append([eleLen,cos,sin])
        
    return mesDat

