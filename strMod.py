"""
Module for building the structural model
pierre.jehel@centralesupelec.fr
v. 2017-12-17
"""

import numpy as np
import numpy.matlib

########################################################################
def masMat(nDof,inpMas,fixDof):
    """
    Returns structural lumped mass matrix.
    Input:
    - nDof: number of active DOFs
    - inpMas: list with the lumped masses
    - fixDof: list of active DOFs
    Output:
    - M_m: structural mass matrix
    """
    M_m = np.matlib.zeros((nDof,nDof))
    for masNod in inpMas:
        nod = masNod[0]
        dof = np.sum(fixDof[0:nod][:])
        j = 1
        for act in fixDof[nod][:]:
            M_m[dof,dof] = masNod[j]
            dof+=act
            j+=1

    return M_m

########################################################################
def stiMat(nDof,strEle,mesDat,eltPro,fixDof):
    """
    Assemble and returns structural stiffness matrix.
    -------
    Inputs:
    strEle: for each element [node I, node J, type]
    mesDat: for each element [length, cos, sin]
    eltPro: for each elemet type [material properties]
     - type 0 (beam): [matE,secA,secIy]
     - type 1 (connection): [secMpl]
    matE: material elastic modulus
    secA: element section area
    secIy: section moment of inertia about its centroid axis
    secMpl: section plastic moment
    -------
    Output:
    K_m: structural stiffness matrix
    """
    # initialize
    # Note: in FE programs, matrices are not stored like below (zeros are not stored); here
    #       this is just for illustration
    K_m = np.matlib.zeros((nDof,nDof))

    # loop over the elements
    eNum = 0
    for elt in strEle:

        nodI = elt[0]     # node numbers
        nodJ = elt[1]
        eltTyp = elt[2]   # element type
        secTyp = elt[3]   # section type

        # build element stiffness matrix in local coord. system
        if eltTyp == 0:   # element of type 0 (beam/column)
            eltL = mesDat[eNum][0]
            matE = eltPro[secTyp][0]
            secA = eltPro[secTyp][1]
            secI = eltPro[secTyp][2]
            KeLoc_m = beaSti_EB_2D_loc(matE,secA,eltL,secI)

        elif eltTyp == 1:   # rigid rotational spring
            secMpl = eltPro[eltTyp][0]
    #        d_a1 = np.concatenate(dis_v[I:I+3], dis_v[J:J+3])
    #        KeLoc_a2 = strMod.rigRotSpr_2D_loc(E,A,Lf,I,My,d_a1) -> To be developed for NL response
            KeLoc_m = rigRotSprSti(secMpl)

        # transform to global coordinate system
        cos = mesDat[eNum][1]
        sin = mesDat[eNum][2]
        R_m = glo2loc_2D(cos,sin)
        KeGlo_m = np.transpose(R_m)*KeLoc_m*R_m

        # assemble structural stiffness
        dofI = np.sum(fixDof[0:nodI][:])
        dofI = np.int(dofI)
        dofJ = np.sum(fixDof[0:nodJ][:])
        dofJ = np.int(dofJ)
        i = 0
        j = 0
        dof1 = dofI
        for actI in fixDof[nodI][:]:
            actI = np.int(actI)
            dof2 = dofI
            for actJ in fixDof[nodI][:]:
                actJ = np.int(actJ)
                K_m[dof1,dof2] += actI*actJ*KeGlo_m[i,j]
                dof2+=actJ
                j+=1
            dof2 = dofJ
            for actJ in fixDof[nodJ][:]:
                actJ = np.int(actJ)
                K_m[dof1,dof2] += actI*actJ*KeGlo_m[i,j]
                dof2+=actJ
                j+=1
            dof1+=actI
            i+=1
            j = 0
        dof1 = dofJ
        for actI in fixDof[nodJ][:]:
            actI = np.int(actI)
            dof2 = dofI
            for actJ in fixDof[nodI][:]:
                actJ = np.int(actJ)
                K_m[dof1,dof2] += actI*actJ*KeGlo_m[i,j]
                dof2+=actJ
                j+=1
            dof2 = dofJ
            for actJ in fixDof[nodJ][:]:
                actJ = np.int(actJ)
                K_m[dof1,dof2] += actI*actJ*KeGlo_m[i,j]
                dof2+=actJ
                j+=1
            dof1+=actI
            i+=1
            j = 0

        eNum+=1

    return K_m

########################################################################
def rayDamCoe(xi,ome):
    """
    Returns Rayleigh damping coefficients.
    Initial stiffness is used.
    Coefficients are computed once for all in the beginning of the analysis.
    -------
    Inputs:
    xi: strutural damping ratio (same for all modes)
    ome: eigen angular frequencies
    -------
    Output:
    alp, bet: Rayleigh damping coefficients
    """
    bet = 2*xi/(ome[0]+ome[1])
    alp = bet*ome[0]*ome[1]

    return alp,bet

########################################################################
def beaSti_EB_2D_loc(E,A,L,I):
    """
    Returns 2D Euler-Bernoulli beam stiffness in local coordinate system.
    Section is assumed to be constant all over the beam length.
    -------
    Inputs:
    E: elastic modulus
    A: beam section area
    L: beam length
    I: section moment of inertia about its centroid axis
    -------
    Output:
    Ke_m: stiffness matrix
    """
    k1 = E*A/L
    k2 = 12*E*I/L**3
    k3 = 6*E*I/L**2
    k4 = 4*E*I/L

    # start filling the upper triangle only (K_m is symmetric)
    Ke_m = np.matlib.zeros((6,6))
    Ke_m[0,0] = k1
    Ke_m[0,3] = -k1
    Ke_m[1,1] = k2
    Ke_m[1,2] = k3
    Ke_m[1,4] = -k2
    Ke_m[1,5] = k3
    Ke_m[2,2] = k4
    Ke_m[2,4] = -k3
    Ke_m[2,5] = k4/2
    Ke_m[3,3] = k1
    Ke_m[4,4] = k2
    Ke_m[4,5] = -k3
    Ke_m[5,5] = k4

    # fill lower triangle
    for i in range(1,6):
        for j in range(0,i):
            Ke_m[i,j] = Ke_m[j,i]

    return Ke_m

########################################################################
def rigRotSprSti(secIMpl):

    kInf = 1e16
    kRot = 1e16

    # start filling the upper triangle only (K_m is symmetric)
    Ke_m = np.matlib.zeros((6,6))
    Ke_m[0,0] = kInf
    Ke_m[0,3] = -kInf
    Ke_m[1,1] = kInf
    Ke_m[1,2] = kInf
    Ke_m[1,4] = -kInf
    Ke_m[1,5] = kInf
    Ke_m[2,2] = kRot
    Ke_m[2,4] = -kInf
    Ke_m[2,5] = -kRot
    Ke_m[3,3] = kInf
    Ke_m[4,4] = kInf
    Ke_m[4,5] = -kInf
    Ke_m[5,5] = kRot

    # fill lower triangle
    for i in range(1,6):
        for j in range(0,i):
            Ke_m[i,j] = Ke_m[j,i]

    return Ke_m

########################################################################
def glo2loc_2D(c,s):
    """
    Build rotation matrix from global to local 2D coordinate system.
    -------
    Inputs:
    c: cosine in radian of the angle from global to local coordinate system
    s: sine in radian of the angle from global to local coordinate system
    -------
    Output:
    R_m: rotation matrix from local to global coordinate system
    """
    R_m = np.matlib.zeros((6,6))
    R_m[0,0] = c
    R_m[0,1] = s
    R_m[1,0] = -s
    R_m[1,1] = c
    R_m[2,2] = 1
    for i in range(3,6):
        for j in range(3,6):
            I = i-3
            J = j-3
            R_m[i,j] = R_m[I,J]

    return R_m

########################################################################
def extLoa(nDof,fixDof,bouCon):
    """
    Build external forces vector at active DOFs.
    -------
    Inputs:
    - nDof: number of active DOFs
    - fixDof: list of active DOFs
    - bouCon: list of boundary conditions (displacements or forces)
    -------
    Output:
    fe_v: external forces vector at active DOFs
    """

    fe_v = np.matlib.zeros((nDof,1))
    i = 0
    k = 0
    for nodDof in fixDof:
        j = 0
        for dof in nodDof:
            if dof == 0:
                j+=1
            else:
                fe_v[k] = dof*bouCon[i][j]
                j+=1
                k+=1
        i+=1

    return fe_v

########################################################################
#def main():
