# -*- coding: utf-8 -*-
"""
RUS Module - All of the functions needed to compute resonant frequencies of 
freely supported RPP samples for hexagonal materials.
"""

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import time


def Hex_mat(c11, c33, c44, c13, c12):
    c66 = c66 = .5 * (c11 - c12)
    C_voigt = np.matrix([[c11, c12, c13, 0, 0, 0],  # Voigt matrix for Hexagonal symmetry
                         [c12, c11, c13, 0, 0, 0],
                         [c13, c13, c33, 0, 0, 0],
                         [0, 0, 0, c44, 0, 0],
                         [0, 0, 0, 0, c44, 0],
                         [0, 0, 0, 0, 0, c66]])
    return C_voigt


# This Section is where we convert the voigt stiffness matrix to the 4-D tensor
# notation used in the Gamma function.
def Voigt2Ten(C_voigt):
    C_tensor = np.floor(np.zeros((3, 3, 3, 3)))
    Decoder = np.matrix([[0, 5, 4],
                         [5, 1, 3],
                         [4, 3, 2]])
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    C_tensor[i][j][k][l] = C_voigt[Decoder[i, j], Decoder[k, l]]
    return C_tensor


def XYZ_exponents(nn):
    dim = (nn + 1) * (nn + 2) * (nn + 3) / 6
    dim = int(dim)
    ob = np.zeros(dim)
    mb = np.zeros(dim)
    nb = np.zeros(dim)
    ig = 0  # This sets up exponents for the powers-of-polynomial
    for o in range(nn + 1):  # basis used by Visscher (XYZ algorithm).
        for m in range(nn + 1):  # "on the normal modes ..."
            for n in range(nn + 1):
                if o + m + n <= nn:
                    ob[ig] = o
                    mb[ig] = m
                    nb[ig] = n
                    ig = ig + 1
    return dim, ob, mb, nb


def build_E_Matrix(d1, d2, d3, nn):
    dx = d1 / 2  # These are the values used in the E integrals. We integrate from
    dy = d2 / 2  # -dx to dx for each dimension.
    dz = d3 / 2
    (dim, ob, mb, nb) = XYZ_exponents(nn)
    # This will give the E blocks for which i and i' (assigning function
    # coefficients to the cartesian axes) are equal. When i=/=i' the E matrix 
    # has a zero value.       
    Ep = np.zeros((dim, dim))  # Ep means E partial, since its a part of the full E
    for q1 in range(dim):
        for q2 in range(dim):
            O = ob[q1] + ob[q2] + 1
            M = mb[q1] + mb[q2] + 1
            N = nb[q1] + nb[q2] + 1
            Ep[q1][q2] = (((dx) ** O - (-dx) ** O) * ((dy) ** M - (-dy) ** M) *
                          ((dz) ** N - (-dz) ** N) / (O * M * N))

    # Construct the E block (a block diagonal matrix)
    zBlock = np.zeros((dim, dim))
    E = np.block([[Ep, zBlock, zBlock], [zBlock, Ep, zBlock], [zBlock, zBlock, Ep]])
    return E


def build_Gamma_Matrix(d1, d2, d3, nn, C_tensor):
    dx = d1 / 2  # These are the values used in the E integrals. We integrate from
    dy = d2 / 2  # -dx to dx for each dimension.
    dz = d3 / 2
    (dim, ob, mb, nb) = XYZ_exponents(nn)
    # dPhi is theMatrix of the integrated partial derivatives of the phi function,
    # multiplied by stiffness tensor to obtain gamma matrix
    dPhi = np.zeros((3, 3))
    GammaMat = np.zeros((3 * dim, 3 * dim))
    for i in range(3):
        for k in range(3):
            for q1 in range(dim):
                for q2 in range(dim):
                    O11 = ob[q1] + ob[q2] - 1
                    M11 = mb[q1] + mb[q2] + 1
                    N11 = nb[q1] + nb[q2] + 1
                    if O11 * M11 * N11 == 0:
                        dPhi[0][0] = 0
                    else:
                        dPhi[0][0] = ((ob[q1] * ob[q2] * ((dx) ** O11 - (-dx) ** O11) *
                                       ((dy) ** M11 - (-dy) ** M11) * ((dz) ** N11 - (-dz) ** N11))
                                      / (O11 * M11 * N11))
                    O22 = ob[q1] + ob[q2] + 1
                    M22 = mb[q1] + mb[q2] - 1
                    N22 = nb[q1] + nb[q2] + 1
                    if O22 * M22 * N22 == 0:
                        dPhi[1][1] = 0
                    else:
                        dPhi[1][1] = ((mb[q1] * mb[q2] * ((dx) ** O22 - (-dx) ** O22) *
                                       ((dy) ** M22 - (-dy) ** M22) * ((dz) ** N22 - (-dz) ** N22))
                                      / (O22 * M22 * N22))
                    O33 = ob[q1] + ob[q2] + 1
                    M33 = mb[q1] + mb[q2] + 1
                    N33 = nb[q1] + nb[q2] - 1
                    if O33 * M33 * N33 == 0:
                        dPhi[2][2] = 0
                    else:
                        dPhi[2][2] = ((nb[q1] * nb[q2] * ((dx) ** O33 - (-dx) ** O33) *
                                       ((dy) ** M33 - (-dy) ** M33) * ((dz) ** N33 - (-dz) ** N33))
                                      / (O33 * M33 * N33))
                    O12 = ob[q1] + ob[q2]
                    M12 = mb[q1] + mb[q2]
                    N12 = nb[q1] + nb[q2] + 1
                    if O12 * M12 * N12 == 0:
                        dPhi[0][1] = 0
                        dPhi[1][0] = 0
                    else:
                        dPhi[0][1] = ((ob[q1] * mb[q2] * ((dx) ** O12 - (-dx) ** O12) *
                                       ((dy) ** M12 - (-dy) ** M12) * ((dz) ** N12 - (-dz) ** N12))
                                      / (O12 * M12 * N12))
                        dPhi[1][0] = ((ob[q2] * mb[q1] * ((dx) ** O12 - (-dx) ** O12) *
                                       ((dy) ** M12 - (-dy) ** M12) * ((dz) ** N12 - (-dz) ** N12))
                                      / (O12 * M12 * N12))
                    O13 = ob[q1] + ob[q2]
                    M13 = mb[q1] + mb[q2] + 1
                    N13 = nb[q1] + nb[q2]
                    if O13 * M13 * N13 == 0:
                        dPhi[0][2] = 0
                        dPhi[2][0] = 0
                    else:
                        dPhi[0][2] = ((ob[q1] * nb[q2] * ((dx) ** O13 - (-dx) ** O13) *
                                       ((dy) ** M13 - (-dy) ** M13) * ((dz) ** N13 - (-dz) ** N13))
                                      / (O13 * M13 * N13))
                        dPhi[2][0] = ((ob[q2] * nb[q1] * ((dx) ** O13 - (-dx) ** O13) *
                                       ((dy) ** M13 - (-dy) ** M13) * ((dz) ** N13 - (-dz) ** N13))
                                      / (O13 * M13 * N13))
                    O23 = ob[q1] + ob[q2] + 1
                    M23 = mb[q1] + mb[q2]
                    N23 = nb[q1] + nb[q2]
                    if O23 * M23 * N23 == 0:
                        dPhi[1][2] = 0
                        dPhi[2][1] = 0
                    else:
                        dPhi[1][2] = ((mb[q1] * nb[q2] * ((dx) ** O23 - (-dx) ** O23) *
                                       ((dy) ** M23 - (-dy) ** M23) * ((dz) ** N23 - (-dz) ** N23))
                                      / (O23 * M23 * N23))
                        dPhi[2][1] = ((mb[q2] * nb[q1] * ((dx) ** O23 - (-dx) ** O23) *
                                       ((dy) ** M23 - (-dy) ** M23) * ((dz) ** N23 - (-dz) ** N23))
                                      / (O23 * M23 * N23))

                    GammaMat[q1 + i * dim][q2 + k * dim] = (C_tensor[i][0][k][0] * dPhi[0][0] +
                                                            C_tensor[i][0][k][1] * dPhi[0][1] +
                                                            C_tensor[i][0][k][2] * dPhi[0][2] +
                                                            C_tensor[i][1][k][0] * dPhi[1][0] +
                                                            C_tensor[i][1][k][1] * dPhi[1][1] +
                                                            C_tensor[i][1][k][2] * dPhi[1][2] +
                                                            C_tensor[i][2][k][0] * dPhi[2][0] +
                                                            C_tensor[i][2][k][1] * dPhi[2][1] +
                                                            C_tensor[i][2][k][2] * dPhi[2][2])
    return GammaMat


def rus_function_hex(c11, c33, c44, c13, c12, d1, d2, d3, rho, nn):
    C_voigt = Hex_mat(c11, c33, c44, c13, c12)
    C_tensor = Voigt2Ten(C_voigt)
    E = build_E_Matrix(d1, d2, d3, nn)
    gamma = build_Gamma_Matrix(d1, d2, d3, nn, C_tensor)
    # Eigenvalue problem
    eVal, V = linalg.eigh(gamma, rho * E)
    w2 = eVal[6:]  # Throw away the rigid-body modes.
    frequency = np.sqrt(w2) / (2 * np.pi)  # Resonant Frequencies in Hz.
    return frequency, V


def plot_modeshape(mode_number, d1, d2, d3, nn, eigVec):  # modes are numbered beginning with 1, not 0.
    dx = d1 / 2
    dy = d2 / 2
    dz = d3 / 2
    V = eigVec
    (dim, ob, mb, nb) = XYZ_exponents(nn)
    modenum = mode_number
    modenum = modenum + 5
    nn = int((len(V[:, 1]) / 3))
    Vvec = eigVec[:, modenum]
    ax = np.matrix([Vvec[0:nn]])
    ay = np.matrix([Vvec[nn:2 * nn]])
    az = np.matrix([Vvec[2 * nn:]])

    samplesx = 100
    samplesy = 100
    phi = np.zeros([dim])
    ux = np.zeros([samplesx, samplesy])
    uy = np.zeros([samplesx, samplesy])
    uz = np.zeros([samplesx, samplesy])
    normU = np.zeros([samplesx, samplesy])
    for i in range(samplesx):
        x = dx * (2 * i) / (samplesx - 1) - dx
        for j in range(samplesy):
            y = dy / (samplesy - 1) * j * 2 - dy
            z = dz
            for q1 in range(dim):
                O = ob[q1]
                M = mb[q1]
                N = nb[q1]
                phi[q1] = x ** O * y ** M * z ** N
            phi = phi.T
            ux[i][j] = ax @ phi
            uy[i][j] = ay @ phi
            uz[i][j] = az @ phi
            normU[i][j] = np.sqrt((ux[i][j] ** 2 +
                                   uy[i][j] ** 2 +
                                   uz[i][j] ** 2))

    X = np.linspace(-dx, dx, samplesx)
    Y = np.linspace(-dy, dy, samplesy)
    XX, YY = np.meshgrid(X, Y)
    plt.contourf(XX, YY, uz.T)
    plt.show()
