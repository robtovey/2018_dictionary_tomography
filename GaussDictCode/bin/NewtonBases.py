'''
Created on 14 Sep 2018

@author: Rob Tovey
'''
from numpy import empty, sqrt, exp, pi, array
from numba import cuda, float32 as f4, int32 as i4
from math import exp as c_exp, sqrt as c_sqrt, floor
from cmath import exp as comp_exp, sqrt as comp_sqrt
from .numba_cuda_aux import THREADS, __GPU_reduce_n
DIM = 3
'''
Generic:
F(I,x,r,y) = f(r(y-x))
RF = I*__Rf(r(y-x), rtheta/|rtheta|)/|rtheta|

Radial:
f = g(|r(y-x)|^2)
__Rf = __Rg(|(I-\theta\theta^T)y|^2) 
__Rg(y^2) = \int g(x^2+y^2) dx

Simplifications:
Y = y-x
rY = r(y-x)
hT = (r\theta)/|r\theta|
MrY = (I-hThT^T)rY
n = |MrY|^2
__Rf = __Rg(n)


__f = evaluate in volume
__Rf = evaluate in sinogram
__dRf = evaluate derivatives in sinogram
__derivs_dRf = derivatives of |Rf-C|^2
'''

# Radial basis:


# @cuda.jit('f4(f4)', device=True, inline=True)
@cuda.jit(device=True, inline=True)
def __g(n):
    if n > 1:
        return 0
    else:
        return c_exp(-30 * n) * c_sqrt(30 / pi)


# @cuda.jit('f4(f4)', device=True, inline=True)
@cuda.jit(device=True, inline=True)
def __Rg(n):
    # int exp(-(x^2+n)/2)/sqrt(2pi) dx = exp(-n/2)
    if n > 1:
        return 0
    else:
        return c_exp(-30 * n)


# @cuda.jit('void(f4,f4[:])', device=True, inline=True)
@cuda.jit(device=True, inline=True)
def __dRg(n, dg):
    # R = exp(-n/2)
    # dR = -R/2, ddR = R/4
    if n > 1:
        dg[0] = 0
        dg[1] = 0
        dg[2] = 0
    else:
        dg[0] = c_exp(-30 * n)
        dg[1] = -30 * dg[0]
        dg[2] = 900 * dg[0]

# @cuda.jit('f4(f4)', device=True, inline=True)
# def __g(n):
#     # sum(a*sqrt(g/pi)*exp(-gn)) -> sum(a*exp(-g*n))
#     if n > 1:
#         return 0
#     else:
#         v = 0
#
#         a = complex(7.238054e-01, -1.226411e+00)
#         g = 4.5*complex(3.572318e+00, -2.101695e+01)
#         v += (a * comp_sqrt(g / pi) * comp_exp(-g * n)).real
#         a = complex(-7.482410e+01, 7.694003e+01)
#         g = 4.5*complex(8.079788e+00, -1.740200e+01)
#         v += (a * comp_sqrt(g / pi) * comp_exp(-g * n)).real
#         a = complex(1.346014e+03, -1.435227e+03)
#         g = 4.5*complex(1.107646e+01, -1.408444e+01)
#         v += (a * comp_sqrt(g / pi) * comp_exp(-g * n)).real
#         a = complex(-9.149758e+03, 1.234362e+04)
#         g = 4.5*complex(1.318472e+01, -1.088592e+01)
#         v += (a * comp_sqrt(g / pi) * comp_exp(-g * n)).real
#         a = complex(2.929669e+04, -5.669524e+04)
#         g = 4.5*complex(1.464249e+01, -7.746260e+00)
#         v += (a * comp_sqrt(g / pi) * comp_exp(-g * n)).real
#         a = complex(2.409383e+04, -2.429935e+05)
#         g = 4.5*complex(1.601469e+01, -1.544321e+00)
#         v += (a * comp_sqrt(g / pi) * comp_exp(-g * n)).real
#         a = complex(-4.551067e+04, 1.506981e+05)
#         g = 4.5*complex(1.556592e+01, -4.637619e+00)
#         v += (a * comp_sqrt(g / pi) * comp_exp(-g * n)).real
#
#         return v
#
#
# @cuda.jit('f4(f4)', device=True, inline=True)
# def __Rg(n):
#     # int exp(-(x^2+n)/2)/sqrt(2pi) dx = exp(-n/2)
#     # sum(a*sqrt(g/pi)*exp(-gn)) -> sum(a*exp(-g*n))
#     if n > 1:
#         return 0
#     else:
#         v = 0
#
#         a = complex(7.238054e-01, -1.226411e+00)
#         g = 4.5*complex(3.572318e+00, -2.101695e+01)
#         v += (a * comp_exp(-g * n)).real
#         a = complex(-7.482410e+01, 7.694003e+01)
#         g = 4.5*complex(8.079788e+00, -1.740200e+01)
#         v += (a * comp_exp(-g * n)).real
#         a = complex(1.346014e+03, -1.435227e+03)
#         g = 4.5*complex(1.107646e+01, -1.408444e+01)
#         v += (a * comp_exp(-g * n)).real
#         a = complex(-9.149758e+03, 1.234362e+04)
#         g = 4.5*complex(1.318472e+01, -1.088592e+01)
#         v += (a * comp_exp(-g * n)).real
#         a = complex(2.929669e+04, -5.669524e+04)
#         g = 4.5*complex(1.464249e+01, -7.746260e+00)
#         v += (a * comp_exp(-g * n)).real
#         a = complex(2.409383e+04, -2.429935e+05)
#         g = 4.5*complex(1.601469e+01, -1.544321e+00)
#         v += (a * comp_exp(-g * n)).real
#         a = complex(-4.551067e+04, 1.506981e+05)
#         g = 4.5*complex(1.556592e+01, -4.637619e+00)
#         v += (a * comp_exp(-g * n)).real
#
#         return v
#
#
# @cuda.jit('void(f4,f4[:])', device=True, inline=True)
# def __dRg(n, dg):
#     # sum(a*sqrt(g/pi)*exp(-gn)) -> sum(a*exp(-g*n))
#     dg[0] = 0
#     dg[1] = 0
#     dg[2] = 0
#     if n <= 1:
#         a = complex(7.238054e-01, -1.226411e+00)
#         g = 4.5*complex(3.572318e+00, -2.101695e+01)
#         AexpG = a * comp_exp(-g * n)
#         dg[0] += AexpG.real
#         AexpG *= g
#         dg[1] -= AexpG.real
#         AexpG *= g
#         dg[2] += AexpG.real
#         a = complex(-7.482410e+01, 7.694003e+01)
#         g = 4.5*complex(8.079788e+00, -1.740200e+01)
#         AexpG = a * comp_exp(-g * n)
#         dg[0] += AexpG.real
#         AexpG *= g
#         dg[1] -= AexpG.real
#         AexpG *= g
#         dg[2] += AexpG.real
#         a = complex(1.346014e+03, -1.435227e+03)
#         g = 4.5*complex(1.107646e+01, -1.408444e+01)
#         AexpG = a * comp_exp(-g * n)
#         dg[0] += AexpG.real
#         AexpG *= g
#         dg[1] -= AexpG.real
#         AexpG *= g
#         dg[2] += AexpG.real
#         a = complex(-9.149758e+03, 1.234362e+04)
#         g = 4.5*complex(1.318472e+01, -1.088592e+01)
#         AexpG = a * comp_exp(-g * n)
#         dg[0] += AexpG.real
#         AexpG *= g
#         dg[1] -= AexpG.real
#         AexpG *= g
#         dg[2] += AexpG.real
#         a = complex(2.929669e+04, -5.669524e+04)
#         g = 4.5*complex(1.464249e+01, -7.746260e+00)
#         AexpG = a * comp_exp(-g * n)
#         dg[0] += AexpG.real
#         AexpG *= g
#         dg[1] -= AexpG.real
#         AexpG *= g
#         dg[2] += AexpG.real
#         a = complex(2.409383e+04, -2.429935e+05)
#         g = 4.5*complex(1.601469e+01, -1.544321e+00)
#         AexpG = a * comp_exp(-g * n)
#         dg[0] += AexpG.real
#         AexpG *= g
#         dg[1] -= AexpG.real
#         AexpG *= g
#         dg[2] += AexpG.real
#         a = complex(-4.551067e+04, 1.506981e+05)
#         g = 4.5*complex(1.556592e+01, -4.637619e+00)
#         AexpG = a * comp_exp(-g * n)
#         dg[0] += AexpG.real
#         AexpG *= g
#         dg[1] -= AexpG.real
#         AexpG *= g
#         dg[2] += AexpG.real

# @cuda.jit('f4(f4)', device=True, inline=True)
# def __g(n):
#     if n >= 1:
#         return 0
#     else:
#         tmp = 1 - n
#         return tmp * tmp * (15 / 16)
#
#
# @cuda.jit('f4(f4)', device=True, inline=True)
# def __Rg(n):
#     if n >= 1:
#         return 0
#     else:
#         return (1 - n) ** 3.5
#
#
# @cuda.jit('void(f4,f4[:])', device=True, inline=True)
# def __dRg(n, dg):
#     if n >= 1:
#         dg[0] = 0
#         dg[1] = 0
#         dg[2] = 0
#     else:
#         # (1-n)^3.5, 3.5(1-n)^2.5, 8.75*(1-n)^1.5
#         tmp0 = 1 - n
#         tmp1 = tmp0 * c_sqrt(tmp0)
#         dg[2] = 8.75 * tmp1
#         tmp1 *= tmp0
#         dg[1] = 3.5 * tmp1
#         dg[0] = tmp0 * tmp1


# @cuda.jit('f4(f4,f4[:],f4[:],f4[:])', device=True, inline=True)
@cuda.jit(device=True, inline=True)
def __f(I, x, r, y):
    iso = (r.shape[0] == 1)
    Y, rY = cuda.local.array((DIM,), f4), cuda.local.array((DIM,), f4)

    for i in range(DIM):
        Y[i] = y[i] - x[i]
    if iso:
        for i in range(DIM):
            rY[i] = r[0] * Y[i]
    else:
        rY[0] = r[0] * Y[0] + r[3] * Y[1] + r[5] * Y[2]
        rY[1] = r[1] * Y[1] + r[4] * Y[2]
        rY[2] = r[2] * Y[2]

    n = 0
    for i in range(DIM):
        n += rY[i] * rY[i]

    return I * __g(n)


# @cuda.jit('f4(f4,f4[:],f4[:],f4[:], f4[:])', device=True, inline=True)
@cuda.jit(device=True, inline=True)
def __Rf(I, x, r, y, T):
    Y, rY, rT, MY = cuda.local.array((DIM,), f4), cuda.local.array(
        (DIM,), f4), cuda.local.array((DIM,), f4), cuda.local.array((DIM,), f4)
    lenT = T.shape[0]

    for i in range(DIM):
        Y[i] = y[i] - x[i]
    if r.shape[0] == 1:
        for i in range(DIM):
            rY[i] = r[0] * Y[i]

        rT[0] = r[0] * T[0]
        rT[1] = r[0] * T[1]
        if lenT == 2:
            rT[2] = 0
        else:
            rT[2] = r[0] * T[2]
    else:
        rY[0] = r[0] * Y[0] + r[3] * Y[1] + r[5] * Y[2]
        rY[1] = r[1] * Y[1] + r[4] * Y[2]
        rY[2] = r[2] * Y[2]

        if lenT == 2:
            rT[0] = r[0] * T[0] + r[3] * T[1]
            rT[1] = r[1] * T[1]
            rT[2] = 0
        else:
            rT[0] = r[0] * T[0] + r[3] * T[1] + r[5] * T[2]
            rT[1] = r[1] * T[1] + r[4] * T[2]
            rT[2] = r[2] * T[2]

    # Normalise rT:
    n = 0
    for i in range(lenT):
        n += rT[i] * rT[i]
    n = 1 / c_sqrt(n)
    for i in range(lenT):
        rT[i] *= n

    # Final vector:
    IP = 0
    for i in range(lenT):
        IP += rT[i] * rY[i]
    m = 0
    for i in range(DIM):
        MY[i] = rY[i] - rT[i] * IP
        m += MY[i] * MY[i]

    return I * __Rg(m) * n


# @cuda.jit('void(f4,f4[:],f4[:],f4[:],f4[:], f4[:],f4[:],f4[:,:],i4)', device=True, inline=True)
@cuda.jit(device=True, inline=True)
def __dRf_aniso(I, x, r, y, tt, R, dR, ddR, order):
    # __Rf = I*__Rg(|M(y-x)|^2)/|rT|
    Y, rY, rT, MY = cuda.local.array((DIM,), f4), cuda.local.array(
        (DIM,), f4), cuda.local.array((DIM,), f4), cuda.local.array((DIM,), f4)
    index = cuda.local.array((6, 2), dtype=i4)
    index[0, 0], index[0, 1] = 0, 0
    index[1, 0], index[1, 1] = 1, 1
    index[2, 0], index[2, 1] = 2, 2
    index[3, 0], index[3, 1] = 0, 1
    index[4, 0], index[4, 1] = 1, 2
    index[5, 0], index[5, 1] = 0, 2
    lenT = tt.shape[0]

    for i in range(DIM):
        Y[i] = y[i] - x[i]
    rY[0] = r[0] * Y[0] + r[3] * Y[1] + r[5] * Y[2]
    rY[1] = r[1] * Y[1] + r[4] * Y[2]
    rY[2] = r[2] * Y[2]

    T = cuda.local.array((DIM,), f4)
    T[0], T[1] = tt[0], tt[1]
    if lenT == 2:
        T[2] = 0
        rT[0] = r[0] * T[0] + r[3] * T[1]
        rT[1] = r[1] * T[1]
        rT[2] = 0
    else:
        T[2] = tt[2]
        rT[0] = r[0] * T[0] + r[3] * T[1] + r[5] * T[2]
        rT[1] = r[1] * T[1] + r[4] * T[2]
        rT[2] = r[2] * T[2]

    # Normalise rT:
    n = 0
    for i in range(lenT):
        n += rT[i] * rT[i]
    n = 1 / c_sqrt(n)
    for i in range(lenT):
        rT[i] *= n

    # Final vector:
    IP = 0
    for i in range(lenT):
        IP += rT[i] * rY[i]
    m = 0
    for i in range(DIM):
        MY[i] = rY[i] - rT[i] * IP
        m += MY[i] * MY[i]

    # Preliminary derivatives:
    dMydT = cuda.local.array((DIM, DIM), f4)
    # Missing a factor of n
    for i in range(DIM):
        for j in range(DIM):
            dMydT[i, j] = rT[i] * (IP * rT[j] - MY[j])
        dMydT[i, i] -= IP
    dg, dJdy, dJdT = cuda.local.array((3,), f4), cuda.local.array(
        (DIM,), f4), cuda.local.array((DIM,), f4)
    __dRg(m, dg)

    if (dg[0] == 0) and (dg[1] == 0) and (dg[2] == 0):
        R[0] = 0
        for i in range(10):
            dR[i] = 0
            for j in range(10):
                ddR[i, j] = 0

    else:
        tmp = dg[1] * 2 * n
        for i in range(DIM):
            dJdy[i] = tmp * MY[i]
        tmp = -n * n
        for i in range(DIM):
            dJdT[i] = tmp * (2 * IP * MY[i] * dg[1] + rT[i] * dg[0])

        ddJdyy, ddJdyT, ddJdTT = cuda.local.array((DIM, DIM), f4), cuda.local.array(
            (DIM, DIM), f4), cuda.local.array((DIM, DIM), f4)

        if order > 1:
            tmp = 2 * n
            for i in range(DIM):
                for j in range(i):
                    ddJdyy[i, j] = ddJdyy[j, i]
                ddJdyy[i, i] = tmp * (2 * MY[i] * MY[i] * dg[2] + 
                                      (1 - rT[i] * rT[i]) * dg[1])
                for j in range(i + 1, DIM):
                    ddJdyy[i, j] = tmp * (2 * MY[i] * MY[j] * 
                                          dg[2] - rT[i] * rT[j] * dg[1])
            tmp = 2 * n * n
            for i in range(DIM):
                for j in range(DIM):
                    ddJdyT[i, j] = tmp * ((dMydT[i, j] - MY[i] * rT[j]) * dg[1]
                                          -2 * IP * MY[i] * MY[j] * dg[2])
            tmp = 2 * n * n * n
            for i in range(DIM):
                for j in range(i):
                    ddJdTT[i, j] = ddJdTT[j, i]
                ddJdTT[i, i] = tmp * (
                    (1.5 * rT[i] * rT[i] - .5) * dg[0]
                    +(2 * IP * MY[i] * rT[i] + IP * rT[i] * MY[i] - 
                       IP * dMydT[i, i] - MY[i] * MY[i]) * dg[1]
                    +2 * IP * IP * MY[i] * MY[i] * dg[2])
                for j in range(i + 1, DIM):
                    ddJdTT[i, j] = tmp * (
                        1.5 * rT[i] * rT[j] * dg[0]
                        +(2 * IP * MY[i] * rT[j] + IP * rT[i] * MY[j] - 
                           IP * dMydT[i, j] - MY[i] * MY[j]) * dg[1]
                        +2 * IP * IP * MY[i] * MY[j] * dg[2])

        # Fill in values:
        R[0] = I * dg[0] * n

        # dI
        dR[0] = dg[0] * n
        ddR[0, 0] = 0
        # dIdx
        # ddR[0, 1 + i] = - r_{j,i}dJdy[j]
        ddR[0, 1 + 0] = -r[0] * dJdy[0]
        ddR[0, 1 + 1] = -(r[3] * dJdy[0] + r[1] * dJdy[1])
        ddR[0, 1 + 2] = -(r[5] * dJdy[0] + r[4] * dJdy[1] + r[2] * dJdy[2])
        # dIdr
        # ddR[0, 4 + (I, i)] = dJdy[I] * Y[i] + dJdT[I] * T[i]
        for i in range(6):
            i0, i1 = index[i]
            ddR[0, 4 + i] = dJdy[i0] * Y[i1] + dJdT[i0] * T[i1]
        # dr, dx
        for i in range(9):
            dR[1 + i] = I * ddR[0, 1 + i]

        if order > 1:
            #     d^2r
            #     ddR[1+(I,i), 1+(J,j)] = I*( ddJdyy[I,J]Y[i]Y[j] + ddJdYT[I,J]Y[i]T[j]
            #                               + ddJdYT[J,I]Y[j]T[i] + ddJdTT[I,J]T[i]T[j])
            for i in range(6):
                i0, i1 = index[i]
                for j in range(i, 6):
                    j0, j1 = index[j]
                    ddR[4 + i, 4 + j] = I * (ddJdyy[i0, j0] * Y[i1] * Y[j1]
                                             +ddJdyT[i0, j0] * Y[i1] * T[j1]
                                             +ddJdyT[j0, i0] * Y[j1] * T[i1]
                                             +ddJdTT[i0, j0] * T[i1] * T[j1]
                                             )
            # dxdr
            # ddR[1+j, 4+(I,i)] = -I(ddJdyy[I,k]Y[i]r[k,j] + ddJdYT[k,I]T[i]r[k,j]
            #                             + dJdY[I](i==j))
            # 0 -> 0,0
            # 1 -> 1,1
            # 2 -> 2,2
            # 3 -> 0,1
            # 4 -> 1,2
            # 5 -> 0,2
            for i in range(6):
                i0, i1 = index[i]
                ddR[1 + 0, 4 + i] = -I * (
                    (ddJdyy[i0, 0] * Y[i1] + ddJdyT[0, i0] * T[i1]) * r[0]
                )
                ddR[1 + 1, 4 + i] = -I * (
                    (ddJdyy[i0, 1] * Y[i1] + ddJdyT[1, i0] * T[i1]) * r[1]
                    +(ddJdyy[i0, 0] * Y[i1] + ddJdyT[0, i0] * T[i1]) * r[3]
                )
                ddR[1 + 2, 4 + i] = -I * (
                    (ddJdyy[i0, 2] * Y[i1] + ddJdyT[2, i0] * T[i1]) * r[2]
                    +(ddJdyy[i0, 1] * Y[i1] + ddJdyT[1, i0] * T[i1]) * r[4]
                    +(ddJdyy[i0, 0] * Y[i1] + ddJdyT[0, i0] * T[i1]) * r[5]
                )
                # j == i1
                ddR[1 + i1, 4 + i] -= I * dJdy[i0]
            # d^2x
            # ddR[1+i,1+j] = I(ddJdYY[I,J]r[I,i]r[J,j])
            ddR[1 + 0, 1 + 0] = I * (
                ddJdyy[0, 0] * r[0] * r[0]
            )
            ddR[1 + 0, 1 + 1] = I * (
                ddJdyy[0, 1] * r[0] * r[1]
                +ddJdyy[0, 0] * r[0] * r[3]
            )
            ddR[1 + 0, 1 + 2] = I * (
                ddJdyy[0, 2] * r[0] * r[2]
                +ddJdyy[0, 1] * r[0] * r[4]
                +ddJdyy[0, 0] * r[0] * r[5]
            )

            ddR[1 + 1, 1 + 1] = I * (
                ddJdyy[1, 1] * r[1] * r[1]
                +ddJdyy[1, 0] * r[1] * r[3]
                +ddJdyy[0, 1] * r[3] * r[1]
                +ddJdyy[0, 0] * r[3] * r[3]
            )
            ddR[1 + 1, 1 + 2] = I * (
                ddJdyy[1, 2] * r[1] * r[2]
                +ddJdyy[1, 1] * r[1] * r[4]
                +ddJdyy[1, 0] * r[1] * r[5]
                +ddJdyy[0, 2] * r[3] * r[2]
                +ddJdyy[0, 1] * r[3] * r[4]
                +ddJdyy[0, 0] * r[3] * r[5]
            )

            ddR[1 + 2, 1 + 2] = I * (
                ddJdyy[2, 2] * r[2] * r[2]
                +ddJdyy[2, 1] * r[2] * r[4]
                +ddJdyy[2, 0] * r[2] * r[5]
                +ddJdyy[1, 2] * r[4] * r[2]
                +ddJdyy[1, 1] * r[4] * r[4]
                +ddJdyy[1, 0] * r[4] * r[5]
                +ddJdyy[0, 2] * r[5] * r[2]
                +ddJdyy[0, 1] * r[5] * r[4]
                +ddJdyy[0, 0] * r[5] * r[5]
            )

            # Symmetrise the Hessian
            for i in range(10):
                for j in range(i):
                    ddR[i, j] = ddR[j, i]


# @cuda.jit('void(f4,f4[:],f4[:],f4[:],f4[:], f4[:],f4[:],f4[:,:],i4)', device=True, inline=True)
@cuda.jit(device=True, inline=True)
def __dRf_iso(I, x, r, y, tt, R, dR, ddR, order):
    # __Rf = I*__Rg(|M(y-x)|^2)/|rT|
    Y, rY, MY = cuda.local.array((DIM,), f4), cuda.local.array(
        (DIM,), f4), cuda.local.array((DIM,), f4)
    lenT = tt.shape[0]

    for i in range(DIM):
        Y[i] = y[i] - x[i]
        rY[i] = r[0] * Y[i]

    T = cuda.local.array((DIM,), f4)
    T[0], T[1] = tt[0], tt[1]
    if lenT == 2:
        T[2] = 0
    else:
        T[2] = tt[2]

    # Normalise rT:
    # rT/|rT| = T
    n = 1 / r[0]
    rT = T

    # Final vector:
    IP = 0
    for i in range(lenT):
        IP += rT[i] * rY[i]
    m = 0
    for i in range(DIM):
        MY[i] = rY[i] - rT[i] * IP
        m += MY[i] * MY[i]

    # Preliminary derivatives:
    dMydT = cuda.local.array((DIM, DIM), f4)
    # Missing a factor of n
    for i in range(DIM):
        for j in range(DIM):
            dMydT[i, j] = rT[i] * (IP * rT[j] - MY[j])
        dMydT[i, i] -= IP
    dg, dJdy, dJdT = cuda.local.array((DIM,), f4), cuda.local.array(
        (DIM,), f4), cuda.local.array((DIM,), f4)
    __dRg(m, dg)

    if (dg[0] == 0) and (dg[1] == 0) and (dg[2] == 0):
        R[0] = 0
        for i in range(10):
            dR[i] = 0
            for j in range(10):
                ddR[i, j] = 0

    else:
        tmp = dg[1] * 2 * n
        for i in range(DIM):
            dJdy[i] = tmp * MY[i]
        tmp = -n * n
        for i in range(DIM):
            dJdT[i] = tmp * (2 * IP * MY[i] * dg[1] + rT[i] * dg[0])

        ddJdyy, ddJdyT, ddJdTT = cuda.local.array((DIM, DIM), f4), cuda.local.array(
            (DIM, DIM), f4), cuda.local.array((DIM, DIM), f4)

        if order > 1:
            tmp = 2 * n
            for i in range(DIM):
                for j in range(i):
                    ddJdyy[i, j] = ddJdyy[j, i]
                ddJdyy[i, i] = tmp * (2 * MY[i] * MY[i] * dg[2] + 
                                      (1 - rT[i] * rT[i]) * dg[1])
                for j in range(i + 1, DIM):
                    ddJdyy[i, j] = tmp * (2 * MY[i] * MY[j] * 
                                          dg[2] - rT[i] * rT[j] * dg[1])
            tmp = 2 * n * n
            for i in range(DIM):
                for j in range(DIM):
                    ddJdyT[i, j] = tmp * ((dMydT[i, j] - MY[i] * rT[j]) * dg[1]
                                          -2 * IP * MY[i] * MY[j] * dg[2])
            tmp = 2 * n * n * n
            for i in range(DIM):
                for j in range(i):
                    ddJdTT[i, j] = ddJdTT[j, i]
                ddJdTT[i, i] = tmp * (
                    (1.5 * rT[i] * rT[i] - .5) * dg[0]
                    +(2 * IP * MY[i] * rT[i] + IP * rT[i] * MY[i] - 
                       IP * dMydT[i, i] - MY[i] * MY[i]) * dg[1]
                    +2 * IP * IP * MY[i] * MY[i] * dg[2])
                for j in range(i + 1, DIM):
                    ddJdTT[i, j] = tmp * (
                        1.5 * rT[i] * rT[j] * dg[0]
                        +(2 * IP * MY[i] * rT[j] + IP * rT[i] * MY[j] - 
                           IP * dMydT[i, j] - MY[i] * MY[j]) * dg[1]
                        +2 * IP * IP * MY[i] * MY[j] * dg[2])

        # Fill in values:
        R[0] = I * dg[0] * n

        # dI
        dR[0] = dg[0] * n
        ddR[0, 0] = 0
        # dIdx
        # ddR[0, 1 + i] = - r_{j,i}dJdy[j]
        ddR[0, 1 + 0] = -r[0] * dJdy[0]
        ddR[0, 1 + 1] = -r[0] * dJdy[1]
        ddR[0, 1 + 2] = -r[0] * dJdy[2]
        # dIdr
        # ddR[0, 4] = dJdy[i] * Y[i] + dJdT[i] * T[i]
        ddR[0, 4] = 0
        for i in range(3):
            ddR[0, 4] += dJdy[i] * Y[i] + dJdT[i] * T[i]
        # dr, dx
        for i in range(4):
            dR[1 + i] = I * ddR[0, 1 + i]

        if order > 1:
            #     d^2r
            #     ddR[4, 4] = I*( ddJdyy[i,i]Y[i]Y[i] + ddJdYT[i,i]Y[i]T[i]
            #                               + ddJdYT[i,i]Y[i]T[i] + ddJdTT[i,i]T[i]T[i])
            ddR[4, 4] = 0
            for i in range(3):
                ddR[4, 4] += I * (ddJdyy[i, i] * Y[i] * Y[i]
                                  +ddJdyT[i, i] * Y[i] * T[i]
                                  +ddJdyT[i, i] * Y[i] * T[i]
                                  +ddJdTT[i, i] * T[i] * T[i]
                                  )
            # dxdr
            # ddR[1+j, 4] = -I(ddJdyy[i,j]Y[i]r[j,j] + ddJdYT[j,i]T[i]r[j,j]
            #                             + dJdY[i](i==j))
            for j in range(3):
                ddR[1 + j, 4] = -I * r[0] * \
                    (ddJdyy[0, j] * Y[0] + ddJdyT[j, 0] * T[0] + 
                     ddJdyy[1, j] * Y[1] + ddJdyT[j, 1] * T[1] + 
                     ddJdyy[2, j] * Y[2] + ddJdyT[j, 2] * T[2])

            # d^2x
            # ddR[1+i,1+j] = I(ddJdYY[i,j]r[i,i]r[j,j])
            for i in range(3):
                for j in range(i, 3):
                    ddR[1 + i, 1 + j] = I * r[0] * r[0] * ddJdyy[i, j]

            # Symmetrise the Hessian
            for i in range(10):
                for j in range(i):
                    ddR[i, j] = ddR[j, i]


# @cuda.jit('void(f4,f4,f4[:,:],f4[:,:],i4,f4[:])', device=True, inline=True)
@cuda.jit(device=True, inline=True)
def __tovec3D(p0, p1, w0, w1, i, Y):
    Y[0] = p0 * w0[i, 0] + p1 * w1[i, 0]
    Y[1] = p0 * w0[i, 1] + p1 * w1[i, 1]
    Y[2] = p0 * w0[i, 2] + p1 * w1[i, 2]


# @cuda.jit('void(f4,f4,f4[:,:],f4[:,:],i4,f4[:])', device=True, inline=True)
@cuda.jit(device=True, inline=True)
def __tovec2p5D(p0, p1, w0, w1, i, Y):
    Y[0] = p0 * w0[i, 0]
    Y[1] = p0 * w0[i, 1]
    Y[2] = p1


def VolProj(atom, y, u):
    grid = y[0].size, y[1].size, y[2].size
    tpb = 4
    __VolProj[tuple(-(-g // tpb) for g in grid), (tpb, tpb, tpb)
              ](atom.I, atom.x, atom.r, *y, u)


# @cuda.jit('void(f4[:],f4[:,:],f4[:,:],f4[:],f4[:],f4[:],f4[:,:,:])', inline=True)
@cuda.jit(inline=True)
def __VolProj(I, x, r, y0, y1, y2, u):
    jj, kk, ll = cuda.grid(DIM)
    if jj >= y0.shape[0] or kk >= y1.shape[0] or ll >= y2.shape[0]:
        return
    y = cuda.local.array((DIM,), f4)
    y[0] = y0[jj]
    y[1] = y1[kk]
    y[2] = y2[ll]
    tmp = 0
    for ii in range(x.shape[0]):
        tmp += __f(I[ii], x[ii], r[ii], y)
    u[jj, kk, ll] = tmp


def RadProj(atom, Rad, R):
    S = Rad.range
    t, w, p = S.orientations, S.ortho, S.detector

    grid = w[0].shape[0], p[0].shape[0], p[1].shape[0]
    tpb = 4
    if len(w) == 1:
        w = (w[0], empty((1, 1), dtype='f4'))
        __RadProj_2p5D[tuple(-(-g // tpb) for g in grid),
                       (tpb, tpb, tpb)](atom.I, atom.x, atom.r, t, w[0], w[1], p[0], p[1], R)
    else:
        __RadProj_3D[tuple(-(-g // tpb) for g in grid),
                     (tpb, tpb, tpb)](atom.I, atom.x, atom.r, t, w[0], w[1], p[0], p[1], R)


# @cuda.jit('void(f4[:],f4[:,:],f4[:,:],f4[:,:],f4[:,:],f4[:,:],f4[:],f4[:],f4[:,:,:])', inline=True)
@cuda.jit(inline=True)
def __RadProj_3D(I, x, r, t, w0, w1, p0, p1, R):
    jj, k0, k1 = cuda.grid(DIM)
    if jj >= t.shape[0] or k0 >= p0.shape[0] or k1 >= p1.shape[0]:
        return
    y = cuda.local.array((DIM,), f4)
    __tovec3D(p0[k0], p1[k1], w0, w1, jj, y)
    T = t[jj]
    tmp = 0
    for ii in range(x.shape[0]):
        tmp += __Rf(I[ii], x[ii], r[ii], y, T)
    R[jj, k0, k1] = tmp


# @cuda.jit('void(f4[:],f4[:,:],f4[:,:],f4[:,:],f4[:,:],f4[:,:],f4[:],f4[:],f4[:,:,:])', inline=True)
@cuda.jit(inline=True)
def __RadProj_2p5D(I, x, r, t, w0, w1, p0, p1, R):
    jj, k0, k1 = cuda.grid(DIM)
    if jj >= t.shape[0] or k0 >= p0.shape[0] or k1 >= p1.shape[0]:
        return
    y = cuda.local.array((DIM,), f4)
    __tovec2p5D(p0[k0], p1[k1], w0, w1, jj, y)
    T = t[jj]
    tmp = 0
    for ii in range(x.shape[0]):
        tmp += __Rf(I[ii], x[ii], r[ii], y, T)
    R[jj, k0, k1] = tmp


def L2derivs_RadProj(atom, Rad, C, order=2):
    S = Rad.range
    t, w, p = S.orientations, S.ortho, S.detector
    if hasattr(C, 'asarray'):
        C = C.asarray()
    iso = (atom.r.shape[1] == 1)

    block = w[0].shape[0]
    sz = [w[0].shape[0], p[0].shape[0], p[1].shape[0],
          (w[0].shape[0] * p[0].shape[0] * p[1].shape[0]) // (THREADS * block)]
    if sz[3] * THREADS * block < sz[0] * sz[1] * sz[2]:
        sz[3] += 1

    if iso:
        f, Df, DDf = empty((block,), dtype='f4'), empty(
            (5, block), dtype='f4'), empty((5, 5, block), dtype='f4')
    else:
        f, Df, DDf = empty((block,), dtype='f4'), empty(
            (10, block), dtype='f4'), empty((10, 10, block), dtype='f4')

    if len(w) == 1:
        w = (w[0], empty((1, 1), dtype='f4'))

    if iso:
        __L2derivs_RadProj_iso_3D[block, THREADS](atom.I[0], atom.x[0], atom.r[0], t, w[0], w[1],
                                                p[0], p[1], C, f, Df, DDf, array(sz, dtype='i4'), order)
    else:
        __L2derivs_RadProj_aniso_3D[block, THREADS](atom.I[0], atom.x[0], atom.r[0], t, w[0], w[1],
                                                  p[0], p[1], C, f, Df, DDf, array(sz, dtype='i4'), order)

    return f.sum(axis=-1), Df.sum(axis=-1), DDf.sum(axis=-1)


# @cuda.jit('void(f4,f4[:],f4[:],f4[:,:],f4[:,:],f4[:,:],f4[:],f4[:],f4[:,:,:], f4[:],f4[:,:],f4[:,:,:],i4[:],i4)', inline=True)
@cuda.jit(inline=True)
def __L2derivs_RadProj_aniso_3D(I, x, r, t, w0, w1, p0, p1, C, f, Df, DDf, sz, order):
    '''
    Computes the 0, 1 and 2 order derivatives of |R(I,x,r)-C|^2/2 for a single atom
    '''
    buf = cuda.shared.array((THREADS, 66), f4)
    F = cuda.local.array((1,), f4)
    DF = cuda.local.array((10,), f4)
    DDF = cuda.local.array((10, 10), f4)
    R = cuda.local.array((1,), f4)
    dR = cuda.local.array((10,), f4)
    ddR = cuda.local.array((10, 10), f4)
    Y = cuda.local.array((DIM,), f4)

    is3D = (w0.shape[0] == w1.shape[0])

    block, thread = cuda.blockIdx.x, cuda.threadIdx.x
    cc = cuda.grid(1)

    # Zero fill
    F[0] = 0
    for i in range(10):
        DF[i] = 0
        for j in range(10):
            DDF[i, j] = 0

    for indx in range(sz[3] * cc, sz[3] * (cc + 1)):
        T = indx // (sz[1] * sz[2])
        i0 = (indx // sz[2]) - sz[1] * T
        i1 = indx - sz[2] * (i0 + sz[1] * T)
        if T < sz[0]:
            # Y = p0*w0[T] + p1*w1[T]
            if is3D:
                __tovec3D(p0[i0], p1[i1], w0, w1, T, Y)
            else:
                __tovec2p5D(p0[i0], p1[i1], w0, w1, T, Y)

            # Derivatives of atom
            __dRf_aniso(I, x, r, Y, t[T], R, dR, ddR, order)

            # Derivatives of |R-C|^2/2
            R[0] = R[0] - C[T, i0, i1]
            F[0] += R[0] * R[0]
            for j0 in range(10):
                DF[j0] += R[0] * dR[j0]
                for j1 in range(j0, 10):
                    DDF[j0, j1] += dR[j0] * dR[j1] + R[0] * ddR[j0, j1]

    F[0] *= 0.5
    for i0 in range(DDF.shape[0] - 1):
        for i1 in range(i0 + 1, DDF.shape[0]):
            DDF[i1, i0] = DDF[i0, i1]

    # Sum over threads
    buf[thread, 0] = F[0]
    i = 1
    for i0 in range(10):
        buf[thread, i] = DF[i0]
        i += 1
    for i0 in range(10):
        for i1 in range(i0 + 1):
            buf[thread, i] = DDF[i0, i1]
            i += 1

    __GPU_reduce_n(buf, 66)
    if thread == 0:
        f[block] = buf[0, 0]
        i = 1
        for i0 in range(10):
            Df[i0, block] = buf[0, i]
            i += 1
        for i0 in range(10):
            for i1 in range(i0):
                DDf[i0, i1, block] = buf[0, i]
                DDf[i1, i0, block] = buf[0, i]
                i += 1
            DDf[i0, i0, block] = buf[0, i]
            i += 1


# @cuda.jit('void(f4,f4[:],f4[:],f4[:,:],f4[:,:],f4[:,:],f4[:],f4[:],f4[:,:,:], f4[:],f4[:,:],f4[:,:,:],i4[:],i4)', inline=True)
@cuda.jit(inline=True)
def __L2derivs_RadProj_iso_3D(I, x, r, t, w0, w1, p0, p1, C, f, Df, DDf, sz, order):
    '''
    Computes the 0, 1 and 2 order derivatives of |R(I,x,r)-C|^2/2 for a single atom
    '''
    buf = cuda.shared.array((THREADS, 21), f4)
    F = cuda.local.array((1,), f4)
    DF = cuda.local.array((5,), f4)
    DDF = cuda.local.array((5, 5), f4)
    R = cuda.local.array((1,), f4)
    dR = cuda.local.array((5,), f4)
    ddR = cuda.local.array((5, 5), f4)
    Y = cuda.local.array((DIM,), f4)

    is3D = (w0.shape[0] == w1.shape[0])

    block, thread = cuda.blockIdx.x, cuda.threadIdx.x
    cc = cuda.grid(1)

    # Zero fill
    F[0] = 0
    for i in range(5):
        DF[i] = 0
        for j in range(5):
            DDF[i, j] = 0

    for indx in range(sz[3] * cc, sz[3] * (cc + 1)):
        T = indx // (sz[1] * sz[2])
        i0 = (indx // sz[2]) - sz[1] * T
        i1 = indx - sz[2] * (i0 + sz[1] * T)
        if T < sz[0]:
            # Y = p0*w0[T] + p1*w1[T]
            if is3D:
                __tovec3D(p0[i0], p1[i1], w0, w1, T, Y)
            else:
                __tovec2p5D(p0[i0], p1[i1], w0, w1, T, Y)

            # Derivatives of atom
            __dRf_iso(I, x, r, Y, t[T], R, dR, ddR, order)

            # Derivatives of |R-C|^2/2
            R[0] = R[0] - C[T, i0, i1]
            F[0] += R[0] * R[0]
            for j0 in range(10):
                DF[j0] += R[0] * dR[j0]
                for j1 in range(j0, 10):
                    DDF[j0, j1] += dR[j0] * dR[j1] + R[0] * ddR[j0, j1]

    F[0] *= 0.5
    for i0 in range(DDF.shape[0] - 1):
        for i1 in range(i0 + 1, DDF.shape[0]):
            DDF[i1, i0] = DDF[i0, i1]

    # Sum over threads
    buf[thread, 0] = F[0]
    i = 1
    for i0 in range(5):
        buf[thread, i] = DF[i0]
        i += 1
    for i0 in range(5):
        for i1 in range(i0 + 1):
            buf[thread, i] = DDF[i0, i1]
            i += 1

    __GPU_reduce_n(buf, 21)
    if thread == 0:
        f[block] = buf[0, 0]
        i = 1
        for i0 in range(5):
            Df[i0, block] = buf[0, i]
            i += 1
        for i0 in range(5):
            for i1 in range(i0):
                DDf[i0, i1, block] = buf[0, i]
                DDf[i1, i0, block] = buf[0, i]
                i += 1
            DDf[i0, i0, block] = buf[0, i]
            i += 1


def derivs_RadProj(atom, Rad, C, order=2):
    '''
    Returns:
        R(atom)\cdot C = real number
        dR(atom)/da\cdot C = vector of shape (I,x,r)
        d^2R(atom)/da^2\cdot C = square matrix of shape (I,x,r)
    '''
    S = Rad.range
    t, w, p = S.orientations, S.ortho, S.detector
    if hasattr(C, 'asarray'):
        C = C.asarray()
    iso = atom.space.isotropic

    block = w[0].shape[0]
    sz = [w[0].shape[0], p[0].shape[0], p[1].shape[0],
          (w[0].shape[0] * p[0].shape[0] * p[1].shape[0]) // (THREADS * block)]
    if sz[3] * THREADS * block < sz[0] * sz[1] * sz[2]:
        sz[3] += 1

    if iso:
        f, Df, DDf = empty((block,), dtype='f4'), empty(
            (5, block), dtype='f4'), empty((5, 5, block), dtype='f4')
    else:
        f, Df, DDf = empty((block,), dtype='f4'), empty(
            (10, block), dtype='f4'), empty((10, 10, block), dtype='f4')

    if len(w) == 1:
        w = (w[0], empty((1, 1), dtype='f4'))

    if iso:
        __derivs_RadProj_iso_3D[block, THREADS](atom.I[0], atom.x[0], atom.r[0], t, w[0], w[1],
                                                p[0], p[1], C, f, Df, DDf, array(sz, dtype='i4'), order)
    else:
        __derivs_RadProj_aniso_3D[block, THREADS](atom.I[0], atom.x[0], atom.r[0], t, w[0], w[1],
                                                  p[0], p[1], C, f, Df, DDf, array(sz, dtype='i4'), order)

    return f.sum(axis=-1), Df.sum(axis=-1), DDf.sum(axis=-1)


# @cuda.jit('void(f4,f4[:],f4[:],f4[:,:],f4[:,:],f4[:,:],f4[:],f4[:],f4[:,:,:], f4[:],f4[:,:],f4[:,:,:],i4[:],i4)', inline=True)
@cuda.jit(inline=True)
def __derivs_RadProj_aniso_3D(I, x, r, t, w0, w1, p0, p1, C, f, Df, DDf, sz, order):
    '''
    Computes the 0, 1 and 2 order derivatives of |R(I,x,r)-C|^2/2 for a single atom
    '''
    buf = cuda.shared.array((THREADS, 66), f4)
    F = cuda.local.array((1,), f4)
    DF = cuda.local.array((10,), f4)
    DDF = cuda.local.array((10, 10), f4)
    R = cuda.local.array((1,), f4)
    dR = cuda.local.array((10,), f4)
    ddR = cuda.local.array((10, 10), f4)
    Y = cuda.local.array((DIM,), f4)

    is3D = (w0.shape[0] == w1.shape[0])

    block, thread = cuda.blockIdx.x, cuda.threadIdx.x
    cc = cuda.grid(1)

    # Zero fill
    F[0] = 0
    for i in range(10):
        DF[i] = 0
        for j in range(10):
            DDF[i, j] = 0

    for indx in range(sz[3] * cc, sz[3] * (cc + 1)):
        T = indx // (sz[1] * sz[2])
        i0 = (indx // sz[2]) - sz[1] * T
        i1 = indx - sz[2] * (i0 + sz[1] * T)
        if T < sz[0]:
            # Y = p0*w0[T] + p1*w1[T]
            if is3D:
                __tovec3D(p0[i0], p1[i1], w0, w1, T, Y)
            else:
                __tovec2p5D(p0[i0], p1[i1], w0, w1, T, Y)

            # Derivatives of atom
            __dRf_aniso(I, x, r, Y, t[T], R, dR, ddR, order)
            # Derivatives of R\cdot C
            scale = C[T, i0, i1]
            F[0] += R[0] * scale
            for j0 in range(10):
                DF[j0] += dR[j0] * scale
                for j1 in range(j0, 10):
                    DDF[j0, j1] += ddR[j0, j1] * scale

    for i0 in range(DDF.shape[0] - 1):
        for i1 in range(i0 + 1, DDF.shape[0]):
            DDF[i1, i0] = DDF[i0, i1]

    # Sum over threads
    buf[thread, 0] = F[0]
    i = 1
    for i0 in range(10):
        buf[thread, i] = DF[i0]
        i += 1
    for i0 in range(10):
        for i1 in range(i0 + 1):
            buf[thread, i] = DDF[i0, i1]
            i += 1

    __GPU_reduce_n(buf, 66)
    if thread == 0:
        f[block] = buf[0, 0]
        i = 1
        for i0 in range(10):
            Df[i0, block] = buf[0, i]
            i += 1
        for i0 in range(10):
            for i1 in range(i0):
                DDf[i0, i1, block] = buf[0, i]
                DDf[i1, i0, block] = buf[0, i]
                i += 1
            DDf[i0, i0, block] = buf[0, i]
            i += 1


# @cuda.jit('void(f4,f4[:],f4[:],f4[:,:],f4[:,:],f4[:,:],f4[:],f4[:],f4[:,:,:], f4[:],f4[:,:],f4[:,:,:],i4[:],i4)', inline=True)
@cuda.jit(inline=True)
def __derivs_RadProj_iso_3D(I, x, r, t, w0, w1, p0, p1, C, f, Df, DDf, sz, order):
    '''
    Computes the 0, 1 and 2 order derivatives of |R(I,x,r)-C|^2/2 for a single atom
    '''
    buf = cuda.shared.array((THREADS, 21), f4)
    F = cuda.local.array((1,), f4)
    DF = cuda.local.array((5,), f4)
    DDF = cuda.local.array((5, 5), f4)
    R = cuda.local.array((1,), f4)
    dR = cuda.local.array((5,), f4)
    ddR = cuda.local.array((5, 5), f4)
    Y = cuda.local.array((DIM,), f4)

    is3D = (w0.shape[0] == w1.shape[0])

    block, thread = cuda.blockIdx.x, cuda.threadIdx.x
    cc = cuda.grid(1)

    # Zero fill
    F[0] = 0
    for i in range(5):
        DF[i] = 0
        for j in range(5):
            DDF[i, j] = 0

    for indx in range(sz[3] * cc, sz[3] * (cc + 1)):
        T = indx // (sz[1] * sz[2])
        i0 = (indx // sz[2]) - sz[1] * T
        i1 = indx - sz[2] * (i0 + sz[1] * T)
        if T < sz[0]:
            # Y = p0*w0[T] + p1*w1[T]
            if is3D:
                __tovec3D(p0[i0], p1[i1], w0, w1, T, Y)
            else:
                __tovec2p5D(p0[i0], p1[i1], w0, w1, T, Y)

            # Derivatives of atom
            __dRf_iso(I * C[T, i0, i1], x, r, Y, t[T], R, dR, ddR, order)

            # Derivatives of R\cdot C which has already been pre-multiplied to I
            F[0] += R[0]
            scale = 1 / C[T, i0, i1]
            DF[0] += dR[0] * scale
            for j1 in range(10):
                DDF[0, j1] += ddR[0, j1] * scale
            for j0 in range(1, 10):
                DF[j0] += dR[j0]
                for j1 in range(j0, 10):
                    DDF[j0, j1] += ddR[j0, j1]

    for i0 in range(DDF.shape[0] - 1):
        for i1 in range(i0 + 1, DDF.shape[0]):
            DDF[i1, i0] = DDF[i0, i1]

    # Sum over threads
    buf[thread, 0] = F[0]
    i = 1
    for i0 in range(5):
        buf[thread, i] = DF[i0]
        i += 1
    for i0 in range(5):
        for i1 in range(i0 + 1):
            buf[thread, i] = DDF[i0, i1]
            i += 1

    __GPU_reduce_n(buf, 21)
    if thread == 0:
        f[block] = buf[0, 0]
        i = 1
        for i0 in range(5):
            Df[i0, block] = buf[0, i]
            i += 1
        for i0 in range(5):
            for i1 in range(i0):
                DDf[i0, i1, block] = buf[0, i]
                DDf[i1, i0, block] = buf[0, i]
                i += 1
            DDf[i0, i0, block] = buf[0, i]
            i += 1


if __name__ == '__main__':
    from numpy import array_str, random, concatenate
    from sympy import symbols, exp as s_exp, sqrt as s_sqrt, diff
    random.seed(0)

    def niceprint(thing): return array_str(
        array(thing, 'float64'), precision=3)

    class dict2obj():

        def __init__(self, **extra):
            self.__dict__.update(extra)

    s_I, s_x, s_y, s_r, s_T, s_C = symbols(
        ('I', 'x:3', 'y:3', 'r:6', 'T:3', 'C'), real=True)
    s_Y = [s_y[i] - s_x[i] for i in range(3)]
    s_rY = (s_r[0] * s_Y[0] + s_r[3] * s_Y[1] + s_r[5] * s_Y[2],
            s_r[1] * s_Y[1] + s_r[4] * s_Y[2], s_r[2] * s_Y[2])
    s_rT = (s_r[0] * s_T[0] + s_r[3] * s_T[1] + s_r[5] * s_T[2],
            s_r[1] * s_T[1] + s_r[4] * s_T[2], s_r[2] * s_T[2])
    s_nT = s_sqrt(s_rT[0] ** 2 + s_rT[1] ** 2 + s_rT[2] ** 2)
    s_rYT = (s_rY[0] * s_rT[0] + s_rY[1] * s_rT[1] + s_rY[2] * s_rT[2]) / s_nT
    s_MrY = [s_rY[i] - s_rYT * s_rT[i] / s_nT for i in range(3)]
    s_n = s_MrY[0] ** 2 + s_MrY[1] ** 2 + s_MrY[2] ** 2

    I, x, y, r, T, C = random.rand(1), random.rand(
        1, DIM), random.rand(1, DIM), random.rand(1, 6), random.rand(1, DIM), random.rand(1, 1, 1)
    for thing in 'IxyrTC':
        locals()[thing] = locals()[thing].astype('f4')

    # For testing 2p5d code
    T[0, :] = 1, 1, 0
    y[0, :] = 1, -1, 3

    # # # # # #
    # Test __derivs_aniso
    # # # # # #
    s_Rf = (s_I * s_exp(-s_n / 2) / s_nT - s_C) ** 2 / 2
    var = (s_I,) + s_x + s_r

    s_dRf = [diff(s_Rf, x) for x in var]
    s_ddRf = [[diff(d, x) for x in var] for d in s_dRf]
    var = (s_C, s_I) + s_x + s_y + s_r + s_T
    val = concatenate((C[0, 0], I, x[0], y[0], r[0], T[0]))
    subs = [(var[i], val[i]) for i in range(len(var))]
    s_Rf = s_Rf.subs(subs)
    s_dRf = [d.subs(subs) for d in s_dRf]
    s_ddRf = [[d.subs(subs) for d in dd] for dd in s_ddRf]

    RF, dRF, ddRF = L2derivs_RadProj(
        dict2obj(I=I, x=x, r=r),
        dict2obj(range=dict2obj(orientations=T, ortho=[
                 y, 0 * y], detector=[0 * y[0] + 1, 0 * y[0]])),
        dict2obj(array=C))
#     RF, dRF, ddRF = L2derivs_RadProj(
#         dict2obj(I=I, x=x, r=r),
#         dict2obj(range=dict2obj(orientations=T[:, :2], ortho=[
#                  y[:, :2]], detector=[array([1], dtype='f4'), array([y[0, 2]], dtype='f4')])),
#         dict2obj(array=C))

    print('Printouts for aniso test:')
#     print(niceprint(s_Rf))
#     print(niceprint(RF), '\n')
    print('f error: ', niceprint(abs(s_Rf - RF)), '\n')
#     print(niceprint(s_dRf))
#     print(niceprint(dRF), '\n')
    print('Df error: ', niceprint(abs(array(s_dRf) - dRF).max()), '\n')
#     print(niceprint(s_ddRf), '\n')
#     print(niceprint(ddRF), '\n')
    print('DDf error: ', niceprint(
        abs(array(s_ddRf) - ddRF).max()), '\n')
    for i in range(10):
        for j in range(i, 10):
            if abs(s_ddRf[i][j] - ddRF[i][j]) > 1e-6:
                print('index ', i, j, ' is wrong')
