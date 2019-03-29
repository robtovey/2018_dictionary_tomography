'''
Created on 28 Sep 2018

@author: Rob Tovey
'''
from .NewtonBases import __g, __Rg, __dRg
from numba import cuda, float32 as f4, int32 as i4
from math import exp as c_exp, sqrt as c_sqrt, floor
from cmath import exp as comp_exp, sqrt as comp_sqrt
from .numba_cuda_aux import THREADS, __GPU_reduce_n
from numpy import empty, array
DIM = 2


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
        rY[0] = r[0] * Y[0] + r[2] * Y[1]
        rY[1] = r[1] * Y[1]

    n = 0
    for i in range(DIM):
        n += rY[i] * rY[i]

    return I * __g(n)


# @cuda.jit('f4(f4,f4[:],f4[:],f4[:], f4[:])', device=True, inline=True)
@cuda.jit(device=True, inline=True)
def __Rf(I, x, r, y, T):
    Y, rY, rT, MY = cuda.local.array((DIM,), f4), cuda.local.array(
        (DIM,), f4), cuda.local.array((DIM,), f4), cuda.local.array((DIM,), f4)

    for i in range(DIM):
        Y[i] = y[i] - x[i]
    if r.shape[0] == 1:
        for i in range(DIM):
            rY[i] = r[0] * Y[i]
            rT[i] = r[0] * T[i]
    else:
        rY[0] = r[0] * Y[0] + r[2] * Y[1]
        rY[1] = r[1] * Y[1]
        rT[0] = r[0] * T[0] + r[2] * T[1]
        rT[1] = r[1] * T[1]

    # Normalise rT:
    n = 0
    for i in range(DIM):
        n += rT[i] * rT[i]
    n = 1 / c_sqrt(n)
    for i in range(DIM):
        rT[i] *= n

    # Final vector:
    IP = 0
    for i in range(DIM):
        IP += rT[i] * rY[i]
    m = 0
    for i in range(DIM):
        MY[i] = rY[i] - rT[i] * IP
        m += MY[i] * MY[i]

    return I * __Rg(m) * n


# @cuda.jit('void(f4,f4[:],f4[:],f4[:],f4[:], f4[:],f4[:],f4[:,:],i4)', device=True, inline=True)
@cuda.jit(device=True, inline=True)
def __dRf(I, x, r, y, T, R, dR, ddR, order):
    # __Rf = I*__Rg(|M(y-x)|^2)/|rT|
    Y, rY, rT, MY = cuda.local.array((DIM,), f4), cuda.local.array(
        (DIM,), f4), cuda.local.array((DIM,), f4), cuda.local.array((DIM,), f4)
    index = cuda.local.array((3, 2), dtype=i4)
    index[0, 0], index[0, 1] = 0, 0
    index[1, 0], index[1, 1] = 1, 1
    index[2, 0], index[2, 1] = 0, 1

    for i in range(DIM):
        Y[i] = y[i] - x[i]
    rY[0] = r[0] * Y[0] + r[2] * Y[1]
    rY[1] = r[1] * Y[1]
    rT[0] = r[0] * T[0] + r[2] * T[1]
    rT[1] = r[1] * T[1]

    # Normalise rT:
    n = 0
    for i in range(DIM):
        n += rT[i] * rT[i]
    n = 1 / c_sqrt(n)
    for i in range(DIM):
        rT[i] *= n

    # Final vector:
    IP = 0
    for i in range(DIM):
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
        for i in range(6):
            dR[i] = 0
            for j in range(6):
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
        ddR[0, 1 + 1] = -(r[2] * dJdy[0] + r[1] * dJdy[1])
        # dIdr
        # ddR[0, 3 + (I, i)] = dJdy[I] * Y[i] + dJdT[I] * T[i]
        for i in range(3):
            i0, i1 = index[i]
            ddR[0, 3 + i] = dJdy[i0] * Y[i1] + dJdT[i0] * T[i1]
        # dr, dx
        for i in range(5):
            dR[1 + i] = I * ddR[0, 1 + i]

        if order > 1:
            # d^2r
            # ddR[1+(I,i), 1+(J,j)] = I*( ddJdyy[I,J]Y[i]Y[j] + ddJdYT[I,J]Y[i]T[j]
            #                           + ddJdYT[J,I]Y[j]T[i] + ddJdTT[I,J]T[i]T[j])
            for i in range(3):
                i0, i1 = index[i]
                for j in range(i, 3):
                    j0, j1 = index[j]
                    ddR[3 + i, 3 + j] = I * (ddJdyy[i0, j0] * Y[i1] * Y[j1]
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
            for i in range(3):
                i0, i1 = index[i]
                ddR[1 + 0, 3 + i] = -I * (
                    (ddJdyy[i0, 0] * Y[i1] + ddJdyT[0, i0] * T[i1]) * r[0]
                )
                ddR[1 + 1, 3 + i] = -I * (
                    (ddJdyy[i0, 1] * Y[i1] + ddJdyT[1, i0] * T[i1]) * r[1]
                    +(ddJdyy[i0, 0] * Y[i1] + ddJdyT[0, i0] * T[i1]) * r[2]
                )
                # j == i1
                ddR[1 + i1, 3 + i] -= I * dJdy[i0]
            # d^2x
            # ddR[1+i,1+j] = I(ddJdYY[I,J]r[I,i]r[J,j])
            ddR[1 + 0, 1 + 0] = I * (
                ddJdyy[0, 0] * r[0] * r[0]
            )
            ddR[1 + 0, 1 + 1] = I * (
                ddJdyy[0, 1] * r[0] * r[1]
                +ddJdyy[0, 0] * r[0] * r[2]
            )

            ddR[1 + 1, 1 + 1] = I * (
                ddJdyy[1, 1] * r[1] * r[1]
                +ddJdyy[1, 0] * r[1] * r[2]
                +ddJdyy[0, 1] * r[2] * r[1]
                +ddJdyy[0, 0] * r[2] * r[2]
            )

            # Symmetrise the Hessian
            for i in range(6):
                for j in range(i):
                    ddR[i, j] = ddR[j, i]


# @cuda.jit('void(f4,f4[:,:],i4,f4[:])', device=True, inline=True)
@cuda.jit(device=True, inline=True)
def __tovec(p0, w0, i, Y):
    Y[0] = p0 * w0[i, 0]
    Y[1] = p0 * w0[i, 1]


def VolProj(atom, y, u):
    grid = y[0].size, y[1].size
    tpb = 4
    __VolProj[tuple(-(-g // tpb) for g in grid), (tpb, tpb)
              ](atom.I, atom.x, __to_aniso(atom.r), *y, u)


# @cuda.jit('void(f4[:],f4[:,:],f4[:,:],f4[:],f4[:],f4[:,:])', inline=True)
@cuda.jit(inline=True)
def __VolProj(I, x, r, y0, y1, u):
    jj, kk = cuda.grid(DIM)
    if jj >= y0.shape[0] or kk >= y1.shape[0]:
        return
    y = cuda.local.array((DIM,), f4)
    y[0] = y0[jj]
    y[1] = y1[kk]
    tmp = 0
    for ii in range(x.shape[0]):
        tmp += __f(I[ii], x[ii], r[ii], y)
    u[jj, kk] = tmp


def RadProj(atom, Rad, R):
    S = Rad.range
    t, w, p = S.orientations, S.ortho, S.detector

    grid = w[0].shape[0], p[0].shape[0]
    tpb = 4
    __RadProj[tuple(-(-g // tpb) for g in grid),
              (tpb, tpb)](atom.I, atom.x, __to_aniso(atom.r), t, w[0], p[0], R)


# @cuda.jit('void(f4[:],f4[:,:],f4[:,:],f4[:,:],f4[:,:],f4[:],f4[:,:])', inline=True)
@cuda.jit(inline=True)
def __RadProj(I, x, r, t, w0, p0, R):
    jj, k0 = cuda.grid(DIM)
    if jj >= t.shape[0] or k0 >= p0.shape[0]:
        return
    y = cuda.local.array((DIM,), f4)
    __tovec(p0[k0], w0, jj, y)
    T = t[jj]
    tmp = 0
    for ii in range(x.shape[0]):
        tmp += __Rf(I[ii], x[ii], r[ii], y, T)
    R[jj, k0] = tmp


def L2derivs_RadProj(atom, Rad, C, order=2):
    S = Rad.range
    t, w, p = S.orientations, S.ortho, S.detector
    if hasattr(C, 'asarray'):
        C = C.asarray()

    block = w[0].shape[0]
    sz = [w[0].shape[0], p[0].shape[0],
          (w[0].shape[0] * p[0].shape[0]) // (THREADS * block)]
    if sz[2] * THREADS * block < sz[0] * sz[1]:
        sz[2] += 1

    f, Df, DDf = empty((block,), dtype='f4'), empty(
        (6, block), dtype='f4'), empty((6, 6, block), dtype='f4')

    __L2derivs_RadProj_aniso[block, THREADS](atom.I[0], atom.x[0], __to_aniso(atom.r)[0], t, w[0],
                                           p[0], C, f, Df, DDf,
                                           array(sz, dtype='i4'), order)
    f, Df, DDf = f.sum(axis=-1), Df.sum(axis=-1), DDf.sum(axis=-1)
    
    # TODO: Check this
    if atom.space.isotropic:
        Df[3] = Df[:, 3:5].sum(-1)
        DDf[3, :3] = DDf[3:5, :3]
        DDf[:3, 3] = DDf[:3, 3:5]
        Df, DDf = Df[:4], DDf[:4, :4]

    return f, Df, DDf


# @cuda.jit('void(f4,f4[:],f4[:],f4[:,:],f4[:,:],f4[:],f4[:,:], f4[:],f4[:,:],f4[:,:,:],i4[:],i4)', inline=True)
@cuda.jit(inline=True)
def __L2derivs_RadProj_aniso(I, x, r, t, w0, p0, C, f, Df, DDf, sz, order):
    '''
    Computes the 0, 1 and 2 order derivatives of |R(I,x,r)-C|^2/2 for a single atom
    '''
    buf = cuda.shared.array((THREADS, 28), f4)
    F = cuda.local.array((1,), f4)
    DF = cuda.local.array((6,), f4)
    DDF = cuda.local.array((6, 6), f4)
    R = cuda.local.array((1,), f4)
    dR = cuda.local.array((6,), f4)
    ddR = cuda.local.array((6, 6), f4)
    Y = cuda.local.array((DIM,), f4)

    block, thread = cuda.blockIdx.x, cuda.threadIdx.x
    cc = cuda.grid(1)

    # Zero fill
    F[0] = 0
    for i in range(6):
        DF[i] = 0
        for j in range(6):
            DDF[i, j] = 0

    for indx in range(sz[2] * cc, sz[2] * (cc + 1)):
        T = indx // sz[1]
        i0 = indx - sz[1] * T
        if T < sz[0]:
            # Y = p0*w0[T]
            __tovec(p0[i0], w0, T, Y)

            # Derivatives of atom
            __dRf(I, x, r, Y, t[T], R, dR, ddR, order)

            # Derivatives of |R-C|^2/2
            R[0] = R[0] - C[T, i0]
            F[0] += R[0] * R[0]
            for j0 in range(6):
                DF[j0] += R[0] * dR[j0]
                for j1 in range(j0, 6):
                    DDF[j0, j1] += dR[j0] * dR[j1] + R[0] * ddR[j0, j1]

    F[0] *= 0.5
    for i0 in range(DDF.shape[0] - 1):
        for i1 in range(i0 + 1, DDF.shape[0]):
            DDF[i1, i0] = DDF[i0, i1]

    # Sum over threads
    buf[thread, 0] = F[0]
    i = 1
    for i0 in range(6):
        buf[thread, i] = DF[i0]
        i += 1
    for i0 in range(6):
        for i1 in range(i0 + 1):
            buf[thread, i] = DDF[i0, i1]
            i += 1

    __GPU_reduce_n(buf, 28)
    if thread == 0:
        f[block] = buf[0, 0]
        i = 1
        for i0 in range(6):
            Df[i0, block] = buf[0, i]
            i += 1
        for i0 in range(6):
            for i1 in range(i0):
                DDf[i0, i1, block] = buf[0, i]
                DDf[i1, i0, block] = buf[0, i]
                i += 1
            DDf[i0, i0, block] = buf[0, i]
            i += 1


def __to_aniso(r):
    if r.shape[1] == 1:
        R = empty((r.shape[0], 1), dtype=r.dtype)
        R[:, 0] = r
        R[:, 1] = r
        R[:, 2] = 0
        return R
    else:
        return r


def derivs_RadProj(atom, Rad, C, order=2):
    S = Rad.range
    t, w, p = S.orientations, S.ortho, S.detector
    if hasattr(C, 'asarray'):
        C = C.asarray()

    block = w[0].shape[0]
    sz = [w[0].shape[0], p[0].shape[0],
          (w[0].shape[0] * p[0].shape[0]) // (THREADS * block)]
    if sz[2] * THREADS * block < sz[0] * sz[1]:
        sz[2] += 1

    f, Df, DDf = empty((block,), dtype='f4'), empty(
        (6, block), dtype='f4'), empty((6, 6, block), dtype='f4')

    __derivs_RadProj_aniso[block, THREADS](atom.I[0], atom.x[0], __to_aniso(atom.r)[0], t, w[0],
                                           p[0], C, f, Df, DDf,
                                           array(sz, dtype='i4'), order)

    f, Df, DDf = f.sum(axis=-1), Df.sum(axis=-1), DDf.sum(axis=-1)
    if atom.space.isotropic:
        Df[3] = Df[3:5].sum()
        DDf[3, :3] = DDf[3:5, :3].sum(0)
        DDf[:3, 3] = DDf[:3, 3:5].sum(1)
        DDf[3, 3] += DDf[4, 4]
        Df, DDf = Df[:4], DDf[:4, :4]
    return f, Df, DDf


# @cuda.jit('void(f4,f4[:],f4[:],f4[:,:],f4[:,:],f4[:],f4[:,:], f4[:],f4[:,:],f4[:,:,:],i4[:],i4)', inline=True)
@cuda.jit(inline=True)
def __derivs_RadProj_aniso(I, x, r, t, w0, p0, C, f, Df, DDf, sz, order):
    '''
    Computes the 0, 1 and 2 order derivatives of |R(I,x,r)-C|^2/2 for a single atom
    '''
    buf = cuda.shared.array((THREADS, 28), f4)
    F = cuda.local.array((1,), f4)
    DF = cuda.local.array((6,), f4)
    DDF = cuda.local.array((6, 6), f4)
    R = cuda.local.array((1,), f4)
    dR = cuda.local.array((6,), f4)
    ddR = cuda.local.array((6, 6), f4)
    Y = cuda.local.array((DIM,), f4)

    block, thread = cuda.blockIdx.x, cuda.threadIdx.x
    cc = cuda.grid(1)

    # Zero fill
    F[0] = 0
    for i in range(6):
        DF[i] = 0
        for j in range(6):
            DDF[i, j] = 0

    for indx in range(sz[2] * cc, sz[2] * (cc + 1)):
        T = indx // sz[1]
        i0 = indx - sz[1] * T
        if T < sz[0]:
            # Y = p0*w0[T]
            __tovec(p0[i0], w0, T, Y)

            # Derivatives of atom
            __dRf(I, x, r, Y, t[T], R, dR, ddR, order)

            # Derivatives of R\cdot C
            scale = C[T, i0]
            F[0] += R[0] * scale
            for j0 in range(6):
                DF[j0] += dR[j0] * scale
                for j1 in range(j0, 6):
                    DDF[j0, j1] += ddR[j0, j1] * scale

    for i0 in range(DDF.shape[0] - 1):
        for i1 in range(i0 + 1, DDF.shape[0]):
            DDF[i1, i0] = DDF[i0, i1]

    # Sum over threads
    buf[thread, 0] = F[0]
    i = 1
    for i0 in range(6):
        buf[thread, i] = DF[i0]
        i += 1
    for i0 in range(6):
        for i1 in range(i0 + 1):
            buf[thread, i] = DDF[i0, i1]
            i += 1

    __GPU_reduce_n(buf, 28)
    if thread == 0:
        f[block] = buf[0, 0]
        i = 1
        for i0 in range(6):
            Df[i0, block] = buf[0, i]
            i += 1
        for i0 in range(6):
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
        ('I', 'x:2', 'y:2', 'r:3', 'T:2', 'C'), real=True)
    s_Y = [s_y[i] - s_x[i] for i in range(2)]
    s_rY = (s_r[0] * s_Y[0] + s_r[2] * s_Y[1],
            s_r[1] * s_Y[1])
    s_rT = (s_r[0] * s_T[0] + s_r[2] * s_T[1],
            s_r[1] * s_T[1])
    s_nT = s_sqrt(s_rT[0] ** 2 + s_rT[1] ** 2)
    s_rYT = (s_rY[0] * s_rT[0] + s_rY[1] * s_rT[1]) / s_nT
    s_MrY = [s_rY[i] - s_rYT * s_rT[i] / s_nT for i in range(2)]
    s_n = s_MrY[0] ** 2 + s_MrY[1] ** 2

    I, x, y, r, T, C = random.rand(1), random.rand(
        1, DIM), random.rand(1, DIM), random.rand(1, 3), random.rand(1, DIM), random.rand(1, 1)
    for thing in 'IxyrTC':
        locals()[thing] = locals()[thing].astype('f4')

    # # # # # #
    # Test __derivs_aniso
    # # # # # #
    s_Rf = (s_I * s_exp(-s_n / 2) / s_nT - s_C) ** 2 / 2
    var = (s_I,) + s_x + s_r

    s_dRf = [diff(s_Rf, x) for x in var]
    s_ddRf = [[diff(d, x) for x in var] for d in s_dRf]
    var = (s_C, s_I) + s_x + s_y + s_r + s_T
    val = concatenate((C[0], I, x[0], y[0], r[0], T[0]))
    subs = [(var[i], val[i]) for i in range(len(var))]
    s_Rf = s_Rf.subs(subs)
    s_dRf = [d.subs(subs) for d in s_dRf]
    s_ddRf = [[d.subs(subs) for d in dd] for dd in s_ddRf]

    RF, dRF, ddRF = L2derivs_RadProj(
        dict2obj(I=I, x=x, r=r),
        dict2obj(range=dict2obj(orientations=T, ortho=[y],
                                          detector=[0 * y[0] + 1])),
        dict2obj(array=C))

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
    for i in range(6):
        for j in range(i, 6):
            if abs(s_ddRf[i][j] - ddRF[i][j]) > 1e-6:
                print('index ', i, j, ' is wrong')
