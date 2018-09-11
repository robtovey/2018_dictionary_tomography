'''
Created on 2 Jul 2018

@author: Rob Tovey
'''
from numpy import sqrt, empty, array, pi
from numba import cuda, float32 as f4, int32 as i4
from math import exp as c_exp, sqrt as c_sqrt
from .numba_cuda_aux import THREADS, __GPU_reduce_n


def derivs_iso(atom, Rad, C):
    S = Rad.ProjectionSpace
    w0 = S.ortho[0]
    p0, p1 = S.detector

    block = min(w0.shape[0], 1000)
    sz = [w0.shape[0], p0.shape[0], p1.shape[0],
          (w0.shape[0] * p0.shape[0] * p1.shape[0]) // THREADS]
    if sz[3] * THREADS < sz[0] * sz[1] * sz[2]:
        sz[3] += 1

    f, Df, DDf = empty((block,), dtype='f4'), empty(
        (5, block), dtype='f4'), empty((5, 5, block), dtype='f4')

    if len(S.ortho) == 2:
        __derivs_iso[block, THREADS](atom.I, atom.x, atom.r, w0, S.ortho[1], p0,
                                     p1, C.array / sqrt(2 * pi), f, Df, DDf, array(sz, dtype='i4'))
    else:
        __derivs_iso_2p5D[block, THREADS](atom.I, atom.x, atom.r, w0, p0,
                                          p1, C.array / sqrt(2 * pi), f, Df, DDf, array(sz, dtype='i4'))

    return f.sum(axis=-1), Df.sum(axis=-1), DDf.sum(axis=-1)


@cuda.jit("void(f4,f4[:],f4,f4[:],f4[:],f4[:,:])", device=True, inline=True, cache=True)
def __derivs_iso_aux(I, Y, r, g, Dg, DDg):
    '''
    g = I*r*exp(-.5|Y|^2/r^2)
    Y = y-x
    J = -.5|y-x|^2/r^2

    dg/dI = r*exp(J)
    dg/dx = g(dJ/dx)
    dg/dr = I*exp(J) + g(dJ/dr)

    d^2g/dx^2 = g(dJ/dx)^2 + gd^2J/dx^2
    d^2g/dxdr = I*exp(J)(dJ/dx) + g(dJ/dx)(dJ/dr) + gd^2J/dxdr
    d^2g/dr^2 = 2*I*exp(J)*(dJ/dr) + g(dJ/dr)^2 + gd^2J/dr^2

    dJ/dx = -(x-y)/r^2 = (1/r)(Y/r)
    dJ/dr = |y-x|^2/r^3 = (-1/r)2J

    d^2J/dx^2 = -1/r^2
    d^2J/dxdr = (-1/r^2)(2Y/r)
    d^J/dr^2 = -3|Y|^2/r^4 = (1/r^2)6J

    '''

    # Allocate memory:
    rY = cuda.local.array((3,), f4)
    dJ = cuda.local.array((4,), f4)

    rY[0] = Y[0] / r
    rY[1] = Y[1] / r
    rY[2] = Y[2] / r

    J = -0.5 * (rY[0] * rY[0] + rY[1] * rY[1] + rY[2] * rY[2])

    if J < -30:
        g[0] = 0
        for i in range(5):
            Dg[i] = 0
            for j in range(5):
                DDg[i, j] = 0
        return

    G = c_exp(J)

    # # # Zeroth derivative
    g[0] = I * r * G

    # # # First derivative
    Dg[0] = r * G

    # DJ
    tmp = 1 / r
    dJ[0], dJ[1], dJ[2] = tmp * rY[0], tmp * rY[1], tmp * rY[2]
    dJ[3] = -2 * tmp * J

    # Dg/dx
    Dg[1], Dg[2], Dg[3] = g[0] * dJ[0], g[0] * dJ[1], g[0] * dJ[2]

    # Dg/dr
    Dg[4] = I * G + g[0] * dJ[3]

    # # # Second derivative
    DDg[0, 0] = 0

    # ddg/dIdx
    for i in range(3):
        DDg[0, 1 + i] = r * G * dJ[i]
    # ddg/dIdr
    DDg[0, 4] = G * (1 + r * dJ[3])

    # ddg/dxdx
    tmp = 1 / (r * r)
    for i in range(3):
        DDg[1 + i, 1 + i] = g[0] * (dJ[i] * dJ[i] - tmp)
        for j in range(i + 1, 3):
            DDg[1 + i, 1 + j] = g[0] * dJ[i] * dJ[j]

    # ddg/dxdr
    for i in range(3):
        DDg[1 + i, 4] = I * G * dJ[i] + g[0] * \
            (dJ[i] * dJ[3] - 2 * tmp * rY[i])

    # ddg/drdr
    DDg[4, 4] = 2 * I * G * dJ[3] + g[0] * (dJ[3] * dJ[3] + 6 * tmp * J)

    # # # Symmetrise
    for i in range(5):
        for j in range(i + 1, 5):
            DDg[j, i] = DDg[i, j]


@cuda.jit("void(f4[:],f4[:,:],f4[:,:],f4[:,:],f4[:,:],f4[:],f4[:],f4[:,:,:],f4[:],f4[:,:],f4[:,:,:],i4[:])", cache=True)
def __derivs_iso(I, x, r, w0, w1, p0, p1, C, f, Df, DDf, sz):
    '''
    Calculate 0th, 1st and 2nd derivatives of 
        .5\int|I g(r(y-x),r\theta) - C|^2
    D = [D_I, D_x, D_r]
    (I,x,r) must be a length-1 list of atoms
    '''
    buf = cuda.shared.array((THREADS, 21), dtype=f4)
    F = cuda.local.array((1,), f4)
    DF = cuda.local.array((5,), f4)
    DDF = cuda.local.array((5, 5), f4)
    g = cuda.local.array((1,), f4)
    dg = cuda.local.array((5,), f4)
    ddg = cuda.local.array((5, 5), f4)
    Y = cuda.local.array((3,), f4)

    block, thread = cuda.blockIdx.x, cuda.threadIdx.x
    cc = cuda.grid(1)

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
            Y[0] = p0[i0] * w0[T, 0] + p1[i1] * w1[T, 0] - x[0, 0]
            Y[1] = p0[i0] * w0[T, 1] + p1[i1] * w1[T, 1] - x[0, 1]
            Y[2] = p0[i0] * w0[T, 2] + p1[i1] * w1[T, 2] - x[0, 2]
            __derivs_iso_aux(I[0], Y, r[0, 0], g, dg, ddg)

            g[0] = g[0] - C[T, i0, i1]
            F[0] += g[0] * g[0]
            for j0 in range(5):
                DF[j0] += g[0] * dg[j0]
                for j1 in range(5):
                    DDF[j0, j1] += dg[j0] * dg[j1] + g[0] * ddg[j0, j1]

    F[0] *= 0.5
    for i0 in range(DDF.shape[0] - 1):
        for i1 in range(i0 + 1, DDF.shape[0]):
            DDF[i1, i0] = DDF[i0, i1]

    buf[thread, 0] = F[0]
    i = 1
    for i0 in range(5):
        buf[thread, i] = DF[i0]
        i += 1
    for i0 in range(5):
        for i1 in range(i0 + 1):
            buf[thread, i] = DDF[i0, i1]
            i += 1
    cuda.syncthreads()

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


@cuda.jit("void(f4[:],f4[:,:],f4[:,:],f4[:,:],f4[:],f4[:],f4[:,:,:],f4[:],f4[:,:],f4[:,:,:],i4[:])", cache=True)
def __derivs_iso_2p5D(I, x, r, w0, p0, p1, C, f, Df, DDf, sz):
    '''
    Calculate 0th, 1st and 2nd derivatives of 
        .5\int|I g(r(y-x),r\theta) - C|^2
    D = [D_I, D_x, D_r]
    (I,x,r) must be a length-1 list of atoms
    '''
    buf = cuda.shared.array((THREADS, 21), dtype=f4)
    F = cuda.local.array((1,), f4)
    DF = cuda.local.array((5,), f4)
    DDF = cuda.local.array((5, 5), f4)
    g = cuda.local.array((1,), f4)
    dg = cuda.local.array((5,), f4)
    ddg = cuda.local.array((5, 5), f4)
    Y = cuda.local.array((3,), f4)

    block, thread = cuda.blockIdx.x, cuda.threadIdx.x
    cc = cuda.grid(1)

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
            Y[0] = p0[i0] * w0[T, 0] - x[0, 0]
            Y[1] = p0[i0] * w0[T, 1] - x[0, 1]
            Y[2] = p1[i1] - x[0, 2]
            __derivs_iso_aux(I[0], Y, r[0, 0], g, dg, ddg)

            g[0] = g[0] - C[T, i0, i1]
            F[0] += g[0] * g[0]
            for j0 in range(5):
                DF[j0] += g[0] * dg[j0]
                for j1 in range(5):
                    DDF[j0, j1] += dg[j0] * dg[j1] + g[0] * ddg[j0, j1]

    F[0] *= 0.5
    for i0 in range(DDF.shape[0] - 1):
        for i1 in range(i0 + 1, DDF.shape[0]):
            DDF[i1, i0] = DDF[i0, i1]

    buf[thread, 0] = F[0]
    i = 1
    for i0 in range(5):
        buf[thread, i] = DF[i0]
        i += 1
    for i0 in range(5):
        for i1 in range(i0 + 1):
            buf[thread, i] = DDF[i0, i1]
            i += 1
    cuda.syncthreads()

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


def derivs_aniso(atom, Rad, C):
    S = Rad.ProjectionSpace
    t = S.orientations
    w0 = S.ortho[0]
    p0, p1 = S.detector

    block = min(w0.shape[0], 1000)
    sz = [w0.shape[0], p0.shape[0], p1.shape[0],
          (w0.shape[0] * p0.shape[0] * p1.shape[0]) // (THREADS * block)]
    if sz[3] * THREADS * block < sz[0] * sz[1] * sz[2]:
        sz[3] += 1

    f, Df, DDf = empty((block,), dtype='f4'), empty(
        (10, block), dtype='f4'), empty((10, 10, block), dtype='f4')

    if len(S.ortho) == 2:
        __derivs_aniso[block, THREADS](atom.I, atom.x, atom.r, t, w0, S.ortho[1], p0,
                                       p1, C.array / sqrt(2 * pi), f, Df, DDf, array(sz, dtype='i4'))
    else:
        __derivs_aniso_2p5D[block, THREADS](atom.I, atom.x, atom.r, t, w0, p0,
                                            p1, C.array / sqrt(2 * pi), f, Df, DDf, array(sz, dtype='i4'))

    return f.sum(axis=-1), Df.sum(axis=-1), DDf.sum(axis=-1)


@cuda.jit("void(f4,f4[:],f4[:,:],f4[:],f4[:],f4[:],f4[:,:])", device=True, inline=True, cache=True)
def __derivs_aniso_aux(I, Y, rr, t, g, Dg, DDg):
    '''
    g = I*norm*exp(.5*J(rY,rThat)^2 - .5|rY|^2)
    J(y,t) = <y,t>, G = exp(...)
    Y = y-x
    '''

    # Allocate memory:
    ind0 = cuda.local.array((6,), i4)
    ind1 = cuda.local.array((6,), i4)
    ddgdyy = cuda.local.array((3, 3), f4)
    ddgdyT = cuda.local.array((3, 3), f4)
    ddgdTT = cuda.local.array((3, 3), f4)
    rT = cuda.local.array((3,), f4)
    rThat = cuda.local.array((3,), f4)
    rY = cuda.local.array((3,), f4)
    dgdy = cuda.local.array((3,), f4)
    dgdT = cuda.local.array((3,), f4)

    ind0[0], ind0[1], ind0[2] = 0, 1, 2
    ind0[3], ind0[4], ind0[5] = 0, 1, 0
    ind1[0], ind1[1], ind1[2] = 0, 1, 2
    ind1[3], ind1[4], ind1[5] = 1, 2, 2

    rT[0] = rr[0, 0] * t[0] + rr[0, 1] * t[1] + rr[0, 2] * t[2]
    rT[1] = rr[1, 1] * t[1] + rr[1, 2] * t[2]
    rT[2] = rr[2, 2] * t[2]

    norm = 1 / c_sqrt(rT[0] * rT[0] + rT[1]
                      * rT[1] + rT[2] * rT[2])
    rThat[0], rThat[1], rThat[2] = rT[0] * norm, rT[1] * norm, rT[2] * norm
    rY[0] = rr[0, 0] * Y[0] + rr[0, 1] * Y[1] + rr[0, 2] * Y[2]
    rY[1] = rr[1, 1] * Y[1] + rr[1, 2] * Y[2]
    rY[2] = rr[2, 2] * Y[2]

    J = rY[0] * rThat[0] + rY[1] * rThat[1] + rY[2] * rThat[2]
    G = 0.5 * (J * J - rY[0] * rY[0] -
               rY[1] * rY[1] - rY[2] * rY[2])

    if G < -30:
        g[0] = 0
        for i in range(10):
            Dg[i] = 0
            for j in range(10):
                DDg[i, j] = 0
        return

    G = c_exp(G)

    # # # Zeroth derivative
    g[0] = I * norm * G

    # # # First derivative
    Dg[0] = G * norm

    # dg/dy, dg/d\theta
    for i in range(3):
        dgdy[i] = g[0] * (J * rThat[i] - rY[i])
        dgdT[i] = (g[0] * norm) * (J * rY[i] - (J * J + 1) * rThat[i])

    # Dg/dx
    for i in range(3):
        tmp = 0
        for j in range(i + 1):
            tmp -= dgdy[j] * rr[j, i]
        Dg[1 + i] = tmp

    # Dg/dr
    for i in range(6):
        i0, i1 = ind0[i], ind1[i]
        Dg[4 + i] = dgdy[i0] * Y[i1] + dgdT[i0] * t[i1]

    # # # Second derivative
    for i in range(3):
        ddgdyy[i, i] = g[0] * (-1 + rY[i] * rY[i] - 2 * J *
                               rY[i] * rThat[i] + (J * J + 1) * rThat[i] * rThat[i])
    for i in range(3):
        for j in range(i + 1, 3):
            ddgdyy[i, j] = g[0] * (rY[i] * rY[j] - J * (rY[i] * rThat[j] +
                                                        rY[j] * rThat[i]) + (J * J + 1) * rThat[i] * rThat[j])
            ddgdyy[j, i] = ddgdyy[i, j]

    for i in range(3):
        ddgdyT[i, i] = (g[0] * norm) * (J - J * rY[i] * rY[i] + 2 * (J * J + 1) * rThat[i] * rY[i]
                                        - J * (J * J + 3) * rThat[i] * rThat[i])
    for i in range(3):
        for j in range(i + 1, 3):
            ddgdyT[i, j] = (g[0] * norm) * (-J * rY[i] * rY[j] + (J * J + 1) * (rY[i] * rThat[j] + rY[j] * rThat[i])
                                            - J * (J * J + 3) * rThat[i] * rThat[j])
            ddgdyT[j, i] = ddgdyT[i, j]

    for i in range(3):
        ddgdTT[i, i] = (g[0] * norm * norm) * (-(1 + J * J) + (1 + J * J) * rY[i] * rY[i] - 2 * J * (
            3 + J * J) * rY[i] * rThat[i] + (3 + J * J * (6 + J * J)) * rThat[i] * rThat[i])
    for i in range(3):
        for j in range(i + 1, 3):
            ddgdTT[i, j] = (g[0] * norm * norm) * ((1 + J * J) * rY[i] * rY[j] - J * (
                3 + J * J) * (rY[i] * rThat[j] + rY[j] * rThat[i]) + (3 + J * J * (6 + J * J)) * rThat[i] * rThat[j])
            ddgdTT[j, i] = ddgdTT[i, j]

    DDg[0, 0] = 0

    # Remove factor of I:
    for i in range(3):
        dgdy[i] = (G * norm) * (J * rThat[i] - rY[i])
        dgdT[i] = (G * norm * norm) * (J * rY[i] - (J * J + 1) * rThat[i])

    # ddg/dIdx
    for i in range(3):
        tmp = 0
        for j in range(i + 1):
            tmp -= dgdy[j] * rr[j, i]
        DDg[0, 1 + i] = tmp

    # ddg/dIdr
    for i in range(6):
        i0, i1 = ind0[i], ind1[i]
        DDg[0, 4 + i] = dgdy[i0] * Y[i1] + dgdT[i0] * t[i1]

    # ddg/dxdx
    for i in range(3):
        for j in range(i, 3):
            tmp = 0
            for k in range(i + 1):
                for l in range(j + 1):
                    tmp += ddgdyy[k, l] * rr[k, i] * rr[l, j]
            DDg[1 + i, 1 + j] = tmp

    # ddg/dxdr
    for i in range(3):
        for j in range(6):
            j0, j1 = ind0[j], ind1[j]
            tmp = 0
            for k in range(i + 1):
                tmp -= ddgdyy[j0, k] * rr[k, i] * Y[j1] + \
                    ddgdyT[k, j0] * rr[k, i] * t[j1]
            if j1 == i:
                tmp -= I * dgdy[j0]
            DDg[1 + i, 4 + j] = tmp

    # ddg/drdr
    for i in range(6):
        i0, i1 = ind0[i], ind1[i]
        for j in range(i, 6):
            j0, j1 = ind0[j], ind1[j]
            DDg[4 + i, 4 + j] = ddgdyy[i0, j0] * Y[i1] * Y[j1] + \
                ddgdyT[i0, j0] * Y[i1] * t[j1] + ddgdyT[j0, i0] * Y[j1] * t[i1]\
                + ddgdTT[i0, j0] * t[i1] * t[j1]

    # # # Symmetrise
    for i in range(10):
        for j in range(i + 1, 10):
            DDg[j, i] = DDg[i, j]


@cuda.jit("void(f4[:],f4[:,:],f4[:,:],f4[:,:],f4[:,:],f4[:,:],f4[:],f4[:],f4[:,:,:],f4[:],f4[:,:],f4[:,:,:],i4[:])", cache=True)
def __derivs_aniso(I, x, r, t, w0, w1, p0, p1, C, f, Df, DDf, sz):
    '''
    Calculate 0th, 1st and 2nd derivatives of 
        .5\int|I g(r(y-x),r\theta) - C|^2
    D = [D_I, D_x, D_r]
    (I,x,r) must be a length-1 list of atoms
    '''
    buf = cuda.shared.array((THREADS, 66), dtype=f4)
    F = cuda.local.array((1,), f4)
    DF = cuda.local.array((10,), f4)
    DDF = cuda.local.array((10, 10), f4)
    g = cuda.local.array((1,), f4)
    dg = cuda.local.array((10,), f4)
    ddg = cuda.local.array((10, 10), f4)
    rr = cuda.local.array((3, 3), f4)
    Y = cuda.local.array((3,), f4)

    block, thread = cuda.blockIdx.x, cuda.threadIdx.x
    cc = cuda.grid(1)

    F[0] = 0
    for i in range(10):
        DF[i] = 0
        for j in range(10):
            DDF[i, j] = 0
    rr[0, 0] = r[0, 0]
    rr[1, 1] = r[0, 1]
    rr[2, 2] = r[0, 2]
    rr[0, 1] = r[0, 3]
    rr[1, 2] = r[0, 4]
    rr[0, 2] = r[0, 5]

    for indx in range(sz[3] * cc, sz[3] * (cc + 1)):
        T = indx // (sz[1] * sz[2])
        i0 = (indx // sz[2]) - sz[1] * T
        i1 = indx - sz[2] * (i0 + sz[1] * T)
        if T < sz[0]:
            Y[0] = p0[i0] * w0[T, 0] + p1[i1] * w1[T, 0] - x[0, 0]
            Y[1] = p0[i0] * w0[T, 1] + p1[i1] * w1[T, 1] - x[0, 1]
            Y[2] = p0[i0] * w0[T, 2] + p1[i1] * w1[T, 2] - x[0, 2]
            __derivs_aniso_aux(I[0], Y, rr, t[T], g, dg, ddg)

            g[0] = g[0] - C[T, i0, i1]
            F[0] += g[0] * g[0]
            for j0 in range(10):
                DF[j0] += g[0] * dg[j0]
                for j1 in range(10):
                    DDF[j0, j1] += dg[j0] * dg[j1] + g[0] * ddg[j0, j1]

    F[0] *= 0.5
    for i0 in range(DDF.shape[0] - 1):
        for i1 in range(i0 + 1, DDF.shape[0]):
            DDF[i1, i0] = DDF[i0, i1]

    buf[thread, 0] = F[0]
    i = 1
    for i0 in range(10):
        buf[thread, i] = DF[i0]
        i += 1
    for i0 in range(10):
        for i1 in range(i0 + 1):
            buf[thread, i] = DDF[i0, i1]
            i += 1
    cuda.syncthreads()

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


@cuda.jit("void(f4[:],f4[:,:],f4[:,:],f4[:,:],f4[:,:],f4[:],f4[:],f4[:,:,:],f4[:],f4[:,:],f4[:,:,:],i4[:])", cache=True)
def __derivs_aniso_2p5D(I, x, r, t, w0, p0, p1, C, f, Df, DDf, sz):
    '''
    Calculate 0th, 1st and 2nd derivatives of 
        .5\int|I g(r(y-x),r\theta) - C|^2
    D = [D_I, D_x, D_r]
    (I,x,r) must be a length-1 list of atoms
    '''
    buf = cuda.shared.array((THREADS, 66), dtype=f4)
    F = cuda.local.array((1,), f4)
    DF = cuda.local.array((10,), f4)
    DDF = cuda.local.array((10, 10), f4)
    g = cuda.local.array((1,), f4)
    dg = cuda.local.array((10,), f4)
    ddg = cuda.local.array((10, 10), f4)
    rr = cuda.local.array((3, 3), f4)
    Y = cuda.local.array((3,), f4)
    tt = cuda.local.array((3,), f4)

    block, thread = cuda.blockIdx.x, cuda.threadIdx.x
    cc = cuda.grid(1)

    F[0] = 0
    for i in range(10):
        DF[i] = 0
        for j in range(10):
            DDF[i, j] = 0
    rr[0, 0] = r[0, 0]
    rr[1, 1] = r[0, 1]
    rr[2, 2] = r[0, 2]
    rr[0, 1] = r[0, 3]
    rr[1, 2] = r[0, 4]
    rr[0, 2] = r[0, 5]
    tt[2] = 0

    for indx in range(sz[3] * cc, sz[3] * (cc + 1)):
        T = indx // (sz[1] * sz[2])
        i0 = (indx // sz[2]) - sz[1] * T
        i1 = indx - sz[2] * (i0 + sz[1] * T)
        if T < sz[0]:
            Y[0] = p0[i0] * w0[T, 0] - x[0, 0]
            Y[1] = p0[i0] * w0[T, 1] - x[0, 1]
            Y[2] = p1[i1] - x[0, 2]
            tt[0], tt[1] = t[T, 0], t[T, 1]
            __derivs_aniso_aux(I[0], Y, rr, tt, g, dg, ddg)

            g[0] = g[0] - C[T, i0, i1]
            F[0] += g[0] * g[0]
            for j0 in range(10):
                DF[j0] += g[0] * dg[j0]
                for j1 in range(10):
                    DDF[j0, j1] += dg[j0] * dg[j1] + g[0] * ddg[j0, j1]

    F[0] *= 0.5
    for i0 in range(DDF.shape[0] - 1):
        for i1 in range(i0 + 1, DDF.shape[0]):
            DDF[i1, i0] = DDF[i0, i1]

    buf[thread, 0] = F[0]
    i = 1
    for i0 in range(10):
        buf[thread, i] = DF[i0]
        i += 1
    for i0 in range(10):
        for i1 in range(i0 + 1):
            buf[thread, i] = DDF[i0, i1]
            i += 1
    cuda.syncthreads()

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


# @cuda.jit("void(f4,f4[:],f4[:,:],f4[:],f4[:],f4[:],f4[:,:])", cache=True)
# def __test(I, Y, rr, t, g, Dg, DDg):
#     #    __derivs_aniso_aux(I, Y, rr, t, g, Dg, DDg)
#     __derivs_iso_aux(I, Y, rr[0, 0], g, Dg, DDg)


if __name__ == '__main__':
    from numpy import array_str, random, concatenate
    from sympy import symbols, exp as s_exp, sqrt as s_sqrt, diff

    def niceprint(thing): return array_str(
        array(thing, 'float64'), precision=3)

    s_I, s_x, s_y, s_r, s_T, s_C = symbols(
        ('I', 'x:3', 'y:3', 'r:6', 'T:3', 'C'), real=True)
    s_Y = tuple(s_y[i] - s_x[i] for i in range(3))
    s_rY = (s_r[0] * s_Y[0] + s_r[3] * s_Y[1] + s_r[5] * s_Y[2],
            s_r[1] * s_Y[1] + s_r[4] * s_Y[2], s_r[2] * s_Y[2])
    s_rT = (s_r[0] * s_T[0] + s_r[3] * s_T[1] + s_r[5] * s_T[2],
            s_r[1] * s_T[1] + s_r[4] * s_T[2], s_r[2] * s_T[2])
    s_nT = s_sqrt(s_rT[0]**2 + s_rT[1]**2 + s_rT[2]**2)

    I, x, y, r, T, C = random.rand(1), random.rand(
        1, 3), random.rand(1, 3), random.rand(1, 6), random.rand(1, 3), random.rand(1, 1, 1)
    for thing in 'IxyrTC':
        locals()[thing] = locals()[thing].astype('f4')

    # # # # # #
    # Test __derivs_aniso
    # # # # # #
#     J = (s_rY[0] * s_rT[0] + s_rY[1] * s_rT[1] + s_rY[2] * s_rT[2]) / s_nT
#     G = s_exp(0.5 * J**2 - 0.5 * (s_rY[0]**2 + s_rY[1]**2 + s_rY[2]**2))
#
#     f = (s_I / s_nT) * G
#
#     var = (s_I,) + s_x + s_r
#     df = [diff(f, x) for x in var]
#     ddf = [[diff(d, x) for x in var] for d in df]
#
#     g, dg, ddg = empty(1, dtype='f4'), empty(
#         10, dtype='f4'), empty((10, 10), dtype='f4')
#
#     I, x, y, r, T = random.rand(1), random.rand(
#         3), random.rand(3), random.rand(6), random.rand(3)
#     for thing in ('I', 'x', 'y', 'r', 'T'):
#         locals()[thing] = locals()[thing].astype('f4')
#     rr = empty((3, 3), dtype='f4')
#     rr[0, 0] = r[0]
#     rr[1, 1] = r[1]
#     rr[2, 2] = r[2]
#     rr[0, 1] = r[3]
#     rr[1, 2] = r[4]
#     rr[0, 2] = r[5]
#     var = (s_I,) + s_x + s_y + s_r + s_T
#     val = concatenate((I, x, y, r, T))
#     subs = [(var[i], val[i]) for i in range(len(var))]
#
#     f = f.subs(subs)
#     df = [d.subs(subs) for d in df]
#     ddf = [[d.subs(subs) for d in dd] for dd in ddf]
#
#     __test[1, 1](I[0], y - x, rr, T, g, dg, ddg)
#     print(niceprint(f))
#     print(niceprint(g[0]), '\n')
#     print('f error: ', niceprint(abs(f - g[0])), '\n')
#     print(niceprint(df))
#     print(niceprint(dg), '\n')
#     print('Df error: ', niceprint(abs(array(df) - dg).max()), '\n')
#     print(niceprint(ddg), '\n')
#     print('DDf error: ', niceprint(abs(array(ddf) - ddg).max()), '\n')
#     for i in range(10):
#         for j in range(i, 10):
#             if abs(ddf[i][j] - ddg[i][j]) > 1e-8:
#                 print('index ', i, j, ' is wrong')

    # # # # # #
    # Test __derivs_aniso
    # # # # # #
    J = (s_rY[0] * s_rT[0] + s_rY[1] * s_rT[1] + s_rY[2] * s_rT[2]) / s_nT
    G = s_exp(0.5 * J**2 - 0.5 * (s_rY[0]**2 + s_rY[1]**2 + s_rY[2]**2))
    f = 0.5 * ((s_I / s_nT) * G - s_C)**2

    var = (s_I,) + s_x + s_r
    df = [diff(f, x) for x in var]
    ddf = [[diff(d, x) for x in var] for d in df]
    var = (s_C, s_I) + s_x + s_y + s_r + s_T
    val = concatenate((C[0, 0], I, x[0], y[0], r[0], T[0]))
    subs = [(var[i], val[i]) for i in range(len(var))]
    f = f.subs(subs)
    df = [d.subs(subs) for d in df]
    ddf = [[d.subs(subs) for d in dd] for dd in ddf]

    g, dg, ddg = empty(2, dtype='f4'), empty(
        (10, 2), dtype='f4'), empty((10, 10, 2), dtype='f4')
#     __derivs_aniso[2, 1](I, x, r, T, y, array([[0, 0, 0]], dtype='f4'),
#                          array(1, dtype='f4'), array(
#                              1, dtype='f4'), C, g, dg, ddg,
#                          array([1, 1, 1, 1], dtype='i4'))
    __derivs_aniso_2p5D[2, 1](I, x, r, T, y[:, :2], array(1, dtype='f4'),
                              array(y[0, 2], dtype='f4'), C, g, dg, ddg,
                              array([1, 1, 1, 1], dtype='i4'))
    g, dg, ddg = g.sum(-1), dg.sum(-1), ddg.sum(-1)

    print('Printouts for aniso test:')
#     print(niceprint(f))
#     print(niceprint(g[0]), '\n')
    print('f error: ', niceprint(abs(f - g)), '\n')
#     print(niceprint(df))
#     print(niceprint(dg), '\n')
    print('Df error: ', niceprint(abs(array(df) - dg).max()), '\n')
#     print(niceprint(ddg), '\n')
    print('DDf error: ', niceprint(abs(array(ddf) - ddg).max()), '\n')
#     for i in range(10):
#         for j in range(i, 10):
#             if abs(ddf[i][j] - ddg[i][j]) > 1e-8:
#                 print('index ', i, j, ' is wrong')

    # # # # # #
    # Test __derivs_iso
    # # # # # #
    G = s_exp(- 0.5 * (s_Y[0]**2 + s_Y[1]**2 + s_Y[2]**2) / s_r[0]**2)
    f = 0.5 * (s_I * s_r[0] * G - s_C)**2

    I, x, y, r, T, C = random.rand(1), random.rand(
        1, 3), random.rand(1, 3), random.rand(1, 6), random.rand(1, 3), random.rand(1, 1, 1)
    for thing in 'IxyrTC':
        locals()[thing] = locals()[thing].astype('f4')

    var = (s_I,) + s_x + s_r[:1]
    df = [diff(f, x) for x in var]
    ddf = [[diff(d, x) for x in var] for d in df]
    var = (s_C, s_I) + s_x + s_y + s_r[:1]
    val = concatenate((C[0, 0], I, x[0], y[0], r[0]))
    subs = [(var[i], val[i]) for i in range(len(var))]
    f = f.subs(subs)
    df = [d.subs(subs) for d in df]
    ddf = [[d.subs(subs) for d in dd] for dd in ddf]

    g, dg, ddg = empty(1, dtype='f4'), empty(
        (5, 1), dtype='f4'), empty((5, 5, 1), dtype='f4')
#     __derivs_iso[1, 1](I, x, r[:, :1], y, array([[0, 0, 0]], dtype='f4'),
#                        array(1, dtype='f4'), array(1, dtype='f4'),
#                        C, g, dg, ddg, array([1] * 4, dtype='i4'))
    __derivs_iso_2p5D[1, 1](I, x, r[:, :1], y[:, :2], array(1, dtype='f4'),
                            array(y[0, 2], dtype='f4'), C, g, dg, ddg,
                            array([1] * 4, dtype='i4'))
    g, dg, ddg = g.sum(-1), dg.sum(-1), ddg.sum(-1)

    print('Printouts for iso test:')
#     print(niceprint(f))
#     print(niceprint(g), '\n')
    print('f error: ', niceprint(abs(f - g)), '\n')
#     print(niceprint(df))
#     print(niceprint(dg), '\n')
    print('Df error: ', niceprint(abs(array(df) - dg).max()), '\n')
#     print(niceprint(ddf), '\n')
#     print(niceprint(ddg), '\n')
    print('DDf error: ', niceprint(abs(array(ddf) - ddg).max()), '\n')
#     for i in range(5):
#         for j in range(i, 5):
#             if abs(ddf[i][j] - ddg[i][j]) > 1e-8:
#                 print('index ', i, j, ' is wrong')
