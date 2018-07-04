'''
Created on 15 Feb 3018

@author: Rob Tovey
'''
from numpy import sqrt, pi, exp, array
import numba
from numba import cuda
from math import exp as c_exp, sqrt as c_sqrt
from .numba_cuda_aux import __GPU_reduce_1, __GPU_reduce_2, THREADS,\
    __GPU_reduce_n
from .manager import context
Three = array([3], dtype='int32')[0]
BATCH = True


@numba.jit(["void(T[:],T[:,:],T[:,:],T[:,:],T[:,:],T[:],T[:,:])".replace('T', T) for T in ['f4', 'f8']], target='cpu', cache=True, nopython=True)
def GaussRadon_CPU(I, x, r, t, w, p, R):
    s0 = sqrt(2 * pi) * I
    for jj in range(t.shape[0]):
        for kk in range(p.shape[0]):
            tmp = 0
            for ii in range(x.shape[0]):
                rT = [r[ii, 0] * t[jj, 0] + r[ii, 2] * t[jj, 1],
                      r[ii, 1] * t[jj, 1]]
                rX = [p[kk] * w[jj, 0] - x[ii, 0],
                      p[kk] * w[jj, 1] - x[ii, 1]]
                rX = [r[ii, 0] * rX[0] + r[ii, 2] * rX[1],
                      r[ii, 1] * rX[1]]

                s1 = 1 / sqrt(rT[0] * rT[0] + rT[1] * rT[1])
                s2 = rT[0] * rX[0] + rT[1] * rX[1]
                inter = (s2 * s2 * s1 * s1 -
                         (rX[0] * rX[0] + rX[1] * rX[1])) / 2

                if inter > -20:
                    tmp += s0[ii] * s1 * exp(inter)
            R[jj, kk] = tmp


def GaussRadon_GPU(I, x, r, t, w, p, R):
    s0 = context().mul(I, sqrt(2 * pi))
    if BATCH:
        grid = w.shape[0], p.shape[0]
        tpb = 4
        __GaussRadon_GPU[tuple(-(-g // tpb) for g in grid),
                         (tpb, tpb)](s0, x, r, t, w, p, R)
    else:
        grid = 1, p.shape[0]
        tpb = 4
        grid = tuple(-(-g // tpb) for g in grid)
        for i in range(w.shape[0]):
            __GaussRadon_GPU[grid, (1, tpb)](
                s0, x, r, t[i:i + 1], w[i:i + 1], p, R[i:i + 1])


@cuda.jit("void(f4[:],f4[:,:],f4[:,:],f4[:,:],f4[:,:],f4[:],f4[:,:])", cache=True)
def __GaussRadon_GPU(s0, x, r, t, w, p, R):
    jj, kk = cuda.grid(2)
    tmp = 0
    for ii in range(x.shape[0]):
        rT0 = r[ii, 0] * t[jj, 0] + r[ii, 2] * t[jj, 1]
        rT1 = r[ii, 1] * t[jj, 1]
        rX0 = p[kk] * w[jj, 0] - x[ii, 0]
        rX1 = p[kk] * w[jj, 1] - x[ii, 1]
        rX0 = r[ii, 0] * rX0 + r[ii, 2] * rX1
        rX1 = r[ii, 1] * rX1

        s1 = 1 / c_sqrt(rT0 * rT0 + rT1 * rT1)
        s2 = rT0 * rX0 + rT1 * rX1
        inter = (s2 * s2 * s1 * s1 - (rX0 * rX0 + rX1 * rX1)) / 2
        if inter > -20:
            tmp += s0[ii] * s1 * c_exp(inter)

    R[jj, kk] = tmp


@numba.jit(["void(T[:],T[:,:],T[:,:],T[:,:],T[:,:],T[:],T[:,:],T[:])".replace('T', T) for T in ['f4', 'f8']], target='cpu', cache=True, nopython=True)
def GaussRadon_DI_CPU(I, x, r, t, w, p, C, DR):
    s0 = sqrt(2 * pi)
    for ii in range(x.shape[0]):
        DR[ii] = 0
        for jj in range(w.shape[0]):
            for kk in range(p.shape[0]):
                rT = [r[ii, 0] * t[jj, 0] + r[ii, 2] * t[jj, 1],
                      r[ii, 1] * t[jj, 1]]
                rX = [p[kk] * w[jj, 0] - x[ii, 0],
                      p[kk] * w[jj, 1] - x[ii, 1]]
                rX = [r[ii, 0] * rX[0] + r[ii, 2] * rX[1],
                      r[ii, 1] * rX[1]]

                s1 = 1 / sqrt(rT[0] * rT[0] + rT[1] * rT[1])
                s2 = rT[0] * rX[0] + rT[1] * rX[1]
                inter = (s2 * s2 * s1 * s1 -
                         (rX[0] * rX[0] + rX[1] * rX[1])) / 2
                if inter > -20:
                    DR[ii] += s0 * s1 * exp(inter) * C[jj, kk]


def GaussRadon_DI_GPU(I, x, r, t, w, p, C, DR):
    s0 = sqrt(2 * pi)
    grid = x.shape[0]
    if BATCH:
        sz = [w.shape[0], p.shape[0], (w.shape[0] * p.shape[0]) // THREADS]
        if sz[2] * THREADS < sz[0] * sz[1]:
            sz[2] += 1
        __GaussRadon_DI_GPU[grid, THREADS](
            s0, x, r, t, w, p, C, DR, array(sz, dtype='int32'))
    else:
        c = context()
        d = c.copy(DR)
        c.set(DR[:], 0)
        sz = [1, p.shape[0], (w.shape[0] * p.shape[0]) // THREADS]
        if sz[2] * THREADS < sz[0] * sz[1]:
            sz[2] += 1
        sz = array(sz, dtype='int32')
        for i in range(w.shape[0]):
            __GaussRadon_DI_GPU[grid, THREADS](
                s0, x, r, t[i:i + 1], w[i:i + 1], p,
                C[i:i + 1], d, sz)
            c.add(DR, d, DR)


@cuda.jit("void(f4,f4[:,:],f4[:,:],f4[:,:],f4[:,:],f4[:],f4[:,:],f4[:],i4[:])", cache=True)
def __GaussRadon_DI_GPU(s0, x, r, t, w, p, C, DR, sz):
    buf = cuda.shared.array(THREADS, dtype=numba.float32)
    ii = cuda.blockIdx.x
    cc = cuda.threadIdx.x

    sum0 = 0
    for indx in range(sz[2] * cc, sz[2] * (cc + 1)):
        jj = indx // sz[1]
        kk = indx - jj * sz[1]
        if jj < sz[0]:
            rT0 = r[ii, 0] * t[jj, 0] + r[ii, 2] * t[jj, 1]
            rT1 = r[ii, 1] * t[jj, 1]
            rX0 = p[kk] * w[jj, 0] - x[ii, 0]
            rX1 = p[kk] * w[jj, 1] - x[ii, 1]
            rX0 = r[ii, 0] * rX0 + r[ii, 2] * rX1
            rX1 = r[ii, 1] * rX1

            s1 = 1 / c_sqrt(rT0 * rT0 + rT1 * rT1)
            s2 = rT0 * rX0 + rT1 * rX1
            inter = (s2 * s2 * s1 * s1 - (rX0 * rX0 + rX1 * rX1)) / 2
            if inter > -20:
                sum0 += s1 * c_exp(inter) * C[jj, kk]
    buf[cc] = s0 * sum0
    cuda.syncthreads()

    __GPU_reduce_1(buf)
    if cc == 0:
        DR[ii] = buf[0]


@numba.jit(["void(T[:],T[:,:],T[:,:],T[:,:],T[:,:],T[:],T[:,:],T[:,:])".replace('T', T) for T in ['f4', 'f8']], target='cpu', cache=True, nopython=True)
def GaussRadon_Dx_CPU(I, x, r, t, w, p, C, DR):
    s0 = I * sqrt(2 * pi)
    for ii in range(x.shape[0]):
        DR[ii, :] = 0
        for jj in range(w.shape[0]):
            for kk in range(p.shape[0]):
                rT = [r[ii, 0] * t[jj, 0] + r[ii, 2] * t[jj, 1],
                      r[ii, 1] * t[jj, 1]]
                rX = [p[kk] * w[jj, 0] - x[ii, 0],
                      p[kk] * w[jj, 1] - x[ii, 1]]
                rX = [r[ii, 0] * rX[0] + r[ii, 2] * rX[1],
                      r[ii, 1] * rX[1]]

                s1 = 1 / sqrt(rT[0] * rT[0] + rT[1] * rT[1])
                s2 = rT[0] * rX[0] + rT[1] * rX[1]
                inter = (s2 * s2 * s1 * s1 -
                         (rX[0] * rX[0] + rX[1] * rX[1])) / 2
                if inter > -20:
                    inter = s0[ii] * s1 * exp(inter) * C[jj, kk]
                    rX[0] = rX[0] - s2 * s1 * s1 * rT[0]
                    rX[1] = rX[1] - s2 * s1 * s1 * rT[1]
                    DR[ii, 0] += inter * r[ii, 0] * rX[0]
                    DR[ii, 1] += inter * (r[ii, 2] * rX[0] + r[ii, 1] * rX[1])


def GaussRadon_Dx_GPU(I, x, r, t, w, p, C, DR):
    s0 = context().mul(I, sqrt(2 * pi))
    grid = x.shape[0]
    if BATCH:
        sz = [w.shape[0], p.shape[0], (w.shape[0] * p.shape[0]) // THREADS]
        if sz[2] * THREADS < sz[0] * sz[1]:
            sz[2] += 1
        __GaussRadon_Dx_GPU[grid, THREADS](
            s0, x, r, t, w, p, C, DR, array(sz, dtype='int32'))
    else:
        c = context()
        d = c.copy(DR)
        c.set(DR[:], 0)
        sz = [1, p.shape[0], (w.shape[0] * p.shape[0]) // THREADS]
        if sz[2] * THREADS < sz[0] * sz[1]:
            sz[2] += 1
        sz = array(sz, dtype='int32')
        for i in range(w.shape[0]):
            __GaussRadon_Dx_GPU[grid, THREADS](
                s0, x, r, t[i:i + 1], w[i:i + 1], p,
                C[i:i + 1], d, sz)
            c.add(DR, d, DR)


@cuda.jit("void(f4[:],f4[:,:],f4[:,:],f4[:,:],f4[:,:],f4[:],f4[:,:],f4[:,:],i4[:])", cache=True)
def __GaussRadon_Dx_GPU(s0, x, r, t, w, p, C, DR, sz):
    buf = cuda.shared.array((THREADS, 2), dtype=numba.float32)

    ii = cuda.blockIdx.x
    cc = cuda.threadIdx.x

    sum0 = 0
    sum1 = 0
    for indx in range(sz[2] * cc, sz[2] * (cc + 1)):
        jj = indx // sz[1]
        kk = indx - jj * sz[1]
        if jj < sz[0]:
            rT0 = r[ii, 0] * t[jj, 0] + r[ii, 2] * t[jj, 1]
            rT1 = r[ii, 1] * t[jj, 1]
            rX0 = p[kk] * w[jj, 0] - x[ii, 0]
            rX1 = p[kk] * w[jj, 1] - x[ii, 1]
            rX0 = r[ii, 0] * rX0 + r[ii, 2] * rX1
            rX1 = r[ii, 1] * rX1

            s1 = 1 / c_sqrt(rT0 * rT0 + rT1 * rT1)
            s2 = rT0 * rX0 + rT1 * rX1
            inter = (s2 * s2 * s1 * s1 - (rX0 * rX0 + rX1 * rX1)) / 2
            if inter > -20:
                inter = s0[ii] * s1 * c_exp(inter) * C[jj, kk]
                rX0 = rX0 - s2 * s1 * s1 * rT0
                rX1 = rX1 - s2 * s1 * s1 * rT1
                sum0 += inter * r[ii, 0] * rX0
                sum1 += inter * (r[ii, 2] * rX0 + r[ii, 1] * rX1)

    buf[cc, 0] = sum0
    buf[cc, 1] = sum1
    cuda.syncthreads()

    __GPU_reduce_2(buf)
    if cc == 0:
        DR[ii, 0] = buf[0, 0]
        DR[ii, 1] = buf[0, 1]


@numba.jit(["void(T[:],T[:,:],T[:,:],T[:,:],T[:,:],T[:],T[:,:],T[:,:])".replace('T', T) for T in ['f4', 'f8']], target='cpu', cache=True, nopython=True)
def GaussRadon_Dr_CPU(I, x, r, t, w, p, C, DR):
    s0 = sqrt(2 * pi) * I
    for ii in range(x.shape[0]):
        DR[ii, :] = 0
        for jj in range(w.shape[0]):
            for kk in range(p.shape[0]):
                rT = [r[ii, 0] * t[jj, 0] + r[ii, 2] * t[jj, 1],
                      r[ii, 1] * t[jj, 1]]
                X = [p[kk] * w[jj, 0] - x[ii, 0],
                     p[kk] * w[jj, 1] - x[ii, 1]]
                rX = [r[ii, 0] * X[0] + r[ii, 2] * X[1],
                      r[ii, 1] * X[1]]

                s1 = 1 / sqrt(rT[0] * rT[0] + rT[1] * rT[1])
                s2 = rT[0] * rX[0] + rT[1] * rX[1]
                inter = (s2 * s2 * s1 * s1 -
                         (rX[0] * rX[0] + rX[1] * rX[1])) / 2
                if inter > -20:
                    inter = s0[ii] * s1 * exp(inter) * C[jj, kk]
                    s3 = [s2 * s1 * s1, -(s2 * s2 * s1 * s1 + 1) * s1 * s1]

                    # d/dr00 = s30*(rX0*t0 + rT0*X0) + s31*rT0*t0 -rX0*X0
                    DR[ii, 0] += inter * (s3[0] * (rX[0] * t[jj, 0] + rT[0] * X[0])
                                          + s3[1] * rT[0] * t[jj, 0] - rX[0] * X[0])
                    # d/dr11 = s30*(rX1*t1 + rT1*X1) + s31*rT1*t1 -rX1*X1
                    DR[ii, 1] += inter * (s3[0] * (rX[1] * t[jj, 1] + rT[1] * X[1])
                                          + s3[1] * rT[1] * t[jj, 1] - rX[1] * X[1])
                    # d/dr01 = s30*(rX0*t1 + rT0*X1) + s31*rT0*t1 -rX0*X1
                    DR[ii, 2] += inter * (s3[0] * (rX[0] * t[jj, 1] + rT[0] * X[1])
                                          + s3[1] * rT[0] * t[jj, 1] - rX[0] * X[1])


def GaussRadon_Dr_GPU(I, x, r, t, w, p, C, DR):
    s0 = context().mul(I, sqrt(2 * pi))
    grid = x.shape[0]
    if BATCH:
        sz = [w.shape[0], p.shape[0], (w.shape[0] * p.shape[0]) // THREADS]
        if sz[2] * THREADS < sz[0] * sz[1]:
            sz[2] += 1
        __GaussRadon_Dr_GPU[grid, THREADS](
            s0, x, r, t, w, p, C, DR, array(sz, dtype='int32'))
    else:
        c = context()
        d = c.copy(DR)
        c.set(DR[:], 0)
        sz = [1, p.shape[0], (w.shape[0] * p.shape[0]) // THREADS]
        if sz[2] * THREADS < sz[0] * sz[1]:
            sz[2] += 1
        sz = array(sz, dtype='int32')
        for i in range(w.shape[0]):
            __GaussRadon_Dr_GPU[grid, THREADS](
                s0, x, r, t[i:i + 1], w[i:i + 1], p,
                C[i:i + 1], d, sz)
            c.add(DR, d, DR)


@cuda.jit("void(f4[:],f4[:,:],f4[:,:],f4[:,:],f4[:,:],f4[:],f4[:,:],f4[:,:],i4[:])", cache=True)
def __GaussRadon_Dr_GPU(s0, x, r, t, w, p, C, DR, sz):
    buf = cuda.shared.array((THREADS, 3), dtype=numba.float32)
    ii = cuda.blockIdx.x
    cc = cuda.threadIdx.x

    sum0 = 0
    sum1 = 0
    sum2 = 0
    for indx in range(sz[2] * cc, sz[2] * (cc + 1)):
        jj = indx // sz[1]
        kk = indx - jj * sz[1]
        if jj < sz[0]:
            rT0 = r[ii, 0] * t[jj, 0] + r[ii, 2] * t[jj, 1]
            rT1 = r[ii, 1] * t[jj, 1]
            X0 = p[kk] * w[jj, 0] - x[ii, 0]
            X1 = p[kk] * w[jj, 1] - x[ii, 1]
            rX0 = r[ii, 0] * X0 + r[ii, 2] * X1
            rX1 = r[ii, 1] * X1

            s1 = 1 / c_sqrt(rT0 * rT0 + rT1 * rT1)
            s2 = rT0 * rX0 + rT1 * rX1
            inter = (s2 * s2 * s1 * s1 - (rX0 * rX0 + rX1 * rX1)) / 2
            if inter > -20:
                inter = s0[ii] * s1 * c_exp(inter) * C[jj, kk]
                s30 = s2 * s1 * s1
                s31 = -(s2 * s2 * s1 * s1 + 1) * s1 * s1

                sum0 += inter * (s30 * (rX0 * t[jj, 0] + rT0 * X0)
                                 + s31 * rT0 * t[jj, 0] - rX0 * X0)
                sum1 += inter * (s30 * (rX1 * t[jj, 1] + rT1 * X1)
                                 + s31 * rT1 * t[jj, 1] - rX1 * X1)
                sum2 += inter * (s30 * (rX0 * t[jj, 1] + rT0 * X1)
                                 + s31 * rT0 * t[jj, 1] - rX0 * X1)

    buf[cc, 0] = sum0
    buf[cc, 1] = sum1
    buf[cc, 2] = sum2
    cuda.syncthreads()

    __GPU_reduce_n(buf, Three)
    if cc == 0:
        DR[ii, 0] = buf[0, 0]
        DR[ii, 1] = buf[0, 1]
        DR[ii, 2] = buf[0, 2]


@numba.jit(["void(T[:],T[:,:],T[:,:],T[:],T[:],T[:,:])".replace('T', T) for T in ['f4', 'f8']], target='cpu', cache=True, nopython=True)
def GaussVol_CPU(I, x, r, y0, y1, u):
    for jj in range(y0.shape[0]):
        for kk in range(y1.shape[0]):
            tmp = 0
            for ii in range(x.shape[0]):
                rX = [y0[jj] - x[ii, 0],
                      y1[kk] - x[ii, 1]]
                rX = [r[ii, 0] * rX[0] + r[ii, 2] * rX[1],
                      r[ii, 1] * rX[1]]

                interior = -.5 * (rX[0] * rX[0] + rX[1] * rX[1])
                if interior > -20:
                    tmp += I[ii] * exp(interior)
            u[jj, kk] = tmp


def GaussVol_GPU(I, x, r, y0, y1, u):
    if BATCH:
        grid = y0.shape[0], y1.shape[0]
        tpb = 4
        __GaussVol_GPU[tuple(-(-g // tpb) for g in grid), (tpb, tpb)
                       ](I, x, r, y0, y1, u)
    else:
        grid = 1, y1.shape[0]
        tpb = 4
        grid = tuple(-(-g // tpb) for g in grid)
        for i in range(y0.shape[0]):
            __GaussVol_GPU[grid, (1, tpb)
                           ](I, x, r, y0[i:i + 1], y1, u[i:i + 1])


@cuda.jit("void(f4[:],f4[:,:],f4[:,:],f4[:],f4[:],f4[:,:])", cache=True)
def __GaussVol_GPU(I, x, r, y0, y1, u):
    jj, kk = cuda.grid(2)
    tmp = 0
    for ii in range(x.shape[0]):
        rX0 = y0[jj] - x[ii, 0]
        rX1 = y1[kk] - x[ii, 1]
        rX0 = r[ii, 0] * rX0 + r[ii, 2] * rX1
        rX1 = r[ii, 1] * rX1

        interior = -.5 * (rX0 * rX0 + rX1 * rX1)
        if interior > -20:
            tmp += I[ii] * c_exp(interior)
    u[jj, kk] = tmp
