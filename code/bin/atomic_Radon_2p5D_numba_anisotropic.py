'''
Created on 20 Feb 2018

@author: Rob Tovey
'''
from numpy import sqrt, pi, exp, array
import numba
from numba import cuda
from math import exp as c_exp, sqrt as c_sqrt
from .numba_cuda_aux import __GPU_reduce_1, __GPU_reduce_3, THREADS,\
    __GPU_reduce_n
Six = array([6], dtype='int32')[0]
from .manager import context
BATCH = False


@numba.jit(["void(T[:],T[:,:],T[:,:],T[:,:],T[:,:],T[:],T[:],T[:,:,:])".replace('T', T) for T in ['f4', 'f8']], target='cpu', cache=True, nopython=True)
def GaussRadon_CPU(I, x, r, t, w, p0, p1, R):
    s0 = sqrt(2 * pi) * I
    for jj in range(w.shape[0]):
        for k0 in range(p0.shape[0]):
            for k1 in range(p1.shape[0]):
                tmp = 0
                for ii in range(x.shape[0]):
                    rT = [r[ii, 0] * t[jj, 0] + r[ii, 3] * t[jj, 1],
                          r[ii, 1] * t[jj, 1], 0]
                    rX = [p0[k0] * w[jj, 0] - x[ii, 0],
                          p0[k0] * w[jj, 1] - x[ii, 1],
                          p1[k1] - x[ii, 2]]
                    rX = [r[ii, 0] * rX[0] + r[ii, 3] * rX[1] + r[ii, 5] * rX[2],
                          r[ii, 1] * rX[1] + r[ii, 4] * rX[2],
                          r[ii, 2] * rX[2]]

                    # s1 = 1/|r\theta|, s2 = r\theta\cdot r(p-x)
                    s1 = 1 / sqrt(rT[0] * rT[0] + rT[1]
                                  * rT[1] + rT[2] * rT[2])
                    s2 = rT[0] * rX[0] + rT[1] * rX[1] + rT[2] * rX[2]
                    inter = (s1 * s1 * s2 * s2
                             - (rX[0] * rX[0] + rX[1] * rX[1] + rX[2] * rX[2])) / 2
                    if inter > -20:
                        tmp += s0[ii] * s1 * exp(inter)
                R[jj, k0, k1] = tmp


def GaussRadon_GPU(I, x, r, t, w, p0, p1, R):
    s0 = context().mul(I, sqrt(2 * pi))
    if BATCH:
        grid = w.shape[0], p0.shape[0], p1.shape[0]
        tpb = 4
        __GaussRadon_GPU[tuple(-(-g // tpb) for g in grid),
                         (tpb, tpb, tpb)](s0, x, r, t, w, p0, p1, R)
    else:
        grid = 1, p0.shape[0], p1.shape[0]
        tpb = 4
        grid = tuple(-(-g // tpb) for g in grid)
        for i in range(w.shape[0]):
            __GaussRadon_GPU[grid, (1, tpb, tpb)](
                s0, x, r, t[i:i + 1], w[i:i + 1], p0, p1, R[i:i + 1])


@cuda.jit("void(f4[:],f4[:,:],f4[:,:],f4[:,:],f4[:,:],f4[:],f4[:],f4[:,:,:])", cache=True)
def __GaussRadon_GPU(s0, x, r, t, w, p0, p1, R):
    jj, k0, k1 = cuda.grid(3)
    if jj >= t.shape[0] or k0 >= p0.shape[0] or k1 >= p1.shape[0]:
        return
    tmp = 0
    for ii in range(x.shape[0]):
        rT0 = r[ii, 0] * t[jj, 0] + r[ii, 3] * t[jj, 1]
        rT1 = r[ii, 1] * t[jj, 1]
        rT2 = 0
        rX0 = p0[k0] * w[jj, 0] - x[ii, 0]
        rX1 = p0[k0] * w[jj, 1] - x[ii, 1]
        rX2 = p1[k1] - x[ii, 2]

#         rT0 = r[ii, 3] * t[jj, 0] + r[ii, 5] * t[jj, 1]
#         rT1 = r[ii, 1] * t[jj, 0] + r[ii, 4] * t[jj, 1]
#         rT2 = r[ii, 2] * t[jj, 1]
#         rX0 = p0[k0] - x[ii, 0]
#         rX1 = p1[k1] * w[jj, 0] - x[ii, 1]
#         rX2 = p1[k1] * w[jj, 1] - x[ii, 2]
        rX0 = r[ii, 0] * rX0 + r[ii, 3] * rX1 + r[ii, 5] * rX2
        rX1 = r[ii, 1] * rX1 + r[ii, 4] * rX2
        rX2 = r[ii, 2] * rX2

        s1 = 1 / c_sqrt(rT0 * rT0 + rT1 * rT1 + rT2 * rT2)
        s2 = rT0 * rX0 + rT1 * rX1 + rT2 * rX2
        inter = (s2 * s2 * s1 * s1 - (rX0 * rX0 + rX1 * rX1 + rX2 * rX2)) / 2
        if inter > -20:
            tmp += s0[ii] * s1 * c_exp(inter)
    R[jj, k0, k1] = tmp


@numba.jit(["void(T[:],T[:,:],T[:,:],T[:,:],T[:,:],T[:],T[:],T[:,:,:],T[:])".replace('T', T) for T in ['f4', 'f8']], target='cpu', cache=True, nopython=True)
def GaussRadon_DI_CPU(I, x, r, t, w, p0, p1, C, DR):
    s0 = sqrt(2 * pi)
    DR[:] = 0
    for jj in range(w.shape[0]):
        for k0 in range(p0.shape[0]):
            for k1 in range(p1.shape[0]):
                for ii in range(x.shape[0]):
                    rT = [r[ii, 0] * t[jj, 0] + r[ii, 3] * t[jj, 1],
                          r[ii, 1] * t[jj, 1], 0]
                    rX = [p0[k0] * w[jj, 0] - x[ii, 0],
                          p0[k0] * w[jj, 1] - x[ii, 1],
                          p1[k1] - x[ii, 2]]
                    rX = [r[ii, 0] * rX[0] + r[ii, 3] * rX[1] + r[ii, 5] * rX[2],
                          r[ii, 1] * rX[1] + r[ii, 4] * rX[2],
                          r[ii, 2] * rX[2]]

                    # s1 = 1/|r\theta|, s2 = r\theta\cdot r(p-x)
                    s1 = 1 / sqrt(rT[0] * rT[0] + rT[1]
                                  * rT[1] + rT[2] * rT[2])
                    s2 = rT[0] * rX[0] + rT[1] * rX[1] + rT[2] * rX[2]
                    inter = (s1 * s1 * s2 * s2
                             - (rX[0] * rX[0] + rX[1] * rX[1] + rX[2] * rX[2])) / 2
                    if inter > -20:
                        DR[ii] += s0 * s1 * exp(inter) * C[jj, k0, k1]


def GaussRadon_DI_GPU(I, x, r, t, w, p0, p1, C, DR):
    s0 = sqrt(2 * pi)
    grid = x.shape[0]
    if BATCH:
        sz = [w.shape[0], p0.shape[0], p1.shape[0],
              (w.shape[0] * p0.shape[0] * p1.shape[0]) // THREADS]
        if sz[3] * THREADS < sz[0] * sz[1] * sz[2]:
            sz[3] += 1
        __GaussRadon_DI_GPU[grid, THREADS](
            s0, x, r, t, w, p0, p1, C, DR, array(sz, dtype='int32'))
    else:
        c = context()
        d = c.copy(DR)
        c.set(DR[:], 0)
        sz = [1, p0.shape[0], p1.shape[0],
              (p0.shape[0] * p1.shape[0]) // THREADS]
        if sz[3] * THREADS < sz[0] * sz[1] * sz[2]:
            sz[3] += 1
        sz = array(sz, dtype='int32')
        for i in range(w.shape[0]):
            __GaussRadon_DI_GPU[grid, THREADS](
                s0, x, r, t[i:i + 1], w[i:i + 1], p0, p1,
                C[i:i + 1], d, sz)
            c.add(DR, d, DR)


@cuda.jit("void(f4,f4[:,:],f4[:,:],f4[:,:],f4[:,:],f4[:],f4[:],f4[:,:,:],f4[:],i4[:])", cache=True)
def __GaussRadon_DI_GPU(s0, x, r, t, w, p0, p1, C, DR, sz):
    buf = cuda.shared.array(THREADS, dtype=numba.float32)
    ii = cuda.blockIdx.x
    cc = cuda.threadIdx.x

    sum0 = 0
    for indx in range(sz[3] * cc, sz[3] * (cc + 1)):
        jj = indx // (sz[1] * sz[2])
        k0 = (indx // sz[1]) - sz[2] * jj
        k1 = indx - sz[1] * (sz[2] * jj + k0)
        if jj < sz[0]:
            rT0 = r[ii, 0] * t[jj, 0] + r[ii, 3] * t[jj, 1]
            rT1 = r[ii, 1] * t[jj, 1]
            rT2 = 0
            rX0 = p0[k0] * w[jj, 0] - x[ii, 0]
            rX1 = p0[k0] * w[jj, 1] - x[ii, 1]
            rX2 = p1[k1] - x[ii, 2]
            rX0 = r[ii, 0] * rX0 + r[ii, 3] * rX1 + r[ii, 5] * rX2
            rX1 = r[ii, 1] * rX1 + r[ii, 4] * rX2
            rX2 = r[ii, 2] * rX2

            s1 = 1 / c_sqrt(rT0 * rT0 + rT1 * rT1 + rT2 * rT2)
            s2 = rT0 * rX0 + rT1 * rX1 + rT2 * rX2
            inter = (s2 * s2 * s1 * s1 -
                     (rX0 * rX0 + rX1 * rX1 + rX2 * rX2)) / 2
            if inter > -20:
                sum0 += s0 * s1 * c_exp(inter) * C[jj, k0, k1]
    buf[cc] = sum0
    cuda.syncthreads()

    __GPU_reduce_1(buf)
    if cc == 0:
        DR[ii] = buf[0]


@numba.jit(["void(T[:],T[:,:],T[:,:],T[:,:],T[:,:],T[:],T[:],T[:,:,:],T[:,:])".replace('T', T) for T in ['f4', 'f8']], target='cpu', cache=True, nopython=True)
def GaussRadon_Dx_CPU(I, x, r, t, w, p0, p1, C, DR):
    '''
    u(y) = I*exp(-|y-x|^2/(2r^2))
    Ru(w,p) = \int_{y = t*w + p} u(y)
            = I*r*sqrt(2\pi) exp(-|P_{w^\perp}(p-x)|^2/(2r^2))
            = I*r*sqrt(2\pi) exp(-(<w0,p-x>^2+<w1,p-x>^2)/(2r^2))
    DR(w,p) = I*r*sqrt(2\pi) exp(-(<w0,p-x>^2+<w1,p-x>^2)/(2r^2))*(<w0,p-x>w0+<w1,p-x>w1)/r^2
            = I*sqrt(2\pi)/r (<w0,p-x>w0+<w1,p-x>w1)exp(-(<w0,p-x>^2+<w1,p-x>^2)/(2r^2))
    '''
    s0 = I * sqrt(2 * pi)
    DR[:] = 0
    for jj in range(w.shape[0]):
        for k0 in range(p0.shape[0]):
            for k1 in range(p1.shape[0]):
                for ii in range(x.shape[0]):
                    rT = [r[ii, 0] * t[jj, 0] + r[ii, 3] * t[jj, 1],
                          r[ii, 1] * t[jj, 1], 0]
                    rX = [p0[k0] * w[jj, 0] - x[ii, 0],
                          p0[k0] * w[jj, 1] - x[ii, 1],
                          p1[k1] - x[ii, 2]]
                    rX = [r[ii, 0] * rX[0] + r[ii, 3] * rX[1] + r[ii, 5] * rX[2],
                          r[ii, 1] * rX[1] + r[ii, 4] * rX[2],
                          r[ii, 2] * rX[2]]

                    # s1 = 1/|r\theta|, s2 = r\theta\cdot r(p-x)
                    s1 = 1 / sqrt(rT[0] * rT[0] + rT[1]
                                  * rT[1] + rT[2] * rT[2])
                    s2 = rT[0] * rX[0] + rT[1] * rX[1] + rT[2] * rX[2]
                    inter = (s1 * s1 * s2 * s2
                             - (rX[0] * rX[0] + rX[1] * rX[1] + rX[2] * rX[2])) / 2
                    if inter > -20:
                        inter = s0[ii] * s1 * exp(inter) * C[jj, k0, k1]
                        rX[0] = rX[0] - s1 * s1 * s2 * rT[0]
                        rX[1] = rX[1] - s1 * s1 * s2 * rT[1]
                        rX[2] = rX[2] - s1 * s1 * s2 * rT[2]

                        DR[ii, 0] += inter * r[ii, 0] * rX[0]
                        DR[ii, 1] += inter * \
                            (r[ii, 3] * rX[0] + r[ii, 1] * rX[1])
                        DR[ii, 2] += inter * \
                            (r[ii, 5] * rX[0] + r[ii, 4]
                             * rX[1] + r[ii, 2] * rX[2])


def GaussRadon_Dx_GPU(I, x, r, t, w, p0, p1, C, DR):
    s0 = context().mul(I, sqrt(2 * pi))
    grid = x.shape[0]
    if BATCH:
        sz = [w.shape[0], p0.shape[0], p1.shape[0],
              (w.shape[0] * p0.shape[0] * p1.shape[0]) // THREADS]
        if sz[3] * THREADS < sz[0] * sz[1] * sz[2]:
            sz[3] += 1
        __GaussRadon_Dx_GPU[grid, THREADS](
            s0, x, r, t, w, p0, p1, C, DR, array(sz, dtype='int32'))
    else:
        c = context()
        d = c.copy(DR)
        c.set(DR[:], 0)
        sz = [1, p0.shape[0], p1.shape[0],
              (p0.shape[0] * p1.shape[0]) // THREADS]
        if sz[3] * THREADS < sz[0] * sz[1] * sz[2]:
            sz[3] += 1
        sz = array(sz, dtype='int32')
        for i in range(w.shape[0]):
            __GaussRadon_Dx_GPU[grid, THREADS](
                s0, x, r, t[i:i + 1], w[i:i + 1], p0, p1,
                C[i:i + 1], d, sz)
            c.add(DR, d, DR)


@cuda.jit("void(f4[:],f4[:,:],f4[:,:],f4[:,:],f4[:,:],f4[:],f4[:],f4[:,:,:],f4[:,:],i4[:])", cache=True)
def __GaussRadon_Dx_GPU(s0, x, r, t, w, p0, p1, C, DR, sz):
    buf = cuda.shared.array((THREADS, 3), dtype=numba.float32)

    ii = cuda.blockIdx.x
    cc = cuda.threadIdx.x

    sum0 = 0
    sum1 = 0
    sum2 = 0
    for indx in range(sz[3] * cc, sz[3] * (cc + 1)):
        jj = indx // (sz[1] * sz[2])
        k0 = (indx // sz[1]) - sz[2] * jj
        k1 = indx - sz[1] * (sz[2] * jj + k0)
        if jj < sz[0]:
            rT0 = r[ii, 0] * t[jj, 0] + r[ii, 3] * t[jj, 1]
            rT1 = r[ii, 1] * t[jj, 1]
            rT2 = 0
            rX0 = p0[k0] * w[jj, 0] - x[ii, 0]
            rX1 = p0[k0] * w[jj, 1] - x[ii, 1]
            rX2 = p1[k1] - x[ii, 2]
            rX0 = r[ii, 0] * rX0 + r[ii, 3] * rX1 + r[ii, 5] * rX2
            rX1 = r[ii, 1] * rX1 + r[ii, 4] * rX2
            rX2 = r[ii, 2] * rX2

            s1 = 1 / c_sqrt(rT0 * rT0 + rT1 * rT1 + rT2 * rT2)
            s2 = rT0 * rX0 + rT1 * rX1 + rT2 * rX2
            inter = (s2 * s2 * s1 * s1 -
                     (rX0 * rX0 + rX1 * rX1 + rX2 * rX2)) / 2
            if inter > -20:
                inter = s0[ii] * s1 * c_exp(inter) * C[jj, k0, k1]

                rX0 = rX0 - s1 * s1 * s2 * rT0
                rX1 = rX1 - s1 * s1 * s2 * rT1
                rX2 = rX2 - s1 * s1 * s2 * rT2

                sum0 += inter * r[ii, 0] * rX0
                sum1 += inter * (r[ii, 3] * rX0 + r[ii, 1] * rX1)
                sum2 += inter * (r[ii, 5] * rX0 + r[ii, 4]
                                 * rX1 + r[ii, 2] * rX2)
    buf[cc, 0] = sum0
    buf[cc, 1] = sum1
    buf[cc, 2] = sum2
    cuda.syncthreads()

    __GPU_reduce_3(buf)
    if cc == 0:
        DR[ii, 0] = buf[0, 0]
        DR[ii, 1] = buf[0, 1]
        DR[ii, 2] = buf[0, 2]


@numba.jit(["void(T[:],T[:,:],T[:,:],T[:,:],T[:,:],T[:],T[:],T[:,:,:],T[:,:])".replace('T', T) for T in ['f4', 'f8']], target='cpu', cache=True, nopython=True)
def GaussRadon_Dr_CPU(I, x, r, t, w, p0, p1, C, DR):
    s0 = sqrt(2 * pi) * I
    for ii in range(x.shape[0]):
        DR[ii, :] = 0
        for jj in range(w.shape[0]):
            for k0 in range(p0.shape[0]):
                for k1 in range(p1.shape[0]):
                    rT = [r[ii, 0] * t[jj, 0] + r[ii, 3] * t[jj, 1],
                          r[ii, 1] * t[jj, 1], 0]
                    X = [p0[k0] * w[jj, 0] - x[ii, 0],
                         p0[k0] * w[jj, 1] - x[ii, 1],
                         p1[k1] - x[ii, 2]]
                    rX = [r[ii, 0] * X[0] + r[ii, 3] * X[1] + r[ii, 5] * X[2],
                          r[ii, 1] * X[1] + r[ii, 4] * X[2],
                          r[ii, 2] * X[2]]

                    # s1 = 1/|r\theta|, s2 = r\theta\cdot r(p-x)
                    s1 = 1 / sqrt(rT[0] * rT[0] + rT[1]
                                  * rT[1] + rT[2] * rT[2])
                    s2 = rT[0] * rX[0] + rT[1] * rX[1] + rT[2] * rX[2]
                    inter = (s1 * s1 * s2 * s2
                             - (rX[0] * rX[0] + rX[1] * rX[1] + rX[2] * rX[2])) / 2
                    if inter > -20:
                        inter = s0[ii] * s1 * exp(inter) * C[jj, k0, k1]
                        s3 = [s2 * s1 * s1, -(s2 * s2 * s1 * s1 + 1) * s1 * s1]

                        # d/drij = s30*(rXi*tj + rTi*Xj) + s31*rTi*tj -rXi*Xj
                        # t = [t0,t1,0]

                        # d/dr00
                        DR[ii, 0] += inter * (s3[0] * (rX[0] * t[jj, 0] + rT[0] * X[0])
                                              + s3[1] * rT[0] * t[jj, 0] - rX[0] * X[0])
                        # d/dr11
                        DR[ii, 1] += inter * (s3[0] * (rX[1] * t[jj, 1] + rT[1] * X[1])
                                              + s3[1] * rT[1] * t[jj, 1] - rX[1] * X[1])
                        # d/dr22
                        DR[ii, 2] += inter * \
                            (s3[0] * rT[2] * X[2] - rX[2] * X[2])
                        # d/dr01
                        DR[ii, 3] += inter * (s3[0] * (rX[0] * t[jj, 1] + rT[0] * X[1])
                                              + s3[1] * rT[0] * t[jj, 1] - rX[0] * X[1])
                        # d/dr12
                        DR[ii, 4] += inter * \
                            (s3[0] * rT[1] * X[2] - rX[1] * X[2])
                        # d/dr03
                        DR[ii, 5] += inter * \
                            (s3[0] * rT[0] * X[2] - rX[0] * X[2])


def GaussRadon_Dr_GPU(I, x, r, t, w, p0, p1, C, DR):
    s0 = context().mul(I, sqrt(2 * pi))
    grid = x.shape[0]
    if BATCH:
        sz = [w.shape[0], p0.shape[0], p1.shape[0],
              (w.shape[0] * p0.shape[0] * p1.shape[0]) // THREADS]
        if sz[3] * THREADS < sz[0] * sz[1] * sz[2]:
            sz[3] += 1
        __GaussRadon_Dr_GPU[grid, THREADS](
            s0, x, r, t, w, p0, p1, C, DR, array(sz, dtype='int32'))
    else:
        c = context()
        d = c.copy(DR)
        c.set(DR[:], 0)
        sz = [1, p0.shape[0], p1.shape[0],
              (p0.shape[0] * p1.shape[0]) // THREADS]
        if sz[3] * THREADS < sz[0] * sz[1] * sz[2]:
            sz[3] += 1
        sz = array(sz, dtype='int32')
        for i in range(w.shape[0]):
            __GaussRadon_Dr_GPU[grid, THREADS](
                s0, x, r, t[i:i + 1], w[i:i + 1], p0, p1,
                C[i:i + 1], d, sz)
            c.add(DR, d, DR)


@cuda.jit("void(f4[:],f4[:,:],f4[:,:],f4[:,:],f4[:,:],f4[:],f4[:],f4[:,:,:],f4[:,:],i4[:])", cache=True)
def __GaussRadon_Dr_GPU(s0, x, r, t, w, p0, p1, C, DR, sz):
    buf = cuda.shared.array((THREADS, 6), dtype=numba.float32)
    ii = cuda.blockIdx.x
    cc = cuda.threadIdx.x

    sum0, sum1, sum2, sum3, sum4, sum5 = 0, 0, 0, 0, 0, 0
    for indx in range(sz[3] * cc, sz[3] * (cc + 1)):
        jj = indx // (sz[1] * sz[2])
        k0 = (indx // sz[1]) - sz[2] * jj
        k1 = indx - sz[1] * (sz[2] * jj + k0)
        if jj < sz[0]:
            rT0 = r[ii, 0] * t[jj, 0] + r[ii, 3] * t[jj, 1]
            rT1 = r[ii, 1] * t[jj, 1]
            rT2 = 0
            X0 = p0[k0] * w[jj, 0] - x[ii, 0]
            X1 = p0[k0] * w[jj, 1] - x[ii, 1]
            X2 = p1[k1] - x[ii, 2]
            rX0 = r[ii, 0] * X0 + r[ii, 3] * X1 + r[ii, 5] * X2
            rX1 = r[ii, 1] * X1 + r[ii, 4] * X2
            rX2 = r[ii, 2] * X2

            s1 = 1 / c_sqrt(rT0 * rT0 + rT1 * rT1 + rT2 * rT2)
            s2 = rT0 * rX0 + rT1 * rX1 + rT2 * rX2
            inter = (s2 * s2 * s1 * s1 -
                     (rX0 * rX0 + rX1 * rX1 + rX2 * rX2)) / 2
            if inter > -20:
                inter = s0[ii] * s1 * c_exp(inter) * C[jj, k0, k1]
                s30 = s2 * s1 * s1
                s31 = -(s2 * s2 * s1 * s1 + 1) * s1 * s1

                # d/drij = s30*(rXi*tj + rTi*Xj) + s31*rTi*tj -rXi*Xj

                # d/dr00
                sum0 += inter * (s30 * (rX0 * t[jj, 0] + rT0 * X0) +
                                 s31 * rT0 * t[jj, 0] - rX0 * X0)
                # d/dr11
                sum1 += inter * (s30 * (rX1 * t[jj, 1] + rT1 * X1)
                                 + s31 * rT1 * t[jj, 1] - rX1 * X1)
                # d/dr22
                sum2 += inter * (s30 * rT2 * X2 - rX2 * X2)
                # d/dr01
                sum3 += inter * (s30 * (rX0 * t[jj, 1] + rT0 * X1)
                                 + s31 * rT0 * t[jj, 1] - rX0 * X1)
                # d/dr12
                sum4 += inter * (s30 * rT1 * X2 - rX1 * X2)
                # d/dr02
                sum5 += inter * (s30 * rT0 * X2 - rX0 * X2)
    buf[cc, 0] = sum0
    buf[cc, 1] = sum1
    buf[cc, 2] = sum2
    buf[cc, 3] = sum3
    buf[cc, 4] = sum4
    buf[cc, 5] = sum5
    cuda.syncthreads()

    __GPU_reduce_n(buf, Six)
    if cc == 0:
        DR[ii, 0] = buf[0, 0]
        DR[ii, 1] = buf[0, 1]
        DR[ii, 2] = buf[0, 2]
        DR[ii, 3] = buf[0, 3]
        DR[ii, 4] = buf[0, 4]
        DR[ii, 5] = buf[0, 5]
