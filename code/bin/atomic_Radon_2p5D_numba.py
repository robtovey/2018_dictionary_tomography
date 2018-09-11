'''
Created on 17 Jan 2018

@author: Rob Tovey
'''
from numpy import sqrt, pi, exp, array
import numba
from numba import cuda
from math import exp as c_exp
from .numba_cuda_aux import __GPU_reduce_1, __GPU_reduce_3, THREADS
from .manager import context
BATCH = True


@numba.jit(["void(T[:],T[:,:],T[:,:],T[:,:],T[:,:],T[:],T[:],T[:,:,:])".replace('T', T) for T in ['f4', 'f8']], target='cpu', cache=True, nopython=True)
def GaussRadon_CPU(I, x, r, t, w, p0, p1, R):
    s1 = I * r[:, 0] * sqrt(2 * pi)
    s2 = -1 / (2 * r[:, 0]**2)
    for jj in range(w.shape[0]):
        for k0 in range(p0.shape[0]):
            for k1 in range(p1.shape[0]):
                tmp = 0
                for ii in range(x.shape[0]):
                    IP = [x[ii, 0] * w[jj, 0] + x[ii, 1] * w[jj, 1], x[ii, 2]]
                    IP = [p0[k0] - IP[0], p1[k1] - IP[1]]
                    inter = (IP[0]**2 + IP[1]**2) * s2[ii]
                    if inter > -20:
                        tmp += s1[ii] * exp(inter)
                R[jj, k0, k1] = tmp


def GaussRadon_GPU(I, x, r, t, w, p0, p1, R):
    c = context()
    s1 = c.mul(I, c.mul(r[:, 0], sqrt(2 * pi)))
    s2 = c.div(-1 / 2, c.mul(r[:, 0], r[:, 0]))
    if BATCH:
        grid = w.shape[0], p0.shape[0], p1.shape[0]
        tpb = 4
        __GaussRadon_GPU[tuple(-(-g // tpb) for g in grid),
                         (tpb, tpb, tpb)](x, s1, s2, w, p0, p1, R)
    else:
        grid = 1, p0.shape[0], p1.shape[0]
        tpb = 4
        grid = tuple(-(-g // tpb) for g in grid)
        for i in range(w.shape[0]):
            __GaussRadon_GPU[grid, (1, tpb, tpb)](
                x, s1, s2, w[i:i + 1], p0, p1, R[i:i + 1])


@cuda.jit("void(f4[:,:],f4[:],f4[:],f4[:,:],f4[:],f4[:],f4[:,:,:])", cache=True)
def __GaussRadon_GPU(x, s1, s2, w, p0, p1, R):
    jj, k0, k1 = cuda.grid(3)
    if jj >= w.shape[0] or k0 >= p0.shape[0] or k1 >= p1.shape[0]:
        return
    tmp = 0
    for ii in range(x.shape[0]):
        IP0 = p0[k0] - x[ii, 0] * w[jj, 0] - x[ii, 1] * w[jj, 1]
        IP1 = p1[k1] - x[ii, 2]
        inter = (IP0**2 + IP1**2) * s2[ii]
        if inter > -20:
            tmp += s1[ii] * c_exp(inter)
    R[jj, k0, k1] = tmp


@numba.jit(["void(T[:],T[:,:],T[:,:],T[:,:],T[:,:],T[:],T[:],T[:,:,:],T[:])".replace('T', T) for T in ['f4', 'f8']], target='cpu', cache=True, nopython=True)
def GaussRadon_DI_CPU(I, x, r, t, w, p0, p1, C, DR):
    s1 = r[:, 0] * sqrt(2 * pi)
    s2 = -1 / (2 * r[:, 0]**2)
    DR[:] = 0
    for jj in range(w.shape[0]):
        for k0 in range(p0.shape[0]):
            for k1 in range(p1.shape[0]):
                for ii in range(x.shape[0]):
                    IP = [x[ii, 0] * w[jj, 0] + x[ii, 1] * w[jj, 1], x[ii, 2]]
                    IP = [p0[k0] - IP[0], p1[k1] - IP[1]]
                    inter = (IP[0]**2 + IP[1]**2) * s2[ii]
                    if inter > -20:
                        DR[ii] += s1[ii] * \
                            exp(inter) * C[jj, k0, k1]


def GaussRadon_DI_GPU(I, x, r, t, w, p0, p1, C, DR):
    c = context()
    s1 = c.mul(r[:, 0], sqrt(2 * pi))
    s2 = c.div(-1 / 2, c.mul(r[:, 0], r[:, 0]))
    grid = x.shape[0]
    if BATCH:
        sz = [w.shape[0], p0.shape[0], p1.shape[0],
              (w.shape[0] * p0.shape[0] * p1.shape[0]) // THREADS]
        if sz[3] * THREADS < sz[0] * sz[1] * sz[2]:
            sz[3] += 1
        __GaussRadon_DI_GPU[grid, THREADS](
            x, s1, s2, w, p0, p1, C, DR, array(sz, dtype='int32'))
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
                x, s1, s2, w[i:i + 1], p0, p1,
                C[i:i + 1], d, sz)
            c.add(DR, d, DR)


@cuda.jit("void(f4[:,:],f4[:],f4[:],f4[:,:],f4[:],f4[:],f4[:,:,:],f4[:],i4[:])", cache=True)
def __GaussRadon_DI_GPU(x, s1, s2, w, p0, p1, C, DR, sz):
    buf = cuda.shared.array(THREADS, dtype=numba.float32)
    ii = cuda.blockIdx.x
    cc = cuda.threadIdx.x

    sum0 = 0
    for indx in range(sz[3] * cc, sz[3] * (cc + 1)):
        jj = indx // (sz[1] * sz[2])
        k0 = (indx // sz[1]) - sz[2] * jj
        k1 = indx - sz[1] * (sz[2] * jj + k0)
        if jj < sz[0]:
            IP0 = p0[k0] - x[ii, 0] * w[jj, 0] - x[ii, 1] * w[jj, 1]
            IP1 = p1[k1] - x[ii, 2]
            inter = (IP0**2 + IP1**2) * s2[ii]
            if inter > -20:
                sum0 += s1[ii] * c_exp(inter) * C[jj, k0, k1]
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
    s1 = I * sqrt(2 * pi) / r[:, 0]
    s2 = -1 / (2 * r[:, 0]**2)
    DR[:] = 0
    for jj in range(w.shape[0]):
        for k0 in range(p0.shape[0]):
            for k1 in range(p1.shape[0]):
                for ii in range(x.shape[0]):
                    IP = [x[ii, 0] * w[jj, 0] + x[ii, 1] * w[jj, 1], x[ii, 2]]
                    IP = [p0[k0] - IP[0], p1[k1] - IP[1]]
                    inter = (IP[0]**2 + IP[1]**2) * s2[ii]
                    if inter > -20:
                        inter = s1[ii] * exp(inter) * C[jj, k0, k1]

                        DR[ii, 0] += inter * IP[0] * w[jj, 0]
                        DR[ii, 1] += inter * IP[0] * w[jj, 1]
                        DR[ii, 2] += inter * IP[1]


def GaussRadon_Dx_GPU(I, x, r, t, w, p0, p1, C, DR):
    c = context()
    s1 = c.mul(I, c.div(sqrt(2 * pi), r[:, 0]))
    s2 = c.div(-1 / 2, c.mul(r[:, 0], r[:, 0]))
    grid = x.shape[0]
    if BATCH:
        sz = [w.shape[0], p0.shape[0], p1.shape[0],
              (w.shape[0] * p0.shape[0] * p1.shape[0]) // THREADS]
        if sz[3] * THREADS < sz[0] * sz[1] * sz[2]:
            sz[3] += 1
        __GaussRadon_Dx_GPU[grid, THREADS](
            x, s1, s2, w, p0, p1, C, DR, array(sz, dtype='int32'))
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
                x, s1, s2, w[i:i + 1], p0, p1,
                C[i:i + 1], d, sz)
            c.add(DR, d, DR)


@cuda.jit("void(f4[:,:],f4[:],f4[:],f4[:,:],f4[:],f4[:],f4[:,:,:],f4[:,:],i4[:])", cache=True)
def __GaussRadon_Dx_GPU(x, s1, s2, w, p0, p1, C, DR, sz):
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
            IP0 = p0[k0] - x[ii, 0] * w[jj, 0] - x[ii, 1] * w[jj, 1]
            IP1 = p1[k1] - x[ii, 2]
            inter = (IP0**2 + IP1**2) * s2[ii]
            if inter > -20:
                inter = s1[ii] * c_exp(inter) * C[jj, k0, k1]

                sum0 += inter * IP0 * w[jj, 0]
                sum1 += inter * IP0 * w[jj, 1]
                sum2 += inter * IP1
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
    s1 = I * (2 * sqrt(2 * pi))
    s2 = -1 / (2 * r[:, 0]**2)
    for ii in range(x.shape[0]):
        DR[ii, 0] = 0
        for jj in range(w.shape[0]):
            IP = [x[ii, 0] * w[jj, 0] + x[ii, 1] * w[jj, 1], x[ii, 2]]
            for k0 in range(p0.shape[0]):
                for k1 in range(p1.shape[0]):
                    inter = ((p0[k0] - IP[0])**2 +
                             (p1[k1] - IP[1])**2) * s2[ii]
                    if inter > -20:
                        DR[ii, 0] += s1[ii] * \
                            exp(inter) * C[jj, k0, k1] * (0.5 - inter)


def GaussRadon_Dr_GPU(I, x, r, t, w, p0, p1, C, DR):
    c = context()
    s1 = c.mul(I, 2 * sqrt(2 * pi))
    s2 = c.div(-1 / 2, c.mul(r[:, 0], r[:, 0]))
    grid = x.shape[0]
    if BATCH:
        sz = [w.shape[0], p0.shape[0], p1.shape[0],
              (w.shape[0] * p0.shape[0] * p1.shape[0]) // THREADS]
        if sz[3] * THREADS < sz[0] * sz[1] * sz[2]:
            sz[3] += 1
        __GaussRadon_Dr_GPU[grid, THREADS](
            x, s1, s2, w, p0, p1, C, DR[:, 0], array(sz, dtype='int32'))
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
                x, s1, s2, w[i:i + 1], p0, p1,
                C[i:i + 1], d[:, 0], sz)
            c.add(DR, d, DR)


@cuda.jit("void(f4[:,:],f4[:],f4[:],f4[:,:],f4[:],f4[:],f4[:,:,:],f4[:],i4[:])", cache=True)
def __GaussRadon_Dr_GPU(x, s1, s2, w, p0, p1, C, DR, sz):
    buf = cuda.shared.array(THREADS, dtype=numba.float32)
    ii = cuda.blockIdx.x
    cc = cuda.threadIdx.x

    sum0 = 0
    for indx in range(sz[3] * cc, sz[3] * (cc + 1)):
        jj = indx // (sz[1] * sz[2])
        k0 = (indx // sz[1]) - sz[2] * jj
        k1 = indx - sz[1] * (sz[2] * jj + k0)
        if jj < sz[0]:
            IP0 = p0[k0] - x[ii, 0] * w[jj, 0] - x[ii, 1] * w[jj, 1]
            IP1 = p1[k1] - x[ii, 2]
            inter = (IP0**2 + IP1**2) * s2[ii]
            if inter > -20:
                sum0 += s1[ii] * c_exp(inter) * C[jj, k0, k1] * (0.5 - inter)
    buf[cc] = sum0
    cuda.syncthreads()

    __GPU_reduce_1(buf)
    if cc == 0:
        DR[ii] = buf[0]
