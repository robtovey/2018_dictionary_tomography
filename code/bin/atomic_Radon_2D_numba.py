'''
Created on 17 Jan 2018

@author: Rob Tovey
'''
from numpy import sqrt, pi, exp, array
import numba
from numba import cuda
from math import exp as c_exp
from .numba_cuda_aux import __GPU_reduce_1, __GPU_reduce_2, THREADS
from .manager import context
BATCH = True


@numba.jit(["void(T[:],T[:,:],T[:,:],T[:,:],T[:,:],T[:],T[:,:])".replace('T', T) for T in ['f4', 'f8']], target='cpu', cache=True, nopython=True)
def GaussRadon_CPU(I, x, r, t, w, p, R):
    '''
    u(y) = I*exp(-|y-x|^2/(2r^2))
    Ru(w,p) = \int_{y = t*w + p} u(y)
            = I*r*sqrt(2\pi) exp(-|P_{w^\perp}(p-x)|^2/(2r^2))
            = I*r*sqrt(2\pi) exp(-<w^\perp,p-x>^2/(2r^2))
    DR(w,p) = I*r*sqrt(2\pi) exp(-<w^\perp,p-x>^2/(2r^2))*(-<w^\perp,p-x>/r^2)*(-w^\perp)
            = I*sqrt(2\pi)/r <w^\perp,p-x>exp(-<w^\perp,p-x>^2/(2r^2))w^perp
    '''
    s1 = I * r[:, 0] * sqrt(2 * pi)
    s2 = -1 / (2 * r[:, 0]**2)
    for jj in range(w.shape[0]):
        for kk in range(p.shape[0]):
            tmp = 0
            for ii in range(x.shape[0]):
                # orthogonalise w inline
                inter = (p[kk] - x[ii, 0] * w[jj, 0] -
                         x[ii, 1] * w[jj, 1])**2 * s2[ii]
                if inter > -30:
                    tmp += s1[ii] * exp(inter)
            R[jj, kk] = tmp


def GaussRadon_GPU(I, x, r, t, w, p, R):
    c = context()
    s1 = c.mul(I, c.mul(r[:, 0], sqrt(2 * pi)))
    s2 = c.div(-1 / 2, c.mul(r[:, 0], r[:, 0]))
    if BATCH:
        grid = w.shape[0], p.shape[0]
        tpb = 4
        __GaussRadon_GPU[tuple(-(-g // tpb) for g in grid),
                         (tpb, tpb)](x, s1, s2, w, p, R)
    else:
        grid = 1, p.shape[0]
        tpb = 4
        grid = tuple(-(-g // tpb) for g in grid)
        for i in range(w.shape[0]):
            __GaussRadon_GPU[grid, (1, tpb)](
                x, s1, s2, w[i:i + 1], p, R[i:i + 1])


@cuda.jit("void(f4[:,:],f4[:],f4[:],f4[:,:],f4[:],f4[:,:])", cache=True)
def __GaussRadon_GPU(x, s1, s2, w, p, R):
    jj, kk = cuda.grid(2)
    if jj >= w.shape[0] or kk >= p.shape[0]:
        return
    tmp = 0
    for ii in range(x.shape[0]):
        inter = (p[kk] - x[ii, 0] * w[jj, 0] -
                 x[ii, 1] * w[jj, 1])**2 * s2[ii]
        if inter > -20:
            tmp += s1[ii] * c_exp(inter)
    R[jj, kk] = tmp


@numba.jit(["void(T[:],T[:,:],T[:,:],T[:,:],T[:,:],T[:],T[:,:],T[:])".replace('T', T) for T in ['f4', 'f8']], target='cpu', cache=True, nopython=True)
def GaussRadon_DI_CPU(I, x, r, t, w, p, C, DR):
    s1 = r[:, 0] * sqrt(2 * pi)
    s2 = -1 / (2 * r[:, 0]**2)
    for ii in range(x.shape[0]):
        DR[ii] = 0
        for jj in range(w.shape[0]):
            # orthogonalise w inline
            tmp = x[ii, 0] * w[jj, 0] + x[ii, 1] * w[jj, 1]
            for kk in range(p.shape[0]):
                inter = (p[kk] - tmp)**2 * s2[ii]
                if inter > -20:
                    DR[ii] += s1[ii] * exp(inter) * C[jj, kk]


def GaussRadon_DI_GPU(I, x, r, t, w, p, C, DR):
    c = context()
    s1 = c.mul(r[:, 0], sqrt(2 * pi))
    s2 = c.div(-1 / 2, c.mul(r[:, 0], r[:, 0]))
    grid = x.shape[0]
    if BATCH:
        sz = [w.shape[0], p.shape[0], (w.shape[0] * p.shape[0]) // THREADS]
        if sz[2] * THREADS < sz[0] * sz[1]:
            sz[2] += 1
        __GaussRadon_DI_GPU[grid, THREADS](
            x, s1, s2, w, p, C, DR, array(sz, dtype='int32'))
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
                x, s1, s2, w[i:i + 1], p,
                C[i:i + 1], d, sz)
            c.add(DR, d, DR)


@cuda.jit("void(f4[:,:],f4[:],f4[:],f4[:,:],f4[:],f4[:,:],f4[:],i4[:])", cache=True)
def __GaussRadon_DI_GPU(x, s1, s2, w, p, C, DR, sz):
    buf = cuda.shared.array(THREADS, dtype=numba.float32)
    ii = cuda.blockIdx.x
    cc = cuda.threadIdx.x

    sum0 = 0
    for indx in range(sz[2] * cc, sz[2] * (cc + 1)):
        jj = indx // sz[1]
        kk = indx - jj * sz[1]
        if jj < sz[0]:
            inter = (p[kk] - x[ii, 0] * w[jj, 0] -
                     x[ii, 1] * w[jj, 1])**2 * s2[ii]
            if inter > -20:
                sum0 += s1[ii] * c_exp(inter) * C[jj, kk]
    buf[cc] = sum0
    cuda.syncthreads()

    __GPU_reduce_1(buf)
    if cc == 0:
        DR[ii] = buf[0]


@numba.jit(["void(T[:],T[:,:],T[:,:],T[:,:],T[:,:],T[:],T[:,:],T[:,:])".replace('T', T) for T in ['f4', 'f8']], target='cpu', cache=True, nopython=True)
def GaussRadon_Dx_CPU(I, x, r, t, w, p, C, DR):
    '''
    u(y) = I*exp(-|y-x|^2/(2r^2))
    Ru(w,p) = \int_{y = t*w + p} u(y)
            = I*r*sqrt(2\pi) exp(-|P_{w^\perp}(p-x)|^2/(2r^2))
            = I*r*sqrt(2\pi) exp(-<w^\perp,p-x>^2/(2r^2))
    DR(w,p) = I*r*sqrt(2\pi) exp(-<w^\perp,p-x>^2/(2r^2))*(-<w^\perp,p-x>/r^2)*(-w^\perp)
            = I*sqrt(2\pi)/r <w^\perp,p-x>exp(-<w^\perp,p-x>^2/(2r^2))w^perp
    '''
    s1 = I * sqrt(2 * pi) / r[:, 0]
    s2 = -1 / (2 * r[:, 0]**2)
    for ii in range(x.shape[0]):
        DR[ii, :] = 0
        for jj in range(w.shape[0]):
            tmp = x[ii, 0] * w[jj, 0] + x[ii, 1] * w[jj, 1]
            for kk in range(p.shape[0]):
                IP = p[kk] - tmp
                inter = IP**2 * s2[ii]
    # inter = -|P_{w^\perp}(x-p)|^2/(2r^2)
    # inter = (I*r*sqrt(2\pi)/r**2)exp(-|P_{w^\perp}(x-p)|^2/(2r^2))
                if inter > -30:
                    inter = s1[ii] * IP * exp(inter) * C[jj, kk]
                    DR[ii, 0] += inter * w[jj, 0]
                    DR[ii, 1] += inter * w[jj, 1]


def GaussRadon_Dx_GPU(I, x, r, t, w, p, C, DR):
    c = context()
    s1 = c.mul(I, c.div(sqrt(2 * pi), r[:, 0]))
    s2 = c.div(-1 / 2, c.mul(r[:, 0], r[:, 0]))
    grid = x.shape[0]
    if BATCH:
        sz = [w.shape[0], p.shape[0], (w.shape[0] * p.shape[0]) // THREADS]
        if sz[2] * THREADS < sz[0] * sz[1]:
            sz[2] += 1
        __GaussRadon_Dx_GPU[grid, THREADS](
            x, s1, s2, w, p, C, DR, array(sz, dtype='int32'))
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
                x, s1, s2, w[i:i + 1], p,
                C[i:i + 1], d, sz)
            c.add(DR, d, DR)


@cuda.jit("void(f4[:,:],f4[:],f4[:],f4[:,:],f4[:],f4[:,:],f4[:,:],i4[:])", cache=True)
def __GaussRadon_Dx_GPU(x, s1, s2, w, p, C, DR, sz):
    buf = cuda.shared.array((THREADS, 2), dtype=numba.float32)

    ii = cuda.blockIdx.x
    cc = cuda.threadIdx.x

    sum0 = 0
    sum1 = 0
    for indx in range(sz[2] * cc, sz[2] * (cc + 1)):
        jj = indx // sz[1]
        kk = indx - jj * sz[1]
        if jj < sz[0]:
            IP = p[kk] - x[ii, 0] * w[jj, 0] - x[ii, 1] * w[jj, 1]
            inter = IP**2 * s2[ii]
            if inter > -20:
                inter = IP * s1[ii] * c_exp(inter) * C[jj, kk]
                sum0 += inter * w[jj, 0]
                sum1 += inter * w[jj, 1]
    buf[cc, 0] = sum0
    buf[cc, 1] = sum1
    cuda.syncthreads()

    __GPU_reduce_2(buf)
    if cc == 0:
        DR[ii, 0] = buf[0, 0]
        DR[ii, 1] = buf[0, 1]


@numba.jit(["void(T[:],T[:,:],T[:,:],T[:,:],T[:,:],T[:],T[:,:],T[:,:])".replace('T', T) for T in ['f4', 'f8']], target='cpu', cache=True, nopython=True)
def GaussRadon_Dr_CPU(I, x, r, t, w, p, C, DR):
    s1 = I * sqrt(2 * pi)
    s2 = -1 / (2 * r[:, 0]**2)
    for ii in range(x.shape[0]):
        DR[ii, :] = 0
        for jj in range(w.shape[0]):
            tmp = x[ii, 0] * w[jj, 0] + x[ii, 1] * w[jj, 1]
            for kk in range(p.shape[0]):
                inter = (p[kk] - tmp)**2 * s2[ii]
                if inter > -20:
                    DR[ii, 0] += s1[ii] * \
                        exp(inter) * C[jj, kk] * (1 - 2 * inter)


def GaussRadon_Dr_GPU(I, x, r, t, w, p, C, DR):
    c = context()
    s1 = c.mul(I, 2 * sqrt(2 * pi))
    s2 = c.div(-1 / 2, c.mul(r[:, 0], r[:, 0]))
    grid = x.shape[0]
    if BATCH:
        sz = [w.shape[0], p.shape[0], (w.shape[0] * p.shape[0]) // THREADS]
        if sz[2] * THREADS < sz[0] * sz[1]:
            sz[2] += 1
        __GaussRadon_Dr_GPU[grid, THREADS](
            x, s1, s2, w, p, C, DR[:, 0], array(sz, dtype='int32'))
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
                x, s1, s2, w[i:i + 1], p,
                C[i:i + 1], d[:, 0], sz)
            c.add(DR, d, DR)


@cuda.jit("void(f4[:,:],f4[:],f4[:],f4[:,:],f4[:],f4[:,:],f4[:],i4[:])", cache=True)
def __GaussRadon_Dr_GPU(x, s1, s2, w, p, C, DR, sz):
    buf = cuda.shared.array(THREADS, dtype=numba.float32)
    ii = cuda.blockIdx.x
    cc = cuda.threadIdx.x

    sum0 = 0
    for indx in range(sz[2] * cc, sz[2] * (cc + 1)):
        jj = indx // sz[1]
        kk = indx - jj * sz[1]
        if jj < sz[0]:
            inter = (p[kk] - x[ii, 0] * w[jj, 0] -
                     x[ii, 1] * w[jj, 1])**2 * s2[ii]
            if inter > -20:
                sum0 += s1[ii] * c_exp(inter) * C[jj, kk] * (0.5 - inter)
    buf[cc] = sum0
    cuda.syncthreads()

    __GPU_reduce_1(buf)
    if cc == 0:
        DR[ii] = buf[0]


@numba.jit(["void(T[:],T[:,:],T[:,:],T[:],T[:],T[:,:])".replace('T', T) for T in ['f4', 'f8']], target='cpu', cache=True, nopython=True)
def GaussVol_CPU(I, x, r, y0, y1, u):
    s = -1 / (2 * r[:, 0]**2)
    for jj in range(y0.shape[0]):
        for kk in range(y1.shape[0]):
            tmp = 0
            for ii in range(x.shape[0]):
                interior = ((y0[jj] - x[ii, 0])**2 +
                            (y1[kk] - x[ii, 1])**2) * s[ii]
                if interior > -20:
                    tmp += I[ii] * exp(interior)
            u[jj, kk] = tmp


def GaussVol_GPU(I, x, r, y0, y1, u):
    c = context()
    s = c.div(-1 / 2, c.mul(r[:, 0], r[:, 0]))
    if BATCH:
        grid = y0.shape[0], y1.shape[0]
        tpb = 4
        __GaussVol_GPU[tuple(-(-g // tpb) for g in grid), (tpb, tpb)
                       ](I, x, s, y0, y1, u)
    else:
        grid = 1, y1.shape[0]
        tpb = 4
        grid = tuple(-(-g // tpb) for g in grid)
        for i in range(y0.shape[0]):
            __GaussVol_GPU[grid, (1, tpb)
                           ](I, x, s, y0[i:i + 1], y1, u[i:i + 1])


@cuda.jit("void(f4[:],f4[:,:],f4[:],f4[:],f4[:],f4[:,:])", cache=True)
def __GaussVol_GPU(I, x, s, y0, y1, u):
    jj, kk = cuda.grid(2)
    tmp = 0
    for ii in range(x.shape[0]):
        interior = ((y0[jj] - x[ii, 0])**2 +
                    (y1[kk] - x[ii, 1])**2) * s[ii]
        if interior > -20:
            tmp += I[ii] * c_exp(interior)
    u[jj, kk] = tmp
