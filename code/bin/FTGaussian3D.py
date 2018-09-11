'''
Created on 6 Jul 2018

@author: Rob Tovey
'''
from numpy import empty, array
from numba import cuda, float32 as f4, int32 as i4, complex64 as c8
from cmath import exp as c_exp
from .numba_cuda_aux import THREADS, __GPU_reduce_n


def evaluate(atoms, S):
    iso = atoms.space.isotropic

    if len(S) == 4:
        # 3D hyperplane sample
        k0, k1, w0, w1 = S
        out = empty((w0.shape[0], len(k0), len(k1)), dtype='c8')
        grid, tpb = out.shape, 4
        grid = tuple(-(-g // tpb) for g in grid)
        if iso:
            __hyperplane_eval_iso[grid, (tpb, tpb, tpb)](
                atoms.I, atoms.x, atoms.r, k0, k1, w0, w1, out)
        else:
            __hyperplane_eval_aniso[grid, (tpb, tpb, tpb)](
                atoms.I, atoms.x, atoms.r, k0, k1, w0, w1, out)
    elif S[2].ndim == 2:
        # 2p5D hyperplane sample
        k0, k1, w0 = S
        out = empty((w0.shape[0], len(k0), len(k1)), dtype='c8')
        grid, tpb = out.shape, 4
        grid = tuple(-(-g // tpb) for g in grid)
        if iso:
            __hyperplane_eval_iso_2p5D[grid, (tpb, tpb, tpb)](
                atoms.I, atoms.x, atoms.r, k0, k1, w0, out)
        else:
            __hyperplane_eval_aniso_2p5D[grid, (tpb, tpb, tpb)](
                atoms.I, atoms.x, atoms.r, k0, k1, w0, out)
    else:
        # 3D grid sample
        k0, k1, k2 = S
        out = empty((len(k0), len(k1), len(k2)), dtype='c8')
        grid, tpb = out.shape, 4
        grid = tuple(-(-g // tpb) for g in grid)
        if iso:
            __grid_eval_iso[grid, (tpb, tpb, tpb)](
                atoms.I, atoms.x, atoms.r, k0, k1, k2, out)
        else:
            __grid_eval_aniso[grid, (tpb, tpb, tpb)](
                atoms.I, atoms.x, atoms.r, k0, k1, k2, out)

    return out


def derivs(atoms, S, C):
    iso = atoms.space.isotropic
    k0, k1, w0 = S[:3]

    if hasattr(C, 'asarray'):
        C = C.asarray()

    block = min(w0.shape[0], 1000)
    sz = [w0.shape[0], len(k0), len(k1),
          (w0.shape[0] * len(k0) * len(k1)) // THREADS]
    if sz[3] * THREADS < sz[0] * sz[1] * sz[2]:
        sz[3] += 1

    f, Df, DDf = empty((block,), dtype='f4'), empty(
        (10, block), dtype='f4'), empty((10, 10, block), dtype='f4')

    if len(S) == 4:
        # 3D hyperplane sample
        w1 = S[3]
        if iso:
            __derivs_iso[block, THREADS](
                atoms.I, atoms.x, atoms.r, k0, k1, w0, w1, C, f, Df, DDf, array(sz, dtype='i4'))
        else:
            __derivs_aniso[block, THREADS](
                atoms.I, atoms.x, atoms.r, k0, k1, w0, w1, C, f, Df, DDf, array(sz, dtype='i4'))
    elif S[2].ndim == 2:
        # 2p5D hyperplane sample
        if iso:
            __derivs_iso_2p5D[block, THREADS](
                atoms.I, atoms.x, atoms.r, k0, k1, w0, C, f, Df, DDf, array(sz, dtype='i4'))
        else:
            __derivs_aniso_2p5D[block, THREADS](
                atoms.I, atoms.x, atoms.r, k0, k1, w0, C, f, Df, DDf, array(sz, dtype='i4'))
    else:
        # 3D grid sample
        raise NotImplemented

    return f.sum(-1), Df.sum(-1), DDf.sum(-1)


@cuda.jit("void(f4[:],f4[:,:],f4[:,:],f4[:],f4[:],f4[:], c8[:,:,:])", cache=True)
def __grid_eval_aniso(I, x, r, k0, k1, k2, arr):
    '''
    I*exp(-|rk|^2/2)*exp(-(x\cdot k)i)
    '''
    i0, i1, i2 = cuda.grid(3)
    tmp = 0
    for ii in range(x.shape[0]):
        rk0 = r[ii, 0] * k0[i0]
        rk1 = r[ii, 3] * k0[i0] + r[ii, 1] * k1[i1]
        rk2 = r[ii, 5] * k0[i0] + r[ii, 4] * k1[i1] + r[ii, 2] * k2[i2]
        norm = rk0 * rk0 + rk1 * rk1 + rk2 * rk2
        if norm < 30:
            IP = x[ii, 0] * k0[i0] + x[ii, 1] * k1[i1] + x[ii, 2] * k2[i2]
            tmp += I[ii] * c_exp(-.5 * norm) * c_exp(complex(0, -IP))
    arr[i0, i1, i2] = tmp


@cuda.jit("void(f4[:],f4[:,:],f4[:,:],f4[:],f4[:],f4[:,:],f4[:,:], c8[:,:,:])", cache=True)
def __hyperplane_eval_aniso(I, x, r, k0, k1, w0, w1, arr):
    ''' k[i0,i1,i2] = k0[i1]*w0[i0] + k1[i2]*w1[i0] '''
    i0, i1, i2 = cuda.grid(3)
    tmp = 0
    K0 = k0[i1] * w0[i0, 0] + k1[i2] * w1[i0, 0]
    K1 = k0[i1] * w0[i0, 1] + k1[i2] * w1[i0, 1]
    K2 = k0[i1] * w0[i0, 2] + k1[i2] * w1[i0, 2]
    for ii in range(x.shape[0]):
        rk0 = r[ii, 0] * K0
        rk1 = r[ii, 3] * K0 + r[ii, 1] * K1
        rk2 = r[ii, 5] * K0 + r[ii, 4] * K1 + r[ii, 2] * K2
        norm = rk0 * rk0 + rk1 * rk1 + rk2 * rk2
        if norm < 30:
            IP = x[ii, 0] * K0 + x[ii, 1] * K1 + x[ii, 2] * K2
            tmp += I[ii] * c_exp(-.5 * norm) * c_exp(complex(0, -IP))
    arr[i0, i1, i2] = tmp


@cuda.jit("void(f4[:],f4[:,:],f4[:,:],f4[:],f4[:],f4[:,:], c8[:,:,:])", cache=True)
def __hyperplane_eval_aniso_2p5D(I, x, r, k0, k1, w0, arr):
    ''' k[i0,i1,i2] = k0[i1]*w0[i0] + k1[i2]*w1[i0] '''
    i0, i1, i2 = cuda.grid(3)
    tmp = 0
    K0 = k0[i1] * w0[i0, 0]
    K1 = k0[i1] * w0[i0, 1]
    K2 = k1[i2]
    for ii in range(x.shape[0]):
        rk0 = r[ii, 0] * K0
        rk1 = r[ii, 3] * K0 + r[ii, 1] * K1
        rk2 = r[ii, 5] * K0 + r[ii, 4] * K1 + r[ii, 2] * K2
        norm = rk0 * rk0 + rk1 * rk1 + rk2 * rk2
        if norm < 30:
            IP = x[ii, 0] * K0 + x[ii, 1] * K1 + x[ii, 2] * K2
            tmp += I[ii] * c_exp(-.5 * norm) * c_exp(complex(0, -IP))
    arr[i0, i1, i2] = tmp


@cuda.jit("void(f4[:],f4[:,:],f4[:,:],f4[:],f4[:],f4[:], c8[:,:,:])", cache=True)
def __grid_eval_iso(I, x, r, k0, k1, k2, arr):
    '''
    I*exp(-|rk|^2/2)*exp(-(x\cdot k)i)
    '''
    i0, i1, i2 = cuda.grid(3)
    tmp = 0
    for ii in range(x.shape[0]):
        norm = (k0[i0] * k0[i0] + k1[i1] * k1[i1] +
                k2[i2] * k2[i2]) / (r[ii, 0] * r[ii, 0])
        if norm < 30:
            IP = x[ii, 0] * k0[i0] + x[ii, 1] * k1[i1] + x[ii, 2] * k2[i2]
            tmp += I[ii] * c_exp(-.5 * norm) * c_exp(complex(0, -IP))
    arr[i0, i1, i2] = tmp


@cuda.jit("void(f4[:],f4[:,:],f4[:,:],f4[:],f4[:],f4[:,:],f4[:,:], c8[:,:,:])", cache=True)
def __hyperplane_eval_iso(I, x, r, k0, k1, w0, w1, arr):
    ''' k[i0,i1,i2] = k0[i1]*w0[i0] + k1[i2]*w1[i0] '''
    i0, i1, i2 = cuda.grid(3)
    tmp = 0
    K0 = k0[i1] * w0[i0, 0] + k1[i2] * w1[i0, 0]
    K1 = k0[i1] * w0[i0, 1] + k1[i2] * w1[i0, 1]
    K2 = k0[i1] * w0[i0, 2] + k1[i2] * w1[i0, 2]
    for ii in range(x.shape[0]):
        norm = (K0 * K0 + K1 * K1 + K2 * K2) / (r[ii, 0] * r[ii, 0])
        if norm < 30:
            IP = x[ii, 0] * K0 + x[ii, 1] * K1 + x[ii, 2] * K2
            tmp += I[ii] * c_exp(-.5 * norm) * c_exp(complex(0, -IP))
    arr[i0, i1, i2] = tmp


@cuda.jit("void(f4[:],f4[:,:],f4[:,:],f4[:],f4[:],f4[:,:], c8[:,:,:])", cache=True)
def __hyperplane_eval_iso_2p5D(I, x, r, k0, k1, w0, arr):
    ''' k[i0,i1,i2] = k0[i1]*w0[i0] + k1[i2]*w1[i0] '''
    i0, i1, i2 = cuda.grid(3)
    tmp = 0
    K0 = k0[i1] * w0[i0, 0]
    K1 = k0[i1] * w0[i0, 1]
    K2 = k1[i2]
    for ii in range(x.shape[0]):
        norm = (K0 * K0 + K1 * K1 + K2 * K2) / (r[ii, 0] * r[ii, 0])
        if norm < 30:
            IP = x[ii, 0] * K0 + x[ii, 1] * K1 + x[ii, 2] * K2
            tmp += I[ii] * c_exp(-.5 * norm) * c_exp(complex(0, -IP))
    arr[i0, i1, i2] = tmp


@cuda.jit("void(f4,f4[:],f4[:],f4[:],c8[:],c8[:],c8[:,:])", device=True, inline=True, cache=True)
def __derivs_aniso_aux(I, x, r, K, g, dg, ddg):
    '''
    No mixed terms so we have:

    g = I*exp(-|rk|^2/2)*exp(-(x\cdot k)i)
      = I*G

    dg/dI = G
    dg/dx_j = -ik_jg
    dg/dr_{j0,j1} = -rkg (drk/dr_{j0,j1})
            = -(rk)_{j0}k_{j1}g

    ddg/dII = 0
    ddg/dxx = -kk g
    ddg/dr_{j0,j1}r_{j2,j3} = (rk)_{j0}(rk)_{j2}k_{j1}k_{j3} g - k_{j1}k_{j3}g \delta_{j0==j2}

    ddg/dIx_j = -ik_jG 
    ddg/dIr_j = -(rk)_{j0}k_{j1}G 
    ddg/dxr = -ik (dg/dr) 
    '''

    rk = cuda.local.array((3,), f4)
    ind0 = cuda.local.array((6,), i4)
    ind1 = cuda.local.array((6,), i4)

    ind0[0], ind0[1], ind0[2] = 0, 1, 2
    ind0[3], ind0[4], ind0[5] = 1, 2, 2
    ind1[0], ind1[1], ind1[2] = 0, 1, 2
    ind1[3], ind1[4], ind1[5] = 0, 1, 0

    rk[0] = r[0] * K[0]
    rk[1] = r[3] * K[0] + r[1] * K[1]
    rk[2] = r[5] * K[0] + r[4] * K[1] + r[2] * K[2]

    N = -.5 * (rk[0] * rk[0] + rk[1] * rk[1] + rk[2] * rk[2])
    if N < -30:
        g[0] = 0
        for i in range(10):
            dg[i] = 0
            for j in range(10):
                ddg[i, j] = 0
        return

    G = c_exp(complex(N, -(x[0] * K[0] + x[1] * K[1] + x[2] * K[2])))

    g[0] = I * G

    # d/dI
    dg[0] = G

    # d^2/dI^2
    ddg[0, 0] = 0

    # d^2/dx^2
    for i in range(3):
        for j in range(i, 3):
            ddg[1 + i, 1 + j] = -K[i] * K[j] * g[0]

    # d^2/dr^2
    for i in range(6):
        i0, i1 = ind0[i], ind1[i]
        ddg[4 + i, 4 + i] = g[0] * (rk[i0] * K[i1] * rk[i0] * K[i1]
                                    - K[i1] * K[i1])
        for j in range(i + 1, 6):
            j0, j1 = ind0[j], ind1[j]
            if i0 == j0:
                ddg[4 + i, 4 + j] = g[0] * (rk[i0] * K[i1] * rk[j0] * K[j1]
                                            - K[i1] * K[j1])
            else:
                ddg[4 + i, 4 + j] = g[0] * rk[i0] * K[i1] * rk[j0] * K[j1]

    # d^2/dIdx
    for i in range(3):
        ddg[0, 1 + i] = complex(0, -K[i]) * G

    # d/dx
    for i in range(3):
        dg[1 + i] = I * ddg[0, 1 + i]

    # d^2/dIdr
    for i in range(6):
        i0, i1 = ind0[i], ind1[i]
        ddg[0, 4 + i] = -rk[i0] * K[i1] * G

    # d/dr
    for i in range(6):
        dg[4 + i] = ddg[0, 4 + i] * I

    # d^2/dxdr
    for i in range(3):
        for j in range(6):
            ddg[1 + i, 4 + j] = complex(0, -K[i]) * dg[4 + j]

    # Symetrise
    for i in range(10):
        for j in range(i):
            ddg[i, j] = ddg[j, i]


@cuda.jit("void(f4[:],f4[:,:],f4[:,:],f4[:],f4[:],f4[:,:],f4[:,:], c8[:,:,:], f4[:], f4[:,:], f4[:,:,:], i4[:])", cache=True)
def __derivs_aniso(I, x, r, k0, k1, w0, w1, C, f, Df, DDf, sz):
    buf = cuda.shared.array((THREADS, 66), dtype=f4)
    F = cuda.local.array((1,), f4)
    DF = cuda.local.array((10,), f4)
    DDF = cuda.local.array((10, 10), f4)
    g = cuda.local.array((1,), c8)
    dg = cuda.local.array((10,), c8)
    ddg = cuda.local.array((10, 10), c8)
    K = cuda.local.array((3,), f4)

    block, thread = cuda.blockIdx.x, cuda.threadIdx.x
    cc = cuda.grid(1)

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
            K[0] = k0[i0] * w0[T, 0] + k1[i1] * w1[T, 0]
            K[1] = k0[i0] * w0[T, 1] + k1[i1] * w1[T, 1]
            K[2] = k0[i0] * w0[T, 2] + k1[i1] * w1[T, 2]
            __derivs_aniso_aux(I[0], x[0], r[0], K, g, dg, ddg)

            g[0] = g[0] - C[T, i0, i1]
            F[0] += (g[0] * g[0].conjugate()).real
            for j0 in range(10):
                DF[j0] += (g[0] * dg[j0].conjugate()).real
                for j1 in range(10):
                    DDF[j0, j1] += (dg[j0] * dg[j1].conjugate() +
                                    g[0] * ddg[j0, j1].conjugate()).real
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


@cuda.jit("void(f4[:],f4[:,:],f4[:,:],f4[:],f4[:],f4[:,:],c8[:,:,:],f4[:],f4[:,:],f4[:,:,:],i4[:])", cache=True)
def __derivs_aniso_2p5D(I, x, r, k0, k1, w0, C, f, Df, DDf, sz):
    buf = cuda.shared.array((THREADS, 66), dtype=f4)
    F = cuda.local.array((1,), f4)
    DF = cuda.local.array((10,), f4)
    DDF = cuda.local.array((10, 10), f4)
    g = cuda.local.array((1,), c8)
    dg = cuda.local.array((10,), c8)
    ddg = cuda.local.array((10, 10), c8)
    K = cuda.local.array((3,), f4)

    block, thread = cuda.blockIdx.x, cuda.threadIdx.x
    cc = cuda.grid(1)

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
            K[0] = k0[i0] * w0[T, 0]
            K[1] = k0[i0] * w0[T, 1]
            K[2] = k1[i1]
            __derivs_aniso_aux(I[0], x[0], r[0], K, g, dg, ddg)

            g[0] = g[0] - C[T, i0, i1]
            F[0] += (g[0] * g[0].conjugate()).real
            for j0 in range(10):
                DF[j0] += (g[0] * dg[j0].conjugate()).real
                for j1 in range(10):
                    DDF[j0, j1] += (dg[j0] * dg[j1].conjugate() +
                                    g[0] * ddg[j0, j1].conjugate()).real
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


@cuda.jit("void(f4,f4[:],f4,f4[:],c8[:],c8[:],c8[:,:])", device=True, inline=True, cache=True)
def __derivs_iso_aux(I, x, r, K, g, dg, ddg):
    '''
    No mixed terms so we have:

    g = I*exp(-|k/r|^2/2)*exp(-(x\cdot k)i)
      = I*G

    dg/dI = G
    dg/dx_j = -ik_jg
    dg/dr = +(k/r^2)(k/r)g
            = |k/r|^2g/r

    ddg/dII = 0
    ddg/dxx = -kk g
    ddg/drr = |k|^2g(|k|^2/r^6 - 3/r^4)
            = (|k/r|^2-3)|k/r|^2g/r^2

    ddg/dIx_j = -ik_jG 
    ddg/dIr = |k/r|^2G/r 
    ddg/dxr = -ik (dg/dr) 
    '''

    rk = cuda.local.array((3,), f4)

    rk[0] = K[0] / r
    rk[1] = K[1] / r
    rk[2] = K[2] / r

    N = rk[0] * rk[0] + rk[1] * rk[1] + rk[2] * rk[2]
    if N > 60:
        g[0] = 0
        for i in range(10):
            dg[i] = 0
            for j in range(10):
                ddg[i, j] = 0
        return

    G = c_exp(complex(-.5 * N, -(x[0] * K[0] + x[1] * K[1] + x[2] * K[2])))

    g[0] = I * G

    # d/dI
    dg[0] = G

    # d^2/dI^2
    ddg[0, 0] = 0

    # d^2/dx^2
    for i in range(3):
        for j in range(i, 3):
            ddg[1 + i, 1 + j] = -K[i] * K[j] * g[0]

    # d^2/dr^2
    ddg[4, 4] = (N - 3) * N * g[0] / (r * r)

    # d^2/dIdx
    for i in range(3):
        ddg[0, 1 + i] = complex(0, -K[i]) * G

    # d/dx
    for i in range(3):
        dg[1 + i] = I * ddg[0, 1 + i]

    # d^2/dIdr
    ddg[0, 4] = N * G / r

    # d/dr
    dg[4] = ddg[0, 4] * I

    # d^2/dxdr
    for i in range(3):
        ddg[1 + i, 4] = complex(0, -K[i]) * dg[4]

    # Symetrise
    for i in range(5):
        for j in range(i):
            ddg[i, j] = ddg[j, i]


@cuda.jit("void(f4[:],f4[:,:],f4[:,:],f4[:],f4[:],f4[:,:],f4[:,:],c8[:,:,:],f4[:],f4[:,:],f4[:,:,:],i4[:])", cache=True)
def __derivs_iso(I, x, r, k0, k1, w0, w1, C, f, Df, DDf, sz):
    buf = cuda.shared.array((THREADS, 21), dtype=f4)
    F = cuda.local.array((1,), f4)
    DF = cuda.local.array((5,), f4)
    DDF = cuda.local.array((5, 5), f4)
    g = cuda.local.array((1,), c8)
    dg = cuda.local.array((5,), c8)
    ddg = cuda.local.array((5, 5), c8)
    K = cuda.local.array((3,), f4)

    block, thread = cuda.blockIdx.x, cuda.threadIdx.x
    cc = cuda.grid(1)

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
            K[0] = k0[i0] * w0[T, 0] + k1[i1] * w1[T, 0]
            K[1] = k0[i0] * w0[T, 1] + k1[i1] * w1[T, 1]
            K[2] = k0[i0] * w0[T, 2] + k1[i1] * w1[T, 2]
            __derivs_iso_aux(I[0], x[0], r[0, 0], K, g, dg, ddg)

            g[0] = g[0] - C[T, i0, i1]
            F[0] += (g[0] * g[0].conjugate()).real
            for j0 in range(5):
                DF[j0] += (g[0] * dg[j0].conjugate()).real
                for j1 in range(5):
                    DDF[j0, j1] += (dg[j0] * dg[j1].conjugate() +
                                    g[0] * ddg[j0, j1].conjugate()).real
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


@cuda.jit("void(f4[:],f4[:,:],f4[:,:],f4[:],f4[:],f4[:,:],c8[:,:,:],f4[:],f4[:,:],f4[:,:,:],i4[:])", cache=True)
def __derivs_iso_2p5D(I, x, r, k0, k1, w0, C, f, Df, DDf, sz):
    buf = cuda.shared.array((THREADS, 21), dtype=f4)
    F = cuda.local.array((1,), f4)
    DF = cuda.local.array((10,), f4)
    DDF = cuda.local.array((10, 10), f4)
    g = cuda.local.array((1,), c8)
    dg = cuda.local.array((10,), c8)
    ddg = cuda.local.array((10, 10), c8)
    K = cuda.local.array((3,), f4)

    block, thread = cuda.blockIdx.x, cuda.threadIdx.x
    cc = cuda.grid(1)

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
            K[0] = k0[i0] * w0[T, 0]
            K[1] = k0[i0] * w0[T, 1]
            K[2] = k1[i1]
            __derivs_iso_aux(I[0], x[0], r[0, 0], K, g, dg, ddg)

            g[0] = g[0] - C[T, i0, i1]
            F[0] += (g[0] * g[0].conjugate()).real
            for j0 in range(5):
                DF[j0] += (g[0] * dg[j0].conjugate()).real
                for j1 in range(5):
                    DDF[j0, j1] += (dg[j0] * dg[j1].conjugate() +
                                    g[0] * ddg[j0, j1].conjugate()).real
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


if __name__ == '__main__':
    from numpy import array_str, random, concatenate, log10, maximum
    from sympy import symbols, exp as s_exp, sqrt as s_sqrt, diff

    def niceprint(thing): return array_str(
        array(thing, 'float32'), precision=3)

    s_I, s_x, s_k, s_r = symbols(
        ('I', 'x:3', 'k:3', 'r:6'), real=True)
    s_C = symbols('C', real=False)
    s_rK = (s_r[0] * s_k[0], s_r[3] * s_k[0] + s_r[1] * s_k[1],
            s_r[5] * s_k[0] + s_r[4] * s_k[1] + s_r[2] * s_k[2])
    s_xK = s_x[0] * s_k[0] + s_x[1] * s_k[1] + s_x[2] * s_k[2]
#    g = I*exp(-|rk|^2/2)*exp(-(x\cdot k)i)

    I, x, k, r, C = random.rand(1), random.rand(
        1, 3), random.rand(1, 3), random.rand(1, 6), random.rand(1, 1, 1) + 1j * random.rand(1, 1, 1)
    for thing in 'Ixkr':
        locals()[thing] = locals()[thing].astype('f4')
    C = C.astype('c8')

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
    G = s_exp(- 0.5 * (s_rK[0]**2 + s_rK[1]**2 + s_rK[2]**2) - 1j * s_xK)
    f = 0.5 * (s_I * G - s_C) * (s_I * G - s_C).conjugate()

    var = (s_I,) + s_x + s_r
    df = [diff(f, x) for x in var]
    ddf = [[diff(d, x) for x in var] for d in df]
    var = (s_C, s_I) + s_x + s_k + s_r
    val = concatenate((C[0, 0], I, x[0], k[0], r[0]))
    subs = [(var[i], val[i]) for i in range(len(var))]
    f = f.subs(subs)
    df = [d.subs(subs) for d in df]
    ddf = [[d.subs(subs) for d in dd] for dd in ddf]

    g, dg, ddg = empty(1, dtype='f4'), empty(
        (10, 1), dtype='f4'), empty((10, 10, 1), dtype='f4')
    __derivs_aniso[1, 1](I, x, r, array(1, dtype='f4'), array(0, dtype='f4'),
                         k, k, C, g, dg, ddg,
                         array([1, 1, 1, 1], dtype='i4'))
#     __derivs_aniso_2p5D[1, 1](I, x, r, array(1, dtype='f4'), array(k[0, 2], dtype='f4'),
#                               k[:, :2], C, g, dg, ddg,
#                               array([1, 1, 1, 1], dtype='i4'))
    g, dg, ddg = g.sum(-1), dg.sum(-1), ddg.sum(-1)
    f, df, ddf = f, array(df, 'complex64').real, array(ddf, 'complex64').real

    print('Printouts for aniso test:')
#     print(niceprint(f))
#     print(niceprint(g), '\n')
    print('f error: ', niceprint(abs(f - g)), '\n')
#     print(niceprint(df))
#     print(niceprint(dg), '\n')
    print('Df error: ', niceprint(abs(df - dg).max()), '\n')
#     print(niceprint(ddf), '\n')
#     print(niceprint(ddg), '\n')
    print('DDf error: ', niceprint(abs(ddf - ddg).max()), '\n')
#     for i in range(10):
#         for j in range(i, 10):
#             if abs(ddf[i, j] - ddg[i, j]) > 1e-7:
#                 print('index ', i, j, ' is wrong')

    # # # # # #
    # Test __derivs_iso
    # # # # # #
    G = s_exp(- 0.5 * (s_k[0]**2 + s_k[1]**2 +
                       s_k[2]**2) / s_r[0]**2 - 1j * s_xK)
    f = 0.5 * (s_I * G - s_C) * (s_I * G - s_C).conjugate()

    I, x, k, r, C = random.rand(1), random.rand(
        1, 3), random.rand(1, 3), random.rand(1, 1), random.rand(1, 1, 1) + 1j * random.rand(1, 1, 1)
    for thing in 'Ixkr':
        locals()[thing] = locals()[thing].astype('f4')
    C = C.astype('c8')

    var = (s_I,) + s_x + s_r[:1]
    df = [diff(f, x) for x in var]
    ddf = [[diff(d, x) for x in var] for d in df]
    var = (s_C, s_I) + s_x + s_k + s_r[:1]
    val = concatenate((C[0, 0], I, x[0], k[0], r[0]))
    subs = [(var[i], val[i]) for i in range(len(var))]
    f = f.subs(subs)
    df = [d.subs(subs) for d in df]
    ddf = [[d.subs(subs) for d in dd] for dd in ddf]

    g, dg, ddg = empty(1, dtype='f4'), empty(
        (5, 1), dtype='f4'), empty((5, 5, 1), dtype='f4')
    __derivs_iso[1, 1](I, x, r, array(1, dtype='f4'), array(0, dtype='f4'),
                       k, k, C, g, dg, ddg,
                       array([1, 1, 1, 1], dtype='i4'))
    __derivs_iso_2p5D[1, 1](I, x, r, array(1, dtype='f4'), array(k[0, 2], dtype='f4'),
                            k[:, :2], C, g, dg, ddg,
                            array([1, 1, 1, 1], dtype='i4'))
    g, dg, ddg = g.sum(-1), dg.sum(-1), ddg.sum(-1)
    f, df, ddf = f, array(df, 'complex64').real, array(ddf, 'complex64').real

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
