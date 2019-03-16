'''
Created on 23 Jan 2018

@author: Rob Tovey
'''
from numba import cuda
import numba
THREADS = 128


@cuda.jit("void(f4[:])", device=True, inline=True)
def __GPU_reduce_1(x):
    cc = cuda.threadIdx.x
    cuda.syncthreads()
    if THREADS >= 512:
        if cc < 256:
            x[cc] += x[cc + 256]
        cuda.syncthreads()
    if THREADS >= 256:
        if cc < 128:
            x[cc] += x[cc + 128]
        cuda.syncthreads()
    if THREADS >= 128:
        if cc < 64:
            x[cc] += x[cc + 64]
        cuda.syncthreads()
    if THREADS >= 64:
        if cc < 32:
            x[cc] += x[cc + 32]
        cuda.syncthreads()
    if cc < 16:
        if THREADS >= 32:
            x[cc] += x[cc + 16]
        x[cc] += x[cc + 8]
        x[cc] += x[cc + 4]
        x[cc] += x[cc + 2]
        x[cc] += x[cc + 1]


@cuda.jit("void(f4[:,:])", device=True, inline=True)
def __GPU_reduce_2(x):
    cc = cuda.threadIdx.x
    cuda.syncthreads()

    if THREADS >= 512:
        if cc < 256:
            x[cc, 0] += x[cc + 256, 0]
            x[cc, 1] += x[cc + 256, 1]
        cuda.syncthreads()
    if THREADS >= 256:
        if cc < 128:
            x[cc, 0] += x[cc + 128, 0]
            x[cc, 1] += x[cc + 128, 1]
        cuda.syncthreads()
    if THREADS >= 128:
        if cc < 64:
            x[cc, 0] += x[cc + 64, 0]
            x[cc, 1] += x[cc + 64, 1]
        cuda.syncthreads()
    if THREADS >= 64:
        if cc < 32:
            x[cc, 0] += x[cc + 32, 0]
            x[cc, 1] += x[cc + 32, 1]
        cuda.syncthreads()
    if THREADS >= 32:
        if cc < 16:
            x[cc, 0] += x[cc + 16, 0]
            x[cc, 1] += x[cc + 16, 1]
        cuda.syncthreads()
    if cc < 8:
        x[cc, 0] += x[cc + 8, 0]
        x[cc, 1] += x[cc + 8, 1]
        x[cc, 0] += x[cc + 4, 0]
        x[cc, 1] += x[cc + 4, 1]
        x[cc, 0] += x[cc + 2, 0]
        x[cc, 1] += x[cc + 2, 1]
        x[cc, 0] += x[cc + 1, 0]
        x[cc, 1] += x[cc + 1, 1]


@cuda.jit("void(f4[:,:])", device=True, inline=True)
def __GPU_reduce_3(x):
    cc = cuda.threadIdx.x
    cuda.syncthreads()
    if THREADS >= 512:
        if cc < 256:
            x[cc, 0] += x[cc + 256, 0]
            x[cc, 1] += x[cc + 256, 1]
            x[cc, 2] += x[cc + 256, 2]
        cuda.syncthreads()
    if THREADS >= 256:
        if cc < 128:
            x[cc, 0] += x[cc + 128, 0]
            x[cc, 1] += x[cc + 128, 1]
            x[cc, 2] += x[cc + 128, 2]
        cuda.syncthreads()
    if THREADS >= 128:
        if cc < 64:
            x[cc, 0] += x[cc + 64, 0]
            x[cc, 1] += x[cc + 64, 1]
            x[cc, 2] += x[cc + 64, 2]
        cuda.syncthreads()
    if THREADS >= 64:
        if cc < 32:
            x[cc, 0] += x[cc + 32, 0]
            x[cc, 1] += x[cc + 32, 1]
            x[cc, 2] += x[cc + 32, 2]
        cuda.syncthreads()
    if THREADS >= 32:
        if cc < 16:
            x[cc, 0] += x[cc + 16, 0]
            x[cc, 1] += x[cc + 16, 1]
            x[cc, 2] += x[cc + 16, 2]
        cuda.syncthreads()
    if cc < 8:
        x[cc, 0] += x[cc + 8, 0]
        x[cc, 1] += x[cc + 8, 1]
        x[cc, 2] += x[cc + 8, 2]
        x[cc, 0] += x[cc + 4, 0]
        x[cc, 1] += x[cc + 4, 1]
        x[cc, 2] += x[cc + 4, 2]
        x[cc, 0] += x[cc + 2, 0]
        x[cc, 1] += x[cc + 2, 1]
        x[cc, 2] += x[cc + 2, 2]
        x[cc, 0] += x[cc + 1, 0]
        x[cc, 1] += x[cc + 1, 1]
        x[cc, 2] += x[cc + 1, 2]


@cuda.jit("void(f4[:,:],i4)", device=True, inline=True)
def __GPU_reduce_n(x, n):
    cc = cuda.threadIdx.x
    cuda.syncthreads()

    if THREADS >= 512:
        if cc < 256:
            for i in range(n):
                x[cc, i] += x[cc + 256, i]
        cuda.syncthreads()
    if THREADS >= 256:
        if cc < 128:
            for i in range(n):
                x[cc, i] += x[cc + 128, i]
        cuda.syncthreads()
    if THREADS >= 128:
        if cc < 64:
            for i in range(n):
                x[cc, i] += x[cc + 64, i]
        cuda.syncthreads()
    if THREADS >= 64:
        if cc < 32:
            for i in range(n):
                x[cc, i] += x[cc + 32, i]
        cuda.syncthreads()
    if THREADS >= 32:
        if cc < 16:
            for i in range(n):
                x[cc, i] += x[cc + 16, i]
        cuda.syncthreads()
    if cc < 8:
        for i in range(n):
            x[cc, i] += x[cc + 8, i]
    cuda.syncthreads()
    if cc < 4:
        for i in range(n):
            x[cc, i] += x[cc + 4, i]
    cuda.syncthreads()
    if cc < 2:
        for i in range(n):
            x[cc, i] += x[cc + 2, i]
    cuda.syncthreads()
    if cc < 1:
        for i in range(n):
            x[cc, i] += x[cc + 1, i]


@cuda.jit('void(f4[:],f4[:],i4[:])')
def __GPU_fill_F(x, y, sz):
    cc = cuda.grid(1)
    start = sz[0] * cc
    end = min(start + sz[0], sz[1])
    if y.size == 1:
        for indx in range(start, end):
            x[indx] = y[0]
    else:
        for indx in range(start, end):
            x[indx] = y[indx]


@cuda.jit('void(c8[:],c8[:],i4[:])')
def __GPU_fill_C(x, y, sz):
    cc = cuda.grid(1)
    start = sz[0] * cc
    end = min(start + sz[0], sz[1])
    if y.size == 1:
        for indx in range(start, end):
            x[indx] = y[0]
    else:
        for indx in range(start, end):
            x[indx] = y[indx]


@cuda.jit('void(f4[:,:],f4[:,:],i4[:])')
def __GPU_fill2(x, y, sz):
    cc = cuda.grid(1)
    for indx in range(sz[2] * cc, sz[2] * (cc + 1)):
        jj = indx // sz[1]
        kk = indx - jj * sz[1]
        if jj < sz[0]:
            if y.size == 1:
                x[jj, kk] = y[0, 0]
            else:
                x[jj, kk] = y[jj, kk]


@cuda.jit('void(c8[:,:],c8[:,:],i4[:])')
def __GPU_fill2_C(x, y, sz):
    cc = cuda.grid(1)
    for indx in range(sz[2] * cc, sz[2] * (cc + 1)):
        jj = indx // sz[1]
        kk = indx - jj * sz[1]
        if jj < sz[0]:
            if y.size == 1:
                x[jj, kk] = y[0, 0]
            else:
                x[jj, kk] = y[jj, kk]


@cuda.jit('void(f4[:,:],f4[:,:],f4[:,:],i4[:])')
def __GPU_mult2_R(x, y, out, sz):
    cc = cuda.grid(1)
    for indx in range(sz[2] * cc, sz[2] * (cc + 1)):
        jj = indx // sz[1]
        kk = indx - jj * sz[1]
        if jj < sz[0]:
            out[jj, kk] = x[jj, kk] * y[jj, kk]


@cuda.jit('void(c8[:,:],c8[:,:],c8[:,:],i4[:])')
def __GPU_mult2_C(x, y, out, sz):
    cc = cuda.grid(1)
    for indx in range(sz[2] * cc, sz[2] * (cc + 1)):
        jj = indx // sz[1]
        kk = indx - jj * sz[1]
        if jj < sz[0]:
            out[jj, kk] = x[jj, kk] * y[jj, kk]


@cuda.jit('void(c8[:,:],c8[:,:],c8[:,:],i4[:])')
def __GPU_mult2_Cconj(x, y, out, sz):
    cc = cuda.grid(1)
    for indx in range(sz[2] * cc, sz[2] * (cc + 1)):
        jj = indx // sz[1]
        kk = indx - jj * sz[1]
        if jj < sz[0]:
            tmp = complex(y[jj, kk].real, -y[jj, kk].imag)
            out[jj, kk] = x[jj, kk] * tmp


@cuda.jit('void(c8[:,:],f4[:],i4[:])')
def __GPU_sum2_C2R(x, out, sz):
    buf = cuda.shared.array(THREADS, dtype=numba.float32)
    cc = cuda.grid(1)

    sum0 = 0
    for indx in range(sz[2] * cc, sz[2] * (cc + 1)):
        jj = indx // sz[1]
        kk = indx - jj * sz[1]
        if jj < sz[0]:
            sum0 += x[jj, kk].real
    buf[cc] = sum0
    cuda.syncthreads()

    __GPU_reduce_1(buf)
    if cc == 0:
        out[0] = buf[0]


@cuda.jit("void(f4[:],f4[:],i4)")
def __GPU_reduce_flex_F(x, out, sz):
    tid = cuda.threadIdx.x
    i = cuda.blockIdx.x * (2 * THREADS) + tid
    step = (THREADS * 2) * cuda.gridDim.x
    end = sz - THREADS

    buf = cuda.shared.array((THREADS,), dtype=numba.float32)

    buf[tid] = 0
    while i < end:
        buf[tid] += x[i] + x[i + THREADS]
        i += step
    if i < sz:
        buf[tid] += x[i]
    cuda.syncthreads()

    __GPU_reduce_1(buf)
    if tid == 0:
        out[cuda.blockIdx.x] = buf[0]


@cuda.jit("void(c8[:],f4[:,:],i4)")
def __GPU_reduce_flex_C(x, out, sz):
    tid = cuda.threadIdx.x
    i = cuda.blockIdx.x * (2 * THREADS) + tid
    step = (THREADS * 2) * cuda.gridDim.x
    end = sz - THREADS

    buf = cuda.shared.array((THREADS, 2), dtype=numba.float32)

    buf[tid, 0] = 0
    buf[tid, 1] = 0
    while i < end:
        buf[tid, 0] += x[i].real + x[i + THREADS].real
        buf[tid, 1] += x[i].imag + x[i + THREADS].imag
        i += step
    if i < sz:
        buf[tid, 0] += x[i].real
        buf[tid, 1] += x[i].imag
    cuda.syncthreads()

    __GPU_reduce_2(buf)
    if tid == 0:
        out[0, cuda.blockIdx.x] = buf[0, 0]
        out[1, cuda.blockIdx.x] = buf[0, 1]
