'''
Created on 6 Jul 2018

@author: Rob Tovey
'''
import numba
from numpy import exp, empty, array
from numba import cuda, float32 as f4, int32 as i4
from math import exp as c_exp
from .numba_cuda_aux import THREADS, __GPU_reduce_n


def evaluate(atoms, S):
    k0, k1 = S
    if k0.dtype.itemsize == 4:
        dtype = 'c8'
    else:
        dtype = 'c16'
    if k1.ndim == 1:
        out = empty((k0.size, k1.size), dtype=dtype)
        if atoms.r.ndim == 1:
            __grid_FTGaussEval_iso_CPU(
                atoms.I, atoms.x, atoms.r, k0, k1, out)
        else:
            __grid_FTGaussEval_aniso_CPU(
                atoms.I, atoms.x, atoms.r, k0, k1, out)
    else:
        out = empty((k1.shape[0], k0.shape[0]), dtype=dtype)
        if atoms.r.ndim == 1:
            __hyperplane_FTGaussEval_iso_CPU(
                atoms.I, atoms.x, atoms.r, k0, k1, out)
        else:
            __hyperplane_FTGaussEval_aniso_CPU(
                atoms.I, atoms.x, atoms.r, k0, k1, out)
    return out


def derivs(atoms, k0, k1):
    raise NotImplemented


@numba.jit(["c8(f4,f4[:],f4[:],f4,f4)", "c16(f8,f8[:],f8[:],f8,f8)"], target='cpu', cache=True, nopython=True)
def __eval_aniso_aux_CPU(I, x, r, K0, K1):
    rK0, rK1 = r[0] * K0, r[2] * K0 + r[1] * K1
    norm = rK0 * rK0 + rK1 * rK1
    if norm < 60:
        return (I * exp(-.5 * norm)) * \
            exp(complex(0, -x[0] * K0 - x[1] * K1))
    else:
        return 0


@numba.jit(["void(f4[:],f4[:,:],f4[:,:],f4[:],f4[:],c8[:,:])", "void(f8[:],f8[:,:],f8[:,:],f8[:],f8[:],c16[:,:])"], target='cpu', cache=True, nopython=True)
def __grid_FTGaussEval_aniso_CPU(I, x, r, k0, k1, arr):
    for i0 in range(k0.shape[0]):
        for i1 in range(k1.shape[0]):
            arr[i0, i1] = 0
            K0, K1 = k0[i0], k1[i1]
            for j in range(x.shape[0]):
                arr[i0, i1] += __eval_aniso_aux_CPU(I[j], x[j], r[j], K0, K1)


@numba.jit(["void(f4[:],f4[:,:],f4[:,:],f4[:],f4[:,:],c8[:,:])", "void(f8[:],f8[:,:],f8[:,:],f8[:],f8[:,:],c16[:,:])"], target='cpu', cache=True, nopython=True)
def __hyperplane_FTGaussEval_aniso_CPU(I, x, r, k, w, arr):
    for i0 in range(w.shape[0]):
        for i1 in range(k.shape[0]):
            arr[i0, i1] = 0
            K0, K1 = k[i1] * w[i0, 0], k[i1] * w[i0, 1]
            for j in range(x.shape[0]):
                arr[i0, i1] += __eval_aniso_aux_CPU(I[j], x[j], r[j], K0, K1)


@numba.jit(["c8(f4,f4[:],f4,f4,f4)", "c16(f8,f8[:],f8,f8,f8)"], target='cpu', cache=True, nopython=True)
def __eval_iso_aux_CPU(I, x, r, K0, K1):
    norm = r * r * (K0 * K0 + K1 * K1)
    if norm < 60:
        return (I * exp(-.5 * norm)) * \
            exp(complex(0, -x[0] * K0 - x[1] * K1))
    else:
        return 0


@numba.jit(["void(f4[:],f4[:,:],f4[:],f4[:],f4[:],c8[:,:])", "void(f8[:],f8[:,:],f8[:],f8[:],f8[:],c16[:,:])"], target='cpu', cache=True, nopython=True)
def __grid_FTGaussEval_iso_CPU(I, x, r, k0, k1, arr):
    for i0 in range(k0.shape[0]):
        for i1 in range(k1.shape[0]):
            arr[i0, i1] = 0
            K0, K1 = k0[i0], k1[i1]
            for j in range(x.shape[0]):
                arr[i0, i1] += __eval_iso_aux_CPU(I[j], x[j], r[j], K0, K1)


@numba.jit(["void(f4[:],f4[:,:],f4[:],f4[:],f4[:,:],c8[:,:])", "void(f8[:],f8[:,:],f8[:],f8[:],f8[:,:],c16[:,:])"], target='cpu', cache=True, nopython=True)
def __hyperplane_FTGaussEval_iso_CPU(I, x, r, k, w, arr):
    for i0 in range(w.shape[0]):
        for i1 in range(k.shape[0]):
            arr[i0, i1] = 0
            K0, K1 = k[i1] * w[i0, 0], k[i1] * w[i0, 1]
            for j in range(x.shape[0]):
                arr[i0, i1] += __eval_iso_aux_CPU(I[j], x[j], r[j], K0, K1)
