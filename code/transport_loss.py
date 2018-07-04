'''
Created on 9 Jan 2018

@author: Rob Tovey
'''

import numba
from numba import cuda
# from pyculib import fft as cu_fft
from numpy import empty, zeros, array, exp, log, maximum, arange
from numpy.fft import fftshift, irfft2, rfft2
from code.bin.dictionary_def import Element
from code.bin.numba_cuda_aux import __GPU_reduce_2, __GPU_fill2,\
    __GPU_mult2_Cconj, __GPU_mult2_C, THREADS, __GPU_fill2_C, __GPU_sum2_C2R
from code.bin.manager import context
################
# Earth movers
################


@numba.jit(["void(T[:,:],T[:,:],T[:])".replace('T', T) for T in ['f4', 'f8']], target='cpu', cache=True, nopython=True)
def earth_mover_1D(x, y, d):
    n = x.shape[1]
    for i0 in range(x.shape[0]):
        # Sum[i] = (x-y)[i:].Sum() = -(x-y)[:i].Sum()
        # f[0] = 0
        # f[i] = sign(Sum[i])
        # d = (f*Sum).Sum()
        d[i0] = 0
        Sum = 0
        for i1 in range(1, n):
            Sum -= x[i0, i1 - 1] - y[i0, i1 - 1]
            d[i0] += abs(Sum)


def earth_mover_1D_GPU(x, y, d):
    blocks = x.shape[0]
    __earth_mover_1D_GPU[max(blocks // THREADS, 1), THREADS](x, y, d)


@cuda.jit('void(f4[:,:],f4[:,:],f4[:])')
def __earth_mover_1D_GPU(x, y, d):
    i = cuda.grid(1)
    d[i] = 0
    Sum = 0
    for j in range(1, x.shape[1]):
        Sum -= x[i, j - 1] - y[i, j - 1]
        d[i] += abs(Sum)


@numba.jit(["void(T[:,:],T[:,:],T[:,:])".replace('T', T) for T in ['f4', 'f8']], target='cpu', cache=True, nopython=True)
def earth_mover_grad_1D(x, y, d):
    n = x.shape[1]
    for i0 in range(x.shape[0]):
        # Sum[i] = (x-y)[i:].Sum() = -(x-y)[:i].Sum()
        # f[0] = 0
        # f[i] = sign(Sum[i])
        # d[i] = f[:i].Sum()
        # d -= d.mean()
        d[i0, 0] = 0
        Sum = 0
        tmp = 0
        for i1 in range(1, n):
            Sum -= x[i0, i1 - 1] - y[i0, i1 - 1]
            if Sum < 0:
                d[i0, i1] = d[i0, i1 - 1] - 1
            else:
                d[i0, i1] = d[i0, i1 - 1] + 1
            tmp += d[i0, i1]
        d[i0, :n] -= tmp / n


def earth_mover_grad_1D_GPU(x, y, d):
    blocks = x.shape[0]
    __earth_mover_grad_1D_GPU[max(blocks // THREADS, 1), THREADS](x, y, d)


@cuda.jit('void(f4[:,:],f4[:,:],f4[:,:])')
def __earth_mover_grad_1D_GPU(x, y, d):
    i = cuda.grid(1)
    d[i, 0] = 0
    Sum = 0
    mean = 0
    for j in range(1, x.shape[1]):
        Sum -= x[i, j - 1] - y[i, j - 1]
        if Sum < 0:
            d[i, j] = d[i, j - 1] - 1
        else:
            d[i, j] = d[i, j - 1] + 1
        mean += d[i, j]

    mean = mean / x.shape[1]
    for j in range(x.shape[1]):
        d[i, j] -= mean


@numba.jit(["void(T[:,:],T[:,:],T[:,:,:],T[:],T[:,:],T[:,:,:],T[:,:,:],T[:,:,:],T[:,:],T[:,:])".replace('T', T) for T in ['f4', 'f8']],
           target='cpu', cache=True, nopython=False)
def earth_mover_2D_pdhg(x, d, mu, tau, dP, muP, buf1, buf2, buf3, buf4):
    '''
primal = mu, dual = d
max_d min_{mu} <d,x> + <d,grad^T(mu)> + |mu|_1
F*(z) = \infty*max(0,|z|-1)
F(mu) = |mu|_1
G^* = eps*|d|^2-<d,x>
G = |z+x|^2/(2*eps)


prox_{tF}(z) = max(|z|-t,0)sign(z)
prox_{sG^*} = (z+s*x)/(1+s*eps)
gap = |mu|_1+(eps/2)|d|^2-<d,x>+|z+x|^2/(2*eps)
feasibility = max(|grad(d)|-1,0)

muP = prox_{tF}(mu-t*grad(d))
d = prox_{sG^*}(d+s*grad^T(2*muP-mu))
  = d+s*(grad^T(2*muP-mu)+x)

inf |mu|_1 + |grad^T(mu)+x|^2/(2eps)
infsup |mu|_1 + <grad^T(mu)+x,d> -(eps/2)|d|^2
sup <d,x>- (eps/2)|d|^2 s.t. |grad(d)|<1
    '''

    eps = 0
    Del, alpha, eta = 2, 0, 0.9
    # Memory allocation
    grad_d = buf1
    grad_dP = buf2
    grad_muP = buf3
    grad_mu = buf4
    # Initial gradient computation
    grad_2d(d, grad_d)
    gradT_2d(mu, grad_mu)
    for _ in range(100):
        # Proximal step in mu
        norm_mu = 0
        for j0 in range(mu.shape[0]):
            for j1 in range(mu.shape[1]):
                for j2 in range(2):
                    muP[j0, j1, j2] = mu[j0, j1, j2] - \
                        tau[0] * grad_d[j0, j1, j2]
                    tmp = abs(muP[j0, j1, j2]) - tau[0]
                    if tmp > 0:
                        if muP[j0, j1, j2] > 0:
                            muP[j0, j1, j2] = tmp
                        else:
                            muP[j0, j1, j2] = -tmp
                        norm_mu += tmp
                    else:
                        muP[j0, j1, j2] = 0
        # Calculate new grad^Tmu:
        gradT_2d(muP, grad_muP)
        # Proximal step in d
        IP_d_x = 0
        if eps == 0:
            for j0 in range(d.shape[0]):
                for j1 in range(d.shape[1]):
                    dP[j0, j1] = d[j0, j1] + tau[1] * \
                        (2 * grad_muP[j0, j1] - grad_mu[j0, j1] + x[j0, j1])
                    IP_d_x += dP[j0, j1] * x[j0, j1]
        else:
            tmp = 1 / (1 + eps * tau[1])
            for j0 in range(d.shape[0]):
                for j1 in range(d.shape[1]):
                    dP[j0, j1] = (d[j0, j1] + tau[1] *
                                  (2 * grad_muP[j0, j1] - grad_mu[j0, j1] + x[j0, j1])) * tmp
                    IP_d_x += dP[j0, j1] * x[j0, j1]

        # Calculate new grad(d)
        grad_2d(dP, grad_dP)

        # Possible early stopping
        if abs(norm_mu - IP_d_x) < 1e-4 * (norm_mu + IP_d_x) and _ > 10:
            break

        # Compute step size
        if 10 * (_ // 10) == _:
            prim, dual = 0, 0
            for j0 in range(d.shape[0]):
                for j1 in range(d.shape[1]):
                    prim += ((muP[j0, j1, 0] - mu[j0, j1, 0] - tau[0]
                              * (grad_dP[j0, j1, 0] - grad_d[j0, j1, 0]))**2
                             + (muP[j0, j1, 1] - mu[j0, j1, 1] - tau[0]
                                * (grad_dP[j0, j1, 1] - grad_d[j0, j1, 1]))**2)
                    dual += (dP[j0, j1] - d[j0, j1] - tau[1]
                             * (grad_muP[j0, j1] - grad_mu[j0, j1]))**2
            prim, dual = prim**.5, dual**.5
            prim /= tau[0]
            dual /= tau[1]
            if prim > Del * dual:
                tau[0] /= 1 - alpha
                tau[1] *= 1 - alpha
                alpha *= eta
            elif Del * prim < dual:
                tau[0] *= 1 - alpha
                tau[1] /= 1 - alpha
                alpha *= eta

#         prim = norm_mu + ((grad_muP + x)**2).sum() / (2 * eps)
#         dual = (d**2).sum() * eps / 2 - IP_d_x
#         print('%d, %E, %E, %E, \n' % (_, abs(prim + dual) / (abs(prim) + abs(dual)),
# abs(norm_mu - IP_d_x) / (norm_mu + IP_d_x), max(__max_abs_3d(grad_dP) -
# 1, 0)))
#         norm_mu = abs(mu).sum()
#         IP_d_x = (d * x).sum()
#         print('%E, %E , %f\n' % (abs(norm_mu - IP_d_x) / (norm_mu + IP_d_x + 1e-4),
#                              max(abs(grad_d).max() - 1, 0), tau[1] / tau[0]))

        tmp1 = grad_mu
        grad_mu = grad_muP
        grad_muP = tmp1
        tmp1 = d
        d = dP
        dP = tmp1

        tmp2 = grad_d
        grad_d = grad_dP
        grad_dP = tmp2
        tmp2 = mu
        mu = muP
        muP = tmp2


@numba.jit(["T(T[:,:],T[:,:],T,T[:,:],T[:,:],T[:])".replace('T', T) for T in ['f4', 'f8']], target='cpu', cache=True, nopython=False)
def sinkhorn_2D(p, q, g, u, v, K):
    '''
W_0(p,q) = earth movers distance = min <M,X>
    such that X1=p, X^T1=q, M_{i,j} = |x_i-x_j|
W_\gamma = min <M,X> - \gamma Entropy(X)
    such that X1=p, X^T1=q
According to Cuturi and Peyre 2016 we have:
Proposition 2.2:
    W_\gamma = max_{u,v} <u,p>+<v,q>-\gamma B(u,v)
        where B(u,v) = sum(exp((u_i+v_j-M_{i,j})/\gamma - 1))
Proposition 2.3:
    \partial_p W_\gamma(p,q) = u s.t. sum(u) = 0
Sinkhorn Iteration: 
    K = exp(-M/\gamma)
    Iterate: 
        v = q./K^Tu
        u = p./Kv
    W_\gamma = <(K.*M)v,u>

Note M_{i,j} = |i_1-j_1|+|i_2-j_2|
    K_{i,j} = exp(-|i_1-j_1|/gamma)exp(-|i_2-j_2|/\gamma)
Note if A_{i,j} = |i-j| for [i,j] in p.shape then we have
    K = [A,A+1,...,A+shape[0]; ...; A+shape[1],...,A]
    i.e. K_{i,j} = A+|i-j| is recursive block Toeplitz
    '''

    for _ in range(K.shape[0]):
        K[_] = exp(-_ / g)

    for _ in range(100):
        for j0 in range(v.shape[0]):
            for j1 in range(v.shape[1]):
                tmp = 0
                for i0 in range(u.shape[0]):
                    for i1 in range(u.shape[1]):
                        tmp += K[abs(j0 - i0)] * K[abs(j1 - i1)] * u[i0, i1]
                v[j0, j1] = q[j0, j1] / tmp

        diff = 0
        mag = 0
        for i0 in range(u.shape[0]):
            for i1 in range(u.shape[1]):
                tmp = 0
                for j0 in range(v.shape[0]):
                    for j1 in range(v.shape[1]):
                        tmp += K[abs(j0 - i0)] * K[abs(j1 - i1)] * v[j0, j1]
                tmp = p[i0, i1] / tmp
                diff += (u[i0, i1] - tmp)**2
                mag += tmp * tmp
                u[i0, i1] = tmp

        if (diff / mag)**.5 < 1e-6:
            print(_, diff, u.sum())
            break

    tmp = 0
    for i0 in range(u.shape[0]):
        for i1 in range(u.shape[1]):
            for j0 in range(v.shape[0]):
                for j1 in range(v.shape[1]):
                    tmp += (u[i0, i1] * v[j0, j1] *
                            (abs(j0 - i0) + abs(j1 - i1)) *
                            K[abs(j0 - i0)] * K[abs(j1 - i1)]
                            )

    return tmp


def sinkhorn_2D_fft(p, q, g, u, d):
    '''
A_{i,j} = K[i-j]
(AU)_i = K[i-j]U_j = (K\star U)[i]
is a 2D convolution between K and U for j\in [-(n-1),n-1]
K.shape = [2n,2n], U.shape = [n,n]
K = [0,1,...,n,n-1,...,1]+[0,1,...,n,n-1,...,1]^T
    '''
    n = [p.shape[-2], p.shape[-1]]
    x = [abs(fftshift(arange(-N, N))).reshape(-1, 1) for N in n]

    M = x[0] + x[1].T
    K = rfft2(exp(-M / g))
    diff, mag = 0, 0

    uu = zeros((2 * n[0], 2 * n[1]), dtype='float64')
    v = zeros(uu.shape, dtype='float64')
    U = empty(n, dtype='float64')
    Ku = empty(n, dtype='float64')

    for i in range(p.shape[0]):
        uu[:n[0], :n[1]] = u[i, :, :]
        for _ in range(100):
            Ku = irfft2(K.conj() * rfft2(uu)).real[:n[0], :n[1]]
            __div_reg(q[i], Ku, v, 1e-5)
    #         v[:n, :n] = q / maximum(Ku, 1e-5)

            Ku = irfft2(K * rfft2(v)).real[:n[0], :n[1]]
            __div_reg(p[i], Ku, U, 1e-5)
    #         U = p / maximum(Ku, 1e-5)

            diff = abs(U - uu[:n[0], :n[1]]).sum()
            mag = abs(U).sum()
            uu[:n[0], :n[1]] = U

            if diff / (1e-5 + mag) < 1e-4:
                break

        u[i, :, :] = U
        d[i] = (irfft2(rfft2(M * exp(-M / g)) * rfft2(v)) * uu).sum()


def sinkhorn_2D_fft_GPU(p, q, g, u, d):
    stream = cuda.stream()
    n = [p.shape[-2], p.shape[-1]]
    x = [abs(fftshift(arange(-N, N))).reshape(-1, 1) for N in n]

    grid = 1
    sz = [n[0], n[1], (n[0] * n[1]) // THREADS]
    if sz[2] * THREADS < sz[0] * sz[1]:
        sz[2] += 1
    sz = cuda.to_device(array(sz, dtype='int32'), stream=stream)
    SZ = [2 * n[0], 2 * n[1], (4 * n[0] * n[1]) // THREADS]
    if SZ[2] * THREADS < SZ[0] * SZ[1]:
        SZ[2] += 1
    SZ = cuda.to_device(array(SZ, dtype='int32'), stream=stream)

    M = x[0] + x[1].T
    K = cuda.to_device(
        (exp(-M / g) / M.size).astype('complex64'), stream=stream)
    KK = cuda.to_device(
        (M * exp(-M / g) / M.size).astype('complex64'), stream=stream)

    diff = cuda.device_array(2, dtype='float32', stream=stream)
    tol = cuda.to_device(zeros(1, dtype='float32') + 1e-5, stream=stream)
    Zero = cuda.to_device(zeros((1, 1), dtype='complex64'), stream=stream)
    uu = cuda.device_array((2 * n[0], 2 * n[1]),
                           dtype='complex64', stream=stream)
    __GPU_fill2_C[grid, THREADS](uu, Zero, SZ)
    v = cuda.device_array((2 * n[0], 2 * n[1]),
                          dtype='complex64', stream=stream)
    __GPU_fill2_C[grid, THREADS](v, Zero, SZ)
    Ku = cuda.device_array((2 * n[0], 2 * n[1]),
                           dtype='complex64', stream=stream)
    stream.synchronize()
#     cu_fft.FFTPlan(shape=Ku.shape, itype=1)
    cu_fft.fft(K, K)
    cu_fft.fft(KK, KK)

    for i in range(p.shape[0]):
        U = cuda.to_device(u[i, :, :].astype('complex64'), stream=stream)
        stream.synchronize()
        __GPU_fill2_C[grid, THREADS](uu, U, sz)
        for _ in range(200):
            cu_fft.fft(uu, Ku)
            __GPU_mult2_Cconj[grid, THREADS](Ku, K, Ku, SZ)
            # Ku *= K.conj()
            cu_fft.ifft(Ku, Ku)
            __sinkhorn_2D_GPU_aux0[grid, THREADS](
                q[i], Ku, v, tol, diff, sz)
            # v = q[i]/max(Ku.real,tol[0])
            # diff = (|v-v_old|_1,|v|_1)

            cu_fft.fft(v, Ku)
            __GPU_mult2_C[grid, THREADS](Ku, K, Ku, SZ)
            # Ku *= K
            cu_fft.ifft_inplace(Ku)
            __sinkhorn_2D_GPU_aux0[grid, THREADS](
                p[i], Ku, uu, tol, diff, sz)
            # uu = p[i]/max(Ku,tol[0])
            # diff = (|uu-uu_old|_1,|uu|_1)

            if diff[0] < 1e-4 * diff[1]:
                break

        cu_fft.fft(uu, Ku)
        __GPU_mult2_C[grid, THREADS](Ku, KK, Ku, sz)
        cu_fft.ifft_inplace(Ku)
        __GPU_sum2_C2R(Ku, d[i:i + 1], sz)

        U = uu[:n[0], :n[1]].copy_to_host(stream=stream)
        stream.synchronize()
        u[i, :, :] = U.real


@cuda.jit('void(f4[:,:],c8[:,:],f4[:,:],f4[:],f4[:],i4[:])')
def __sinkhorn_2D_GPU_aux0(x, y, out, tol, diff, sz):
    buf = cuda.shared.array((THREADS, 2), dtype=numba.float32)

    cc = cuda.grid(1)
    i = cuda.threadIdx.x

    sum0 = 0
    sum1 = 0
    for indx in range(sz[2] * cc, sz[2] * (cc + 1)):
        jj = indx // sz[1]
        kk = indx - jj * sz[1]
        if jj < sz[0]:
            tmp1 = out[jj, kk]
            if y[jj, kk].real < tol[0]:
                out[jj, kk] = x[jj, kk] / tol[0]
            else:
                out[jj, kk] = x[jj, kk] / y[jj, kk].real
            sum0 += abs(out[jj, kk] - tmp1)
            sum1 += abs(out[jj, kk])
    buf[i, 0] = sum0
    buf[i, 1] = sum1
    cuda.syncthreads()

    __GPU_reduce_2(buf)
    if cc == 0:
        diff[0] = buf[0, 0]
        diff[1] = buf[0, 1]


def full_Wasserstein_min_3d():
    '''
Solve following with primal-dual extragadient method (Valkonen)
min_{a,u} max_d |u|_1 + <d,R(a)-div(u)> + <d,x>

Step 0: step sizes
K(a,u) = [R(a);grad^T(u)]
K'(a,u) = [R'(a);grad^T]
|sqrt(t)K'sqrt(s)| = |sqrt(t_1)R'sqrt(s)+sqrt(t_2)grad^Tsqrt(s)| <= 1
<= t_1s|R'|^2 + t_2s*8 <= 1 -> t_2=s=1/4, t_1 adaptive...

Step 1: prox a
aP = argmin |z-Z|^2/2 s.t. z\in {__} @ Z = a-t*R'(a)^Td
   = projection(Z)
Step 2: prox u
uP = argmin |z-Z|^2/2 + t*|z|_1 @ Z = u - t*grad(d)
   = max(0,|Z|-t)sign(Z)
Step 3: prox d
dP = argmin |z-Z|^2/2 - s<z,x> & Z =  d + s*(R(2*aP-a)+grad^T(2*uP-u))
   = Z+s*x

    '''
    pass


################
# Utils
################

@numba.jit(["void(T[:,:],T[:,:,:])".replace('T', T) for T in ['f4', 'f8']], target='cpu', cache=True, nopython=True)
def grad_2d(x, Dx):
    for i in range(x.shape[0] - 1):
        for j in range(x.shape[1] - 1):
            Dx[i, j, 0] = x[i + 1, j] - x[i, j]
            Dx[i, j, 1] = x[i, j + 1] - x[i, j]
        Dx[i, j + 1, 0] = x[i + 1, j + 1] - x[i, j + 1]
        Dx[i, j + 1, 1] = 0
    for j in range(x.shape[1] - 1):
        Dx[i + 1, j, 0] = 0
        Dx[i + 1, j, 1] = x[i + 1, j + 1] - x[i + 1, j]
    Dx[i + 1, j + 1, 0] = 0
    Dx[i + 1, j + 1, 1] = 0


@numba.jit(["void(T[:,:,:],T[:,:])".replace('T', T) for T in ['f4', 'f8']], target='cpu', cache=True, nopython=True)
def gradT_2d(x, Dx):
    Dx[0, 0] = (0 - x[0, 0, 0]
                + 0 - x[0, 0, 1])
    for j in range(1, x.shape[1] - 1):
        Dx[0, j] = (0 - x[0, j, 0]
                    + x[0, j - 1, 1] - x[0, j, 1])
    Dx[0, j + 1] = (0 - x[0, j + 1, 0]
                    + x[0, j, 1] - 0)
    for i in range(1, x.shape[0] - 1):
        Dx[i, 0] = (x[i - 1, 0, 0] - x[i, 0, 0]
                    + 0 - x[i, 0, 1])
        for j in range(1, x.shape[1] - 1):
            Dx[i, j] = (x[i - 1, j, 0] - x[i, j, 0]
                        + x[i, j - 1, 1] - x[i, j, 1])
        Dx[i, j + 1] = (x[i - 1, j + 1, 0] - x[i, j + 1, 0]
                        + x[i, j, 1] - 0)
    Dx[i + 1, 0] = (x[i, 0, 0] - 0
                    + 0 - x[i + 1, 0, 1])
    for j in range(1, x.shape[1] - 1):
        Dx[i + 1, j] = (x[i, j, 0] - 0
                        + x[i + 1, j - 1, 1] - x[i + 1, j, 1])
    Dx[i + 1, j + 1] = (x[i, j + 1, 0] - 0
                        + x[i + 1, j, 1] - 0)

# sum (x_{i+1}-x_i)y_i = sum x_i(y_{i-1}-y_i}
# (x_1-x_0)y_0 -> x_0(0-y_0)
# (x_{end}-x_{end-1})y_{end-1} -> x_{end}(y_{end}-0)


@numba.jit(target='cpu', cache=True, nopython=True)
def __div_reg(x, y, z, eps):
    r_eps = 1 / eps
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if y[i, j] > eps:
                z[i, j] = x[i, j] / y[i, j]
            else:
                z[i, j] = x[i, j] * r_eps


def normest(op, opT, shape):
    from numpy import random, sqrt
    x = random.rand(*shape)

    for _ in range(1000):
        x = opT(op(x))
        x = x / abs(x).sum()

    return sqrt((op(x)**2).sum() / (x**2).sum())


################
# Fidelity objects
################


class Fidelity:
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, x, y):
        return NotImplemented

    def grad(self, x, y, axis=0):
        return NotImplemented


class Transport_loss(Fidelity):
    def __init__(self, dim, device='cpu'):
        self.dim = dim

        if dim == 2:
            if device is 'cpu':
                self.__earth_mover = earth_mover_1D
                self.__earth_mover_grad = earth_mover_grad_1D
            else:
                self.__earth_mover = earth_mover_1D_GPU
                self.__earth_mover_grad = earth_mover_grad_1D_GPU
        else:
            def tmp(x, y, d, func):
                if hasattr(self, 'store'):
                    if (self.store['u'].shape != x.shape):
                        self.store['u'] = zeros(x.shape, dtype=x.dtype) + 1
                else:
                    self.store = {
                        'u': zeros(x.shape, dtype=x.dtype) + 1
                    }
                self.store['x'] = x
                self.store['y'] = y
                self.store['g'] = array(1e0, dtype=x.dtype)

                d.shape = -1
                sz = (-1,) + x.shape[-2:]
                self.store['u'].shape = sz

                func(x.reshape(sz) + 0 * 1e-5, y.reshape(sz) + 0 * 1e-5,
                     self.store['g'], self.store['u'], d)

                self.store['u'].shape = x.shape

            def tmp2(x, y, d):
                if hasattr(self, 'store'):
                    if x is self.store['x'] and y is self.store['y']:
                        pass
                    else:
                        self.__call__(x, y)
                else:
                    self.__call__(x, y)
                u = self.store['u']
                sz = u.shape[:-2] + (1, 1)
                u = maximum(u, 1e-5)
                dd = log(u)
                dd = -self.store['g'] * \
                    (dd.mean(axis=(-2, -1)).reshape(sz) - dd)
                d[...] = dd

            if device is 'cpu':
                self.__earth_mover = lambda x, y, d: tmp(
                    x, y, d, sinkhorn_2D_fft)
            else:
                self.__earth_mover = lambda x, y, d: tmp(
                    x, y, d, sinkhorn_2D_fft)
            self.__earth_mover_grad = tmp2

    def __call__(self, x, y):
        if isinstance(x, Element):
            x = x.array
        if isinstance(y, Element):
            y = y.array
        d = empty(x.shape[:self.dim - 1], dtype=x.dtype)

        c = context()
        x, y = c.asarray(x), c.asarray(y)

        from time import time
        tic = time()
        self.__earth_mover(x, y, d)
        print('Timing:', time() - tic)

        return d.sum()

    def grad(self, x, y, axis=0):
        if isinstance(x, Element):
            x = x.array
        if isinstance(y, Element):
            y = y.array
        d = empty(x.shape, dtype=x.dtype)

        if axis == 1:
            return -self.grad(x, y, axis=0)

        c = context()
        self.__earth_mover_grad(c.asarray(x), c.asarray(y), d)

        return d


class l2_squared_loss(Fidelity):
    def __call__(self, x, y):
        if isinstance(x, Element):
            x = x.array
        if isinstance(y, Element):
            y = y.array

        # d = (x - y)**2
        c = context()
        d = c.sub(x, y)
        c.mul(d, d, d)

        return c.sum(d) / 2

    def grad(self, x, y, axis=0):
        if isinstance(x, Element):
            x = x.array
        if isinstance(y, Element):
            y = y.array

        # d = x - y
        c = context()
        if axis == 0:
            d = c.sub(x, y)
        else:
            d = c.sub(y, x)

        return d


################
# Test functions
################

def test_grad(E, shape, eps, axis):
    from numpy import log10, random
    random.seed(1)
    a = random.rand(*shape)
    b = random.rand(*shape)
    a /= a.sum()
    b /= b.sum()

    grad = E.grad(a, b, axis)
    Eab = E(a, b)

    dx = random.rand(*shape)
    dx -= dx.mean(axis=1).reshape(-1, 1)

    for e in eps:
        if axis == 0:
            aP = a + e * dx
            bP = b
        else:
            aP = a
            bP = b + e * dx

        EabP = E(aP, bP)
        print(log10(abs(EabP - Eab - e * (grad * dx).sum())), log10(e))
#         print(EabP - Eab, e * (grad * dx).sum())


def __test_grad_mat(shape):
    from numpy import random
    dim = len(shape)

# # # Estimates norm
    def A(x):
        Dx = empty(x.shape + (2,))
        grad_2d(x, Dx)
        return Dx

    def AT(x):
        Dx = empty(x.shape[:-1])
        gradT_2d(x, Dx)
        return Dx
    print(normest(AT, A, shape + (2,)))
    return

# # # Checks transpose is exact
    for _ in range(10):
        x = random.rand(*shape)
        Dx = random.rand(*shape, dim)
        y = random.rand(*shape)
        Dy = random.rand(*shape, dim)

        grad_2d(x, Dx)
        gradT_2d(Dy, y)
        print((Dx * Dy).sum(), (x * y).sum(),
              ((Dx * Dy).sum(-1) - x * y).sum())
