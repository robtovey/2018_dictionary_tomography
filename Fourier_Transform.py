'''
Created on 17 May 2018

@author: Rob Tovey
'''
from code.bin.dictionary_def import Dictionary, Element
from numpy.fft import fftn, ifftn, fftfreq, fftshift, ifftshift
from numpy import exp, pi, sqrt
import numba
from numba import cuda
from math import exp as c_exp
from code.bin.numba_cuda_aux import __GPU_reduce_1, __GPU_reduce_3, THREADS


class Fourier_Transform(Dictionary):
    '''
FT = Fourier_transform(from,to) 
is a dictionary object which returns the (truncated) Fourier transform
Cases:
from = to = Gaussian:
    This is a continuous FT returning analytical formulations
from = Gaussian, to = R^n:
    This computes a discretisation of the continuous FT
from = to = R^n:
    This computes a discretisation of the continuous FT assuming Dirichlet boundary

Note that we require R^n to be sampled on a uniform grid.
If both input and output are the same then the grids must be equal. 
    '''

    def __init__(self, fromSpace, toSpace):
        Dictionary.__init__(self, fromSpace, toSpace,
                            isLinear=True, isInvertible=True, inverse=self.__inverse)
        if hasattr(fromSpace, 'grid'):
            self.inGrid = fromSpace.grid
            dim = len(self.inGrid)
            for i in range(dim):
                sz = [1] * dim
                sz[i] = -1
                self.inGrid[i].shape = sz
        else:
            self.inGrid = None
        if hasattr(toSpace, 'grid'):
            self.outGrid = toSpace.grid
            dim = len(self.outGrid)
            for i in range(dim):
                sz = [1] * dim
                sz[i] = -1
                self.outGrid[i].shape = sz
        elif self.inGrid is not None:
            self.outGrid = self.getGrid(self.inGrid)
        else:
            self.outGrid = None

    def __call__(self, var):
        if self.outGrid is None:
            FT = var.copy()
            analyticGaussFourier(var.x, var.r, var.I, FT.x, FT.r, FT.I)
        elif self.inGrid is None:
            FT = var.copy()
            analyticGaussFourier(var.x, var.r, var.I, FT.x, FT.r, FT.I)
        else:
            FT = self.__DFT(var, False)
        return FT

    def __inverse(self, var):
        if self.outGrid is None:
            FT = var.copy()
            analyticGaussiFourier(var.x, var.r, var.I, FT.x, FT.r, FT.I)
        else:
            FT = self.__DFT(var, True)
        return FT

    def __DFT(self, arr, inverse=False):
        '''
        FT(k) = \frac1\sqrt{2\pi}\int arr(x)e^{-ik\cdot x}dx
              = \frac{|dx|}{\sqrt{2\pi}}\sum_x arr[x] e^{-i k\cdot grid[x]}
        fft(arr)[k] = \sum_x arr[x] e^{-2\pi i k\cdot ((x-min(x))/(max(x)-min(x)))}

        Assume periodicity so fft(arr)[-k] = fft(arr)[N-k]
        '''
        if inverse:
            dx = [1 / (g.item(1) - g.item(0)) for g in self.inGrid]
            dx[0] *= 2 * pi
            x0 = [-g.item(0) for g in self.inGrid]
            FT = applyweight(arr, self.outGrid, dx, x0)
            FT = ifftn(ifftshift(FT))
        else:
            dx = [g.item(1) - g.item(0) for g in self.inGrid]
            x0 = [g.item(0) for g in self.inGrid]
            FT = fftshift(fftn(arr))
            FT = applyweight(FT, self.outGrid, dx, x0)

        return FT

    @staticmethod
    def getGrid(grid, real2fourier=True):
        dx = [g.item(1) - g.item(0) for g in grid]
        dim = len(dx)
        if real2fourier:
            k = [fftshift(fftfreq(grid[i].size)) / dx[i]
                 for i in range(dim)]
            for i in range(dim):
                sz = [1] * dim
                sz[i] = -1
                k[i].shape = sz
        return k


def applyweight(f, grid, dx, x0):
    '''
    Multiply f by prod(dx)*e^{-2\pi i dk*k\cdot x_0}
    grid is the Fourier space grid
    '''
    dim = len(x0)
    w = 1 / sqrt(2 * pi)
    print(dx)
    for i in range(dim):
        f = f * exp(complex(0, -2 * pi * x0[i]) * grid[i])
        w *= dx[i]
    f *= w
    return f


def analyticGaussFourier(x, r, I, newx, newr, newI):
    '''
    If f(y) = I e^{-|r(y-x)|^2/2} then 
    F[f](k) = (I\sqrt{2\pi}/|r|) e^{-i(2\pi x)\cdot k}e^{-|2\pi r^{-T}k|^2/2}
        new I = (I\sqrt{2\pi}/|r|)
        new r = 2\pi r^{-1}
        new x = 2\pi x
    F[f](k) = newI e^{-i (k\cdot newx)} e^{-|newr^T k|^2/2}
    '''
    nAtoms = x.shape[0]
    dim = x.shape[1]
    tau = 2 * pi
    if r.shape[1] == 1:
        for i in range(nAtoms):
            newI[i] = I[i] * sqrt(tau) / (r[i, 0]**dim)
            newr[i, 0] = tau / r[i, 0]
            for j in range(dim):
                newx[i, j] = tau * x[i, j]
    else:
        for i in range(nAtoms):
            det = r[i, 0]
            for j in range(1, dim):
                det *= r[i, j]
            newI[i] = I[i] * sqrt(tau) / det
            for j in range(dim):
                newx[i, j] = tau * x[i, j]
            if dim == 2:
                newr[i, 0] = tau / r[i, 0]
                newr[i, 1] = tau / r[i, 1]
                newr[i, 2] = -(tau * r[i, 2]) / (r[i, 0] * r[i, 1])
            else:
                newr[i, 0] = tau / r[i, 0]
                newr[i, 1] = tau / r[i, 1]
                newr[i, 2] = tau / r[i, 2]
                newr[i, 3] = -(tau * r[i, 3]) / (r[i, 0] * r[i, 1])
                newr[i, 4] = -(tau * r[i, 4]) / (r[i, 1] * r[i, 2])
                newr[i, 5] = tau * (r[i, 3] * r[i, 4] /
                                    r[i, 1] - r[i, 5]) / (r[i, 0] * r[i, 2])


def analyticGaussiFourier(x, r, I, newx, newr, newI):
    '''
    If f(y) = newI e^{-|newr(y-newx)|^2/2} then 
    F[f](k) = (newI\sqrt{2\pi}/|newr|) e^{(-2\pi x)\cdot k i}e^{-|2\pi newr^{-T}k|^2/2}
        I = (newI\sqrt{2\pi}/|newr|)
        r = 2\pi newr^{-1}
        x = 2\pi newx
    so:
        newr = 2\pi r^{-1}
        newx = x/(2\pi)
        newI = I*|new r|/sqrt(2\pi)
    '''
    nAtoms = x.shape[0]
    dim = x.shape[1]
    tau = 2 * pi
    if r.shape[1] == 1:
        for i in range(nAtoms):
            newr[i, 0] = tau / r[i, 0]
            for j in range(dim):
                newx[i, j] = x[i, j] / tau
            newI[i] = I[i] * (newr[i, 0]**dim) / sqrt(tau)
    else:
        for i in range(nAtoms):
            if dim == 2:
                newr[i, 0] = tau / r[i, 0]
                newr[i, 1] = tau / r[i, 1]
                newr[i, 2] = -(tau * r[i, 2]) / (r[i, 0] * r[i, 1])
            else:
                newr[i, 0] = tau / r[i, 0]
                newr[i, 1] = tau / r[i, 1]
                newr[i, 2] = tau / r[i, 2]
                newr[i, 3] = -(tau * r[i, 3]) / (r[i, 0] * r[i, 1])
                newr[i, 4] = -(tau * r[i, 4]) / (r[i, 1] * r[i, 2])
                newr[i, 5] = tau * (r[i, 3] * r[i, 4] /
                                    r[i, 1] - r[i, 5]) / (r[i, 0] * r[i, 2])
            for j in range(dim):
                newx[i, j] = x[i, j] / tau
            det = newr[i, 0]
            for j in range(1, dim):
                det *= newr[i, j]
            newI[i] = I[i] * det / sqrt(tau)


@numba.jit(["void(fT[:,:],fT[:,:],fT[:],fT[:,:],fT[:,:],cT[:,:])".replace('T', T) for T in ['f4', 'f8']], target='cpu', cache=True, nopython=True)
def pointGaussEval_2D(x, r, I, k0, k1, arr):
    for i0 in range(k0.shape[0]):
        for i1 in range(k1.shape[0]):
            arr[i0, i1] = 0
            kk = [k0[i0, 0], k1[0, i1]]
            for j in range(x.shape[0]):
                tmp = [r[j, 0] * kk[0],
                       r[j, 2] * kk[0] + r[j, 1] * kk[1]]
                tmp = tmp[0] * tmp[0] + tmp[1] * tmp[1]
                if tmp < 40:
                    arr[i0, i1] += (I[j] * exp(-tmp / 2)) *\
                        exp(complex(0, -x[j, 0] * kk[0] - x[j, 1] * kk[1]))


if __name__ == '__main__':
    import numpy as np
    from code.bin.dictionary_def import VolSpace, AtomSpace
    from matplotlib import pyplot as plt
    sz = [256, 1024]
    x = [10 * np.linspace(-1, 1, sz[0]).reshape(-1, 1),
         10 * np.linspace(-1, 1, sz[1]).reshape(1, -1)]
    k = Fourier_Transform.getGrid(x)
    vSpace = VolSpace(x)
    fSpace = VolSpace(k)
    aSpace = AtomSpace(2, True)

    # Volume to Volume:
#     FT = Fourier_Transform(vSpace, fSpace)
#     a, b = 4, [1, -1]
#     A, B = 2 * pi / a, [complex(0, 2 * pi) * bb for bb in b]
#     tmp1 = exp(-a**2 * ((x[0] - b[0])**2 + (x[1] - b[1])**2) / 2)
#
#     tmp2 = FT(tmp1)
#     tmp3 = sqrt(2 * np.pi) / (a**2) * exp(-(k[0] * B[0] + k[1] * B[1])
#                                           - (A**2 / 2) * (k[0]**2 + k[1]**2))
# #     tmp3 = FT.inverse(tmp2)
# #     tmp2 = tmp1
#
#     print(abs(tmp1).max(), abs(tmp2).max(),
#           abs(tmp3).max(), abs(tmp2 - tmp3).max())
#
#     plt.figure()
#     plt.subplot('131')
#     plt.imshow(tmp1)
#     plt.subplot('132')
#     plt.imshow(abs(tmp2))
#     plt.subplot('133')
# #     plt.imshow(abs(tmp3))
# #     plt.imshow(abs(tmp2) - abs(tmp3))
#     plt.imshow(abs(tmp2 - tmp3))
#     plt.colorbar()

    # Atom to Atom:
    FT = Fourier_Transform(aSpace, aSpace)
    nAtoms = 4
    tmp = aSpace.random(nAtoms)
    tmp1 = FT.inverse(FT(tmp))

    print('I check:\n', abs(tmp.I - tmp1.I), '\n')
    print('x check:\n', abs(tmp.x - tmp1.x), '\n')
    print('r check:\n', abs(tmp.r - tmp1.r), '\n')

    plt.show(block=True)
