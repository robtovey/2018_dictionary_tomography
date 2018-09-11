'''
Created on 4 Jan 2018

@author: Rob Tovey
'''
import numpy as np
from numpy import cos, sqrt, empty, sin
import numba
from code.bin.dictionary_def import Dictionary, Element
from code.bin.manager import context


def theta_to_vec(theta):
    n = [len(t) for t in theta]
    dim = len(n) + 1
    C = [cos(t) for t in theta]
    S = [-sin(t) for t in theta]
    w = np.empty(n + [dim])

    if dim == 2:
        for i in range(n[0]):
            w[i, 0] = S[0][i]
            c = C[0][i]
            w[i, 1] = c
    elif dim == 3:
        for i in range(n[0]):
            for j in range(n[1]):
                w[i, j, 0] = S[0][i] * C[1][j]
                w[i, j, 1] = C[0][i] * C[1][j]
                w[i, j, 2] = -S[1][j]
#                 w[i, j, 0] = S[0][i]
#                 w[i, j, 1] = C[0][i] * S[1][j]
#                 w[i, j, 2] = C[0][i] * C[1][j]

    elif dim == 4:
        for i in range(n[0]):
            for j in range(n[1]):
                for k in range(n[2]):
                    w[i, j, k, 0] = S[0][i]
                    c = C[0][i]
                    w[i, j, k, 1] = c * S[1][j]
                    c *= C[1][j]
                    w[i, j, k, 2] = c * S[2][k]
                    c *= C[2][k]
                    w[i, j, k, 3] = c
    # Look at np.nditer(w,flags=['multi_index','writeonly'])
    return w


def perp(theta):
    n = [len(t) for t in theta]
    dim = len(n) + 1
    C = [cos(t) for t in theta]
    S = [-sin(t) for t in theta]

    if dim == 2:
        ortho = np.empty(n + [dim])
        for i in range(n[0]):
            ortho[i, 0] = C[0][i]
            ortho[i, 1] = -S[0][i]
        return (ortho,)
    elif dim == 3:
        ortho0 = np.empty(n + [dim])
        ortho1 = np.empty(n + [dim])
        for i in range(n[0]):
            for j in range(n[1]):
                ortho0[i, j, 0] = C[0][i]
                ortho0[i, j, 1] = -S[0][i]
                ortho0[i, j, 2] = 0

                ortho1[i, j, 0] = S[0][i] * S[1][j]
                ortho1[i, j, 1] = C[0][i] * S[1][j]
                ortho1[i, j, 2] = C[1][j]
        return ortho0, ortho1


class GaussTomo(Dictionary):
    '''
An atom is going to be defined by a position x = [x1,x2,...] and radius,
r such that density is:
    u(y) = exp(-|y-x|^2/(2r^2))
Radon transform becomes:
    Ru(w,p) = \int_{y = t*w + p} u(y)
            = r*sqrt(2\pi) exp(-|P_{w^\perp}(p-x)|^2/(2r^2))
    D_xRu(w,p) = \frac{P_{w^\perp}(p-x)}{r^2}Ru(w,p)
    D_rRu(w,p) = (1/r + |P_{w^\perp}(p-x)|^2/r^3)Ru(w,p)
We make a slight assumption that $p$ is already given in standard ortho
basis so it is 1 codim and can be factored out of the projection. 
    '''

    def __init__(self, ASpace, VSpace, device='CPU'):
        Dictionary.__init__(self, ASpace, VSpace, isLinear=True)
        self.__device = device.lower()
        if ASpace.isotropic:
            if self.ElementSpace.dim == 2:
                import code.bin.atomic_Radon_2D_numba as myRad
            elif self.ElementSpace.dim == 3:
                if len(self.ProjectionSpace.ortho) == 2:
                    import code.bin.atomic_Radon_3D_numba as myRad
                else:
                    import code.bin.atomic_Radon_2p5D_numba as myRad
        else:
            if self.ElementSpace.dim == 2:
                import code.bin.atomic_Radon_2D_numba_anisotropic as myRad
            elif self.ElementSpace.dim == 3:
                if len(self.ProjectionSpace.ortho) == 2:
                    import code.bin.atomic_Radon_3D_numba_anisotropic as myRad
                else:
                    import code.bin.atomic_Radon_2p5D_numba_anisotropic as myRad

        c = context()
        if self.__device == 'cpu':
            self.__fwrd = myRad.GaussRadon_CPU
            self.__diff = [myRad.GaussRadon_DI_CPU,
                           myRad.GaussRadon_Dx_CPU,
                           myRad.GaussRadon_Dr_CPU]
        else:
            self.__fwrd = myRad.GaussRadon_GPU
            self.__diff = [myRad.GaussRadon_DI_GPU,
                           myRad.GaussRadon_Dx_GPU,
                           myRad.GaussRadon_Dr_GPU]
        self.__params = ((c.copy(VSpace.orientations),)
                         + tuple(c.copy(x) for x in VSpace.ortho)
                         + tuple(c.copy(x) for x in VSpace.detector))

    def __call__(self, atoms, Slice=slice(None)):
        # Slice is a 1D slice object on orientations
        R = self.ProjectionSpace.null(Slice)

        if len(self.__params) == 3:
            tmp = tuple(s[Slice]
                        for s in self.__params[:-1]) + (self.__params[-1],)
        else:
            tmp = tuple(s[Slice]
                        for s in self.__params[:-2]) + self.__params[-2:]
        self.__fwrd(atoms.I, atoms.x, atoms.r, *tmp, R.array)

        return R

    def grad(self, atoms, axis, d, Slice=slice(None)):
        if isinstance(d, Element):
            d = d.array
        if axis == 'I':
            axis = 0
        elif axis == 'x':
            axis = 1
        elif axis == 'r':
            axis = 2

        # Pre-allocate
        if axis == 0:
            g = empty(atoms.I.shape, dtype=atoms.I.dtype)
        elif axis == 1:
            g = empty(atoms.x.shape, dtype=atoms.x.dtype)
        else:
            g = empty(atoms.r.shape, dtype=atoms.r.dtype)

        self.__diff[axis](atoms.I, atoms.x, atoms.r,
                          *self.__params,
                          d, g)
        if len(self.__params) == 3:
            tmp = tuple(s[Slice]
                        for s in self.__params[:-1]) + (self.__params[-1],)
        else:
            tmp = tuple(s[Slice]
                        for s in self.__params[:-2]) + self.__params[-2:]
            tmp = self.__params
        self.__diff[axis](atoms.I, atoms.x, atoms.r, *tmp, d, g)

        return g


class GaussVolume(Dictionary):
    def __init__(self, ASpace, VSpace, device='cpu'):
        Dictionary.__init__(self, ASpace, VSpace, isLinear=True)
        self.__device = device.lower()
        if ASpace.isotropic:
            if self.ElementSpace.dim == 2:
                import code.bin.atomic_Radon_2D_numba as myProj
            elif self.ElementSpace.dim == 3:
                import code.bin.atomic_Radon_3D_numba as myProj
        else:
            if self.ElementSpace.dim == 2:
                import code.bin.atomic_Radon_2D_numba_anisotropic as myProj
            elif self.ElementSpace.dim == 3:
                import code.bin.atomic_Radon_3D_numba_anisotropic as myProj

        if self.__device == 'cpu':
            self.__proj = myProj.GaussVol_CPU
        else:
            self.__proj = myProj.GaussVol_GPU

    def __call__(self, atoms):
        u = self.ProjectionSpace.zero()
        self.__proj(atoms.I, atoms.x, atoms.r,
                    *self.ProjectionSpace.grid,
                    u.asarray())
        return u


@numba.jit("void(f8[:],f8[:,:],f8[:],f8[:,:],f8[:],f8[:,:])", target='cpu', nopython=True)
def BallRadon_2D_CPU(I, x, r, w, p, R):
    '''
    u(y) = I*(|y-x|<r)
    Ru(p,theta) = 2*I*sqrt(r^2-|P_{theta^\perp}(x-p)|^2)
    d/dxRu = -2*I*P(x-p)/sqrt(r^2-|P_{theta^\perp}(x-p)|^2)
    d/drRu = 2*I*r/sqrt(r^2-|P_{theta^\perp}(x-p)|^2)
    '''
    r2 = r**2
    for jj in range(w.shape[0]):
        for kk in range(p.shape[0]):
            tmp = 0
            for ii in range(x.shape[0]):
                inter = (p[kk] - x[ii, 0] * w[jj, 0] - x[ii, 1] * w[jj, 1])**2
                if inter < r2[ii]:
                    tmp += 2 * I[ii] * sqrt(r2[ii] - inter)
            R[jj, kk] = tmp


@numba.jit("void(f8[:],f8[:,:],f8[:],f8[:,:],f8[:],f8[:,:])", target='cpu', nopython=True)
def BallRadon_Dx_2D_CPU(I, x, r, w, p, R):
    '''
    u(y) = I*(|y-x|<r)
    Ru(p,theta) = 2*I*sqrt(r^2-|P_{theta^\perp}(x-p)|^2)
    d/dxRu = -2*I*P(x-p)/sqrt(r^2-|P_{theta^\perp}(x-p)|^2)
    P(x-p) = <w,x-p>w
    '''
    r2 = r**2
    for jj in range(w.shape[0]):
        for kk in range(p.shape[0]):
            tmp = 0
            for ii in range(x.shape[0]):
                IP = x[ii, 0] * w[jj, 0] + x[ii, 1] * w[jj, 1]
                inter = (p[kk] - IP)**2
                if inter < r2[ii]:
                    inter = -2 * I[ii] * IP / sqrt(r2[ii] - inter)
                    tmp += -2 * I[ii] * sqrt(r2[ii] - inter)
            R[jj, kk] = tmp


@numba.jit("void(f8[:],f8[:,:],f8[:],f8[:],f8[:],f8[:,:])", target='cpu', nopython=True)
def BallVol_2D_CPU(I, x, r, y0, y1, u):
    s = r**2
    for jj in range(y0.shape[0]):
        for kk in range(y1.shape[0]):
            tmp = 0
            for ii in range(x.shape[0]):
                interior = ((y0[jj] - x[ii, 0])**2 +
                            (y1[kk] - x[ii, 1])**2)
                if interior < s[ii]:
                    tmp += I[ii]
            u[jj, kk] = tmp


class BallRadon(Dictionary):
    def __init__(self, ASpace, VSpace):
        Dictionary.__init__(self, ASpace, VSpace, isLinear=True)

    def __call__(self, atoms):
        R = self.ProjectionSpace.zero()
        BallRadon_2D_CPU(atoms.I, atoms.x, atoms.r,
                         self.ProjectionSpace.orientations,
                         self.ProjectionSpace.detector[0],
                         R.array)
        return R


class BallVolume(Dictionary):
    def __init__(self, ASpace, VSpace):
        Dictionary.__init__(self, ASpace, VSpace, isLinear=True)

    def __call__(self, atoms):
        u = self.ProjectionSpace.zero()
        BallVol_2D_CPU(atoms.I, atoms.x, atoms.r,
                       self.ProjectionSpace.grid[0],
                       self.ProjectionSpace.grid[1],
                       u.array)
        return u


def test_grad(space, Radon, eps, axis=None):
    if axis is None:
        Axis = range(3)
    else:
        Axis = [axis]
    from numpy import log10
    from time import time
    tic = time()
    a = space.random(10, seed=1)
    R = Radon(a)
    c = context()

    def norm2(x):
        return c.sum(c.mul(x.array, x.array)) / 2
    n = norm2(R)

    for axis in Axis:
        grad = Radon.grad(a, axis, R)
#         print(np.array_str(grad, precision=2))

        if axis == 0:
            da = c.rand(a.I.shape)
        elif axis == 1:
            da = c.rand(a.x.shape)
        else:
            da = c.rand(a.r.shape)

        for e in eps:
            aP = a.copy()
            if axis == 0:
                c.set(aP.I, c.add(aP.I, c.mul(e, da)))
            elif axis == 1:
                c.set(aP.x, c.add(aP.x, c.mul(e, da)))
            elif axis == 2:
                c.set(aP.r, c.add(aP.r, c.mul(e, da)))

            Rp = Radon(aP)
            nP = norm2(Rp)
            print('axis = %d: % 3.2f, % 3.2f' % (axis, log10(
                abs(nP - n - e * c.sum(c.mul(grad, da)))), log10(e)))
#             print(nP, n, abs(grad).max(), abs(Rp.asarray()).max())
#             print((nP - n) / e, c.asarray(c.sum(c.mul(grad, da))))
#             print(abs(nP - n - e * c.sum(c.mul(e, da))), log10(e))
        print()
    print('Finished after time: ', time() - tic)


if __name__ == '__main__':
    from numpy import pi
    import odl
    from code.bin.dictionary_def import VolSpace, ProjSpace, AtomSpace, ProjElement
    from matplotlib import pyplot as plt

#     # 2D comparison
    ASpace = AtomSpace(2, isotropic=False)
    atoms = ASpace.random(2, seed=1)
    angles = odl.uniform_partition(0, 2 * pi, 360, nodes_on_bdry=True)
    detector = odl.uniform_partition(-1, 1, 256)
    vol = odl.uniform_partition([-1] * 2, [1] * 2, [128] * 2)
    view = GaussVolume(ASpace, VolSpace(vol), 'GPU')
    myPSpace = ProjSpace(angles, detector)
    myTomo = GaussTomo(ASpace, myPSpace, device='GPU')
    mySino = myTomo(atoms)
    odlPSpace = odl.tomo.Parallel2dGeometry(angles, detector)
    odlTomo = odl.tomo.RayTransform(
        odl.uniform_discr_frompartition(vol), odlPSpace)
    odlSino = odlTomo(view(atoms).asarray())
    odlSino = ProjElement(myPSpace, odlSino.__array__())

    mySino.plot(plt.subplot('211'), aspect='auto')
    plt.title('2D Atomic Sinogram (top) and volume-sinogram (bottom)')
    odlSino.plot(plt.subplot('212'), aspect='auto')

#     # 2.5D comparison
    ASpace = AtomSpace(3, isotropic=False)
    atoms = ASpace.random(3, seed=2)
    atoms.x[:, :] = np.array([0, 0, .8])
    atoms.I[:] = 1
    atoms.r[:, :3] = 7
    atoms.r[:, 3:] = 0
    angles = odl.uniform_partition(0, pi, 4, nodes_on_bdry=True)
    detector = odl.uniform_partition([-1] * 2, [1] * 2, [128] * 2)
    vol = odl.uniform_partition([-1] * 3, [1] * 3, [256] * 3)
    view = GaussVolume(ASpace, VolSpace(vol), 'GPU')
    myPSpace = ProjSpace(angles, detector)
    myTomo = GaussTomo(ASpace, myPSpace, device='GPU')
    mySino = myTomo(atoms)
    odlPSpace = odl.tomo.Parallel3dAxisGeometry(angles, detector)
    odlTomo = odl.tomo.RayTransform(
        odl.uniform_discr_frompartition(vol), odlPSpace)
    odlSino = odlTomo(view(atoms).asarray())
    odlSino = ProjElement(myPSpace, odlSino.__array__())

    plt.figure()
    mySino.plot(plt.subplot('211'), aspect='auto')
    plt.title('2.5D Atomic Sinogram (top) and volume-sinogram (bottom)')
    odlSino.plot(plt.subplot('212'), aspect='auto')

    # 3D comparison
    ASpace = AtomSpace(3, isotropic=False)
    atoms = ASpace.random(3, seed=2)
#     atoms.x[:, :] = np.array([.8, 0, 0])
#     atoms.I[:] = 1
#     atoms.r[:, :3] = 7
#     atoms.r[:, 3:] = 0
    angles = odl.uniform_partition(
        [0, 0], [pi, pi], [3, 3], nodes_on_bdry=False)
    detector = odl.uniform_partition([-1] * 2, [1] * 2, [128] * 2)
    vol = odl.uniform_partition([-1] * 3, [1] * 3, [256] * 3)
    view = GaussVolume(ASpace, VolSpace(vol), 'GPU')
    myPSpace = ProjSpace(angles, detector)
    myTomo = GaussTomo(ASpace, myPSpace, device='GPU')
    mySino = myTomo(atoms)
    odlPSpace = odl.tomo.Parallel3dEulerGeometry(angles, detector)
    odlTomo = odl.tomo.RayTransform(
        odl.uniform_discr_frompartition(vol), odlPSpace)
    odlSino = odlTomo(view(atoms).asarray())
    odlSino = ProjElement(myPSpace, odlSino.__array__().reshape(-1, 128, 128))

    plt.figure()
    mySino.plot(plt.subplot('211'), aspect='auto')
    plt.title('3D Atomic Sinogram (top) and volume-sinogram (bottom)')
    odlSino.plot(plt.subplot('212'), aspect='auto')

    plt.show(block=True)
