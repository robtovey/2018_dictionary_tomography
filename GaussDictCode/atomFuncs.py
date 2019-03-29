'''
Created on 4 Jan 2018

@author: Rob Tovey
'''
import numpy as np
from numpy import cos, sqrt, empty, sin, vstack
import numba
from GaussDictCode.dictionary_def import DictionaryOp, directionalGradient
from GaussDictCode.bin.manager import context


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


class GaussTomo(DictionaryOp):
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

    def __init__(self, Atoms, Volume, Sino, device='GPU'):
        DictionaryOp.__init__(self, Atoms, Volume, Sino, linear=False)
        self.__device = device.lower()
        
        if self.__device == 'cpu':
            raise ValueError('CPU processing not yet supported')

        self.__fwrd, self.__derivs, self.__L2derivs, self.__proj = self._getOps(Atoms.dim)

        c = context()
#         self.__params = (c.copy(Sino.orientations),
#                          tuple(c.copy(x) for x in Sino.ortho),
#                          tuple(c.copy(x) for x in Sino.detector))
        self.__vol = tuple(c.cast(x) for x in self.embedding.coord_vectors)
        
    def _getOps(self, dim):
        if dim == 2:
            import GaussDictCode.bin.NewtonBases2D as myRad
        elif dim == 3:
            import GaussDictCode.bin.NewtonBases as myRad
        else:
            raise ValueError
        return myRad.RadProj, myRad.derivs_RadProj, myRad.L2derivs_RadProj, myRad.VolProj

    def __call__(self, atoms, out=None):
        if out is None:
            out = self.range.element(context().empty(self.range.shape))
        self.__fwrd(atoms, self, out.data)
        return out
    
    def discretise(self, atoms, out=None):
        if out is None:
            out = self.embedding.element(context().empty(self.embedding.shape))
        self.__proj(atoms, self.__vol, out.data)
        return out

    def _stackAtoms(self, atoms):
        return self.domain.element(vstack(atoms))
        
    def derivs(self, atoms, C, order=2):
        f, df, ddf = [], [], []
        for i in range(atoms.size):
            tmp = self.__derivs(atoms[i], self, C.data, order)
            f.append(tmp[0])
            df.append(tmp[1][None, :])
            ddf.append(tmp[2][None, :, :])
        f, df, ddf = [vstack(thing) for thing in (f, df, ddf)]

        if order == 1:
            return f, df
        else:
            return f, df, ddf
        
    def gradient(self, x): return directionalGradient(x, self.range, lambda x, d: self.derivs(x, d, order=1)[1])

    def L2_derivs(self, atoms, C, order=2):
        f, df, ddf = [], [], []
        for i in range(atoms.size):
            tmp = self.__L2derivs(atoms[i], self, C.data, order)
            f.append(tmp[0])
            df.append(tmp[1][None, :])
            ddf.append(tmp[2][None, :, :])
        f, df, ddf = [vstack(thing) for thing in (f, df, ddf)]

        if order == 1:
            return f, df
        else:
            return f, df, ddf


class SingleParticleGaussTomo(GaussTomo):
    '''
Equivalent to the GaussTomo operator but uses a point 
spread function and multi-core distrubution is aimed
at many low resolution images.
    '''

    def _getOps(self, dim):
        if dim not in (2, 3):
            raise ValueError
        import GaussDictCode.bin.point_spread_functions as myRad
        return myRad.RadProj, myRad.derivs_RadProj, myRad.L2derivs_RadProj, myRad.VolProj


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
        return c.sum(c.mul(x.data, x.data)) / 2

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
    from GaussDictCode.dictionary_def import VolSpace, ProjSpace, AtomSpace, ProjElement
    from matplotlib import pyplot as plt

#     # 2D comparison
    ASpace = AtomSpace(2, isotropic=False)
    atoms = ASpace.random(2, seed=1)
    angles = odl.uniform_partition(0, 2 * pi, 360, nodes_on_bdry=True)
    detector = odl.uniform_partition(-1, 1, 256)
    vol = odl.uniform_partition([-1] * 2, [1] * 2, [128] * 2)
    myPSpace = ProjSpace(angles, detector)
    myTomo = GaussTomo(ASpace, VolSpace(vol), myPSpace, device='GPU')
    mySino = myTomo(atoms)
    odlPSpace = odl.tomo.Parallel2dGeometry(angles, detector)
    odlTomo = odl.tomo.RayTransform(
        odl.uniform_discr_frompartition(vol), odlPSpace)
    odlSino = odlTomo(myTomo.discretise(atoms).asarray())
    odlSino = ProjElement(myPSpace, odlSino.asarray())

    mySino.plot(plt.subplot('211'), aspect='auto')
    plt.title('2D Atomic Sinogram (top) and volume-sinogram (bottom)')
    odlSino.plot(plt.subplot('212'), aspect='auto')

#     # 2.5D comparison
    ASpace = AtomSpace(3, isotropic=False)
    atoms = ASpace.random(3, seed=2)
    atoms.I[:] = 1
    atoms.r[:, :3], atoms.r[:, 3:] = 7, 0
    angles = odl.uniform_partition(0, pi, 4, nodes_on_bdry=True)
    detector = odl.uniform_partition([-1] * 2, [1] * 2, [128] * 2)
    vol = odl.uniform_partition([-1] * 3, [1] * 3, [256] * 3)
    myPSpace = ProjSpace(angles, detector)
    myTomo = GaussTomo(ASpace, VolSpace(vol), myPSpace, device='GPU')
    mySino = myTomo(atoms)
    odlPSpace = odl.tomo.Parallel3dAxisGeometry(angles, detector)
    odlTomo = odl.tomo.RayTransform(
        odl.uniform_discr_frompartition(vol), odlPSpace)
    odlSino = odlTomo(myTomo.discretise(atoms).asarray())
    odlSino = ProjElement(myPSpace, odlSino.asarray())

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
    myPSpace = ProjSpace(angles, detector)
    myTomo = GaussTomo(ASpace, VolSpace(vol), myPSpace, device='GPU')
    mySino = myTomo(atoms)
    odlPSpace = odl.tomo.Parallel3dEulerGeometry(angles, detector)
    odlTomo = odl.tomo.RayTransform(
        odl.uniform_discr_frompartition(vol), odlPSpace)
    odlSino = odlTomo(myTomo.discretise(atoms).asarray())
    odlSino = ProjElement(myPSpace, odlSino.__array__().reshape(-1, 128, 128))

    plt.figure()
    mySino.plot(plt.subplot('211'), aspect='auto')
    plt.title('3D Atomic Sinogram (top) and volume-sinogram (bottom)')
    odlSino.plot(plt.subplot('212'), aspect='auto')

    plt.show(block=True)
