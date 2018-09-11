'''
Created on 17 May 2018

@author: Rob Tovey
'''
from code.bin.dictionary_def import Dictionary, ProjSpace, AtomElement, VolSpace, AtomSpace,\
    ProjElement, VolElement
from numpy.fft import fftn, ifftn, fftfreq, fftshift, ifftshift
from numpy import exp, pi, sqrt, arange, array, where, random
from code.bin.manager import context


class GaussFT(Dictionary):
    '''
FT = GaussFT(space) 
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

    def __init__(self, space):
        '''
        space must be either an AtomSpace, VolSpace or ProjSpace instance.
        AtomSpace: Fourier transform is purely analytic and maps gaussians to
            gaussians
        VolSpace: Forward map evaluates the FT on this regular grid given by to.grid
            fromSpace is ignored. Either atoms or VolSpace elements can be mapped with 
            the resulting operator and the back projection will map to VolSpace
        ProjSpace: Forward map evaluates the FT using the Fourier slice theorem. 
            Inputs must either be atoms or ProjSpace elements.

        '''

        if isinstance(space, AtomSpace):
            self.inGrid = None
            self.outGrid = None
            toSpace = space
            dim = space.dim
        elif isinstance(space, VolSpace):
            self.inGrid = space.grid
            self.outGrid = GaussFT.getGrid(space.grid)
            toSpace = VolSpace(self.outGrid)
            dim = len(space.grid)
        elif isinstance(space, ProjSpace):
            self.inGrid = tuple(space.detector) + tuple(space.ortho)
            self.outGrid = tuple(GaussFT.getGrid(space.detector))
            toSpace = ProjSpace(detector=self.outGrid, orientations=space.orientations,
                                ortho=space.ortho)
            self.outGrid += tuple(space.ortho)
            dim = len(space.detector) + 1
        else:
            raise ValueError

        Dictionary.__init__(self, space, toSpace,
                            isLinear=True, isInvertible=True, inverse=self.__inverse)

        if self.inGrid is not None:
            if dim == 2:
                from code.bin import FTGaussian2D
                self.__evalFT = FTGaussian2D.evaluate
                self.__derivsFT = FTGaussian2D.derivs
            else:
                from code.bin import FTGaussian3D
                self.__evalFT = FTGaussian3D.evaluate
                self.__derivsFT = FTGaussian3D.derivs

        self.dim = dim

    def __call__(self, var):
        if isinstance(var, AtomElement):
            FT = var.copy()
            _analyticGaussFourier(var.I, var.x, var.r, FT.I, FT.x, FT.r)
            if self.outGrid is not None:
                FT = self.__evalFT(FT, self.outGrid)
            else:
                return FT
        else:
            FT = self.__DFT(var, False)
        if isinstance(self.ProjectionSpace, ProjSpace):
            return ProjElement(self.ProjectionSpace, FT)
        else:
            return VolElement(self.ProjectionSpace, FT)

    def __inverse(self, var):
        if isinstance(var, AtomElement):
            FT = var.copy()
            _analyticGaussiFourier(var.I, var.x, var.r, FT.I, FT.x, FT.r)
        else:
            FT = self.__DFT(var, True)
            if isinstance(self.ElementSpace, ProjSpace):
                return ProjElement(self.ElementSpace, FT)
            else:
                return VolElement(self.ElementSpace, FT)
        return FT

    def __DFT(self, arr, inverse=False):
        '''
        X = (x-x0)/dx = 0, 1, ..., L-1
        K = k/dk = -L/2, ..., 0, 1, ..., L/2
        dft(arr)[K] = \sum_X arr[X] e^{-2\pi i KX/L}
                    = \sum_X arr[X] e^{-2\pi i kx/(dkdxL)}e^{2\pi/(dkdxL) i k\cdot x0}

        FT(k) = \int arr(x)e^{-2\pi ik\cdot x}dx
              = \int arr(x) e^{-2\pi ik\cdot x}dx 
              ~ |dx|\sum_x arr[X] e^{-2\pi i Kdk\cdot (Xdx+x0)}
              = |dx| dft(arr)[K] e^{-2\pi i k\cdot x0}
              if 1 = dxdk L


        Assume periodicity so fft(arr)[-k] = fft(arr)[N-k] via fftshift

        if dim == len(inGrid):
            Do a full FT on the array
        else:
            Use Fourier slice theorem
        '''

        if hasattr(arr, 'asarray'):
            arr = arr.asarray()
        arr = context().asarray(arr)
        if inverse:
            dx = [1 / (g.item(1) - g.item(0))
                  for g in self.inGrid]
            x0 = [-g.item(0) for g in self.inGrid]
        else:
            dx = [g.item(1) - g.item(0) for g in self.inGrid]
            x0 = [g.item(0) for g in self.inGrid]

        if isinstance(self.ElementSpace, VolSpace):
            if inverse:
                FT = _applyweight(arr, self.outGrid, dx, x0)
                FT = ifftn(ifftshift(FT))
            else:
                FT = fftshift(fftn(arr))
                FT = _applyweight(FT, self.outGrid, dx, x0)
        else:
            '''
            arr[t,x] = \int_{y-x || t} f(y)dy
            FT[t,X] = FT[f](X)
            x[t,i,...] = detector[0][i]*w[0][t]+...
            '''
            ax = [i + 1 for i in range(self.dim - 1)]
            detector = self.outGrid[:self.dim - 1]
            w = self.outGrid[self.dim - 1:]

            if inverse:
                FT = _applyweight_hyperplane(arr, detector, w, dx, x0)
                FT = ifftn(ifftshift(FT, axes=ax), axes=ax)
            else:
                FT = fftshift(fftn(arr, axes=ax), axes=ax)
                FT = _applyweight_hyperplane(FT, detector, w, dx, x0)

        return FT

    @staticmethod
    def getGrid(grid):
        '''
        L = [len(g) for g in grid]
        dx = [len(g) for g in grid]
        1 = dxdk L
        '''
        dtype = grid[0].dtype
        dx = [g.item(1) - g.item(0) for g in grid]
        dim = len(dx)
        return [fftshift(fftfreq(grid[i].size, dx[i])).astype(dtype)
                for i in range(dim)]


class GaussFTVolume(Dictionary):
    def __init__(self, ASpace, PSpace):
        if isinstance(PSpace, ProjSpace):
            self.__grid = tuple(GaussFT.getGrid(PSpace.detector))
            VSpace = ProjSpace(detector=self.__grid, orientations=PSpace.orientations,
                               ortho=PSpace.ortho)
            self.__grid += tuple(PSpace.ortho)
        Dictionary.__init__(self, ASpace, VSpace, isLinear=True)

        dim = ASpace.dim
        if dim == 2:
            from code.bin.FTGaussian2D import evaluate, derivs
        else:
            from code.bin.FTGaussian3D import evaluate, derivs

        self.__eval = evaluate
        self.__derivs = derivs

    def __call__(self, atoms):
        return ProjElement(self.ProjectionSpace, self.__eval(atoms, self.__grid))

    def derivs(self, atoms):
        dim = atoms.space.dim
        iso = atoms.space.isotropic
        return dim, iso, lambda a, _, C: self.__derivs(a, self.__grid, C)


def doLoc_L2Step(res, atom):
    # argmin_{A,x} \int .5|res(y)-A*atom(y-x)|^2
    #    = argmin \int -A*res(y)atom(y-x) + A^2|atom|^2
    #    = argmax A(res\star atom) - A^2|atom|_2^2
    # atom(y) = e^{-.5|Ry|^2} \implies |atom|_2^2 = sqrt(2pi)^dim/(sqrt(2)|R|)
    # (this assumes data is 0 in unseen fourier data...)
    # NOTE: Computing maximum convolution requires a back-projection of the
    #         data! This is a bit of a pain. Maybe I can bluff it with two?
    FT = GaussFT(res.space)
    conv = FT.inverse(FT(res) * FT(atom)).asarray()
    # NEED TO DO BACK PROJECTION
    m = conv.max()
    ind = where(conv == m)
    dim = len(ind)
    i = random.choice(dim)
    x = [0] * dim
    for j in range(dim):
        x[j] = FT.inGrid[j][ind[j][i]]
    print(x)

    c = context()
    c.set(atom.x[:], x)

    r = c.asarray(atom.r)
    A = m / sqrt(2 * (2 * pi)**dim)
    if r.shape[1] == 1:
        A /= r[0]**dim
    else:
        A *= r[:dim].prod()
    c.set(atom.I[:], A)

    return atom


def _applyweight(f, grid, dx, x0):
    '''
    Multiply f by prod(dx)*e^{-2\pi i grid\cdot x_0}
    grid is the Fourier space grid
    '''
    dim = len(x0)
    grid = _tomesh(grid)
    w = 1
    for i in range(dim):
        if x0[i] != 0:
            f = f * exp(complex(0, -2 * pi * x0[i]) * grid[i])
        w *= dx[i]
    return f * w


def _applyweight_hyperplane(f, det, w, dx, x0):
    '''
    Multiply f by prod(dx)*e^{-2\pi i grid\cdot x_0}
    grid is the Fourier space grid
    grid[t,i,...] = det[0][i]*w[0][t] + ...
    If w are o.n. basis and x_0 is in that basis then we have
    (grid\cdot x_0)[t,i,...] = det[0][i]*x_0[0][t] + ...
    '''
    x0 = 2 * pi * array(x0)

    # w[j] = 2*pi*w[j]\cdot x_0
    if len(det) == 1:
        # 2D
        s = dx[0]
        if all([x == 0 for x in x0]):
            return f * s
        f = f * s * \
            exp(complex(0, -det[0].reshape(1, -1) * x0[0].reshape(-1, 1)))
    elif len(w) == 1:
        # 2.5D
        s = dx[0] * dx[1]
        f = f * s
        f *= exp(-1j * det[0].reshape(1, -1, 1) * x0[0].reshape(-1, 1, 1))
        f *= exp(-1j * det[1].reshape(1, 1, -1) * x0[1].reshape(-1, 1, 1))
    else:
        # 3D
        s = dx[0] * dx[1]
        f = f * s
        f *= exp(-1j * det[0].reshape(1, -1, 1) * x0[0].reshape(-1, 1, 1))
        f *= exp(-1j * det[1].reshape(1, 1, -1) * x0[1].reshape(-1, 1, 1))

    return f


def _analyticGaussFourier(I, x, r, newI, newx, newr):
    '''
    If f(y) = I e^{-|r(y-x)|^2/2} then
    FT[f](k) = \int f(y) e^{-2\pi iky} dy
             = I\int e^{-|ry - rx +2\pi ir^{-T}k|^2/2 + |rx-2\pi ir^{-T}k|^2/2 -|rx|^2/2}
             = (I(2\pi)^{dim/2}/|r|) e^{-|2\pi r^{-T}k|^2/2} e^{-i (2\pi x)\cdot k}
        new I = I(2\pi)^{dim/2}/|r|
        new r = 2\pi r^{-1}
        new x = 2\pi x
    F[f](k) = newI e^{-i (k\cdot newx)} e^{-|newr^T k|^2/2}
    '''
    nAtoms = x.shape[0]
    dim = x.shape[1]
    tau = 2 * pi
    if r.shape[1] == 1:
        for i in range(nAtoms):
            newI[i] = I[i] * sqrt(tau)**dim * (r[i, 0]**dim)
            # Scalar radii are still reciprocal -> e^{-(x/r)^2}
            newr[i, 0] = tau / r[i, 0]
            for j in range(dim):
                newx[i, j] = tau * x[i, j]
    else:
        for i in range(nAtoms):
            det = r[i, 0]
            for j in range(1, dim):
                det *= r[i, j]
            newI[i] = I[i] * sqrt(tau)**dim / det
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


def _analyticGaussiFourier(I, x, r, newI, newx, newr):
    '''
    If f(y) = newI e^{-|newr(y-newx)|^2/2} then 
    F[f](k) = (newI(2\pi)^(dim/2)/|newr|) e^{(-2\pi x)\cdot k i}e^{-|2\pi newr^{-T}k|^2/2}
        I = (newI(2\pi)^{dim/2}/|newr|)
        r = 2\pi newr^{-1}
        x = 2\pi newx
    so:
        newx = x/(2\pi)
        newr = 2\pi r^{-1}
        newI = I*|new r|/(2\pi)^{dim/2}
    '''
    nAtoms = x.shape[0]
    dim = x.shape[1]
    tau = 2 * pi
    if r.shape[1] == 1:
        for i in range(nAtoms):
            for j in range(dim):
                newx[i, j] = x[i, j] / tau
            newr[i, 0] = tau / r[i, 0]
            newI[i] = I[i] / ((newr[i, 0]**dim) * sqrt(tau)**dim)
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
            newI[i] = I[i] * det / sqrt(tau)**dim


def _tomesh(v):
    d = len(v)
    k = [None] * d
    for i in range(d):
        sz = [1] * d
        sz[i] = -1
        k[i] = v[i].reshape(sz)
    return k


if __name__ == '__main__':
    import numpy as np
    from matplotlib import pyplot as plt
    sz = [256] * 3
    x = _tomesh([10 * np.linspace(-.3, 1, s).astype('f4') for s in sz])
#     x = _tomesh([10 * np.linspace(-.3, 1, sz[0]).astype('f4'),
#                  10 * np.linspace(-.3, 1, sz[1]).astype('f4'),
#                  10 * np.linspace(-.3, 1, sz[2]).astype('f4')])
    k = _tomesh(GaussFT.getGrid(x))
    vSpace = VolSpace(x)
    fSpace = VolSpace(k)
    aSpace = AtomSpace(dim=len(sz), isotropic=False)

    a = aSpace.random(2, seed=2)
    a.r[:, :3] = (3 + a.r[:, :3]) / (3 + 1)
    a.r[:, 3:] = (0 + a.r[:, 3:]) / (3 + 1)
    a.x[:] = (a.x + 1) * 2 + 4

    # Ie^{-|R(x-m)|^2/2}
    volRep = 0
    for i in range(len(a)):
        if len(sz) == 2:
            X = [x[j] - a.x[i, j] for j in range(2)]
            Rx = [a.r[i, 0] * X[0] + a.r[i, 2] * X[1], a.r[i, 1] * X[1]]
            volRep += a.I[i] * exp(-(Rx[0]**2 + Rx[1]**2) / 2)
        else:
            X = [x[j] - a.x[i, j] for j in range(3)]
            Rx = [
                a.r[i, 0] * X[0] + a.r[i, 3] * X[1] + a.r[i, 5] * X[2],
                a.r[i, 1] * X[1] + a.r[i, 4] * X[2],
                a.r[i, 2] * X[2]]
            volRep += a.I[i] * exp(-(Rx[0]**2 + Rx[1]**2 + Rx[2]**2) / 2)
    # [sqrt(2pi)I/|R|] e^{-i k\cdot 2pi m} e^{-|2piR^{-1}k|^2/2}
    ftRep = 0
    for i in range(len(a)):
        if len(sz) == 2:
            det = a.r[i, 0] * a.r[i, 1]
            Rx = [a.r[i, 1] * k[0], a.r[i, 0] * k[1] - a.r[i, 2] * k[0]]
            ftRep += (a.I[i] * sqrt(2 * pi)**len(sz) / det) * \
                exp((-2 * pi * 1j) * (k[0] * a.x[i, 0] + k[1] * a.x[i, 1])) * \
                exp(-(Rx[0]**2 + Rx[1]**2) * (2 * pi * pi / (det * det)))
        else:
            det = a.r[i, 0] * a.r[i, 1] * a.r[i, 2]
            newr = [1 / a.r[i, 0], 1 / a.r[i, 1], 1 / a.r[i, 2],
                    -a.r[i, 3] / (a.r[i, 0] * a.r[i, 1]),
                    -a.r[i, 4] / (a.r[i, 1] * a.r[i, 2]),
                    (a.r[i, 3] * a.r[i, 4] / a.r[i, 1] - a.r[i, 5]) / (a.r[i, 0] * a.r[i, 2])]
            Rx = [newr[0] * k[0], newr[3] * k[0] + newr[1] * k[1],
                  newr[5] * k[0] + newr[4] * k[1] + newr[2] * k[2]]
            ftRep += (a.I[i] * sqrt(2 * pi)**len(sz) / det) * \
                exp((-2 * pi * 1j) * (k[0] * a.x[i, 0] + k[1] * a.x[i, 1] + k[2] * a.x[i, 2])) * \
                exp(-(Rx[0]**2 + Rx[1]**2 + Rx[2]**2) * (2 * pi * pi))

    # e^{-ik\cdot 2pi x0}
    if len(sz) == 2:
        P = exp(-1j * 2 * pi * (k[0] * x[0].item(0) + k[1] * x[1].item(0)))
    else:
        P = exp(-1j * 2 * pi * (k[0] * x[0].item(0) + k[1] * x[1].item(0)
                                + k[2] * x[2].item(0)))

    FT = GaussFT(vSpace)
    gFT = GaussFT(aSpace)

    # Volume to Volume:
    ftCalc = FT(volRep).asarray()
    volCalc = FT.inverse(ftCalc).asarray()
    # Atom to Atom:
    aCalc = gFT.inverse(gFT(a)).asarray()
    # Atom to volume
    vCalc = FT(a).asarray()

    print('Precision of DFT array->array:')
    print('FT(vol) error: ', abs(ftCalc - ftRep).max() / abs(ftRep).max())
    print('FT(atoms) error: ', abs(vCalc - ftRep).max() / abs(ftRep).max())
    print('FT.inverse(FT(vol)) == vol: ', abs(
        volRep - volCalc).max() / abs(volRep).max())

    print('\nPrecision of iFT(FT(atoms)) == atoms:')
    print('I check:', abs(aCalc.I - a.I).max())
    print('x check:', abs(aCalc.x - a.x).max())
    print('r check:', abs(aCalc.r - a.r).max())

    plt.subplot('121')
#     plt.plot(ftRep[492:534, int(sz[1] / 2), int(sz[2] / 2)].real)
#     plt.plot(abs(ftRep[246:267, int(sz[1] / 2), int(sz[2] / 2)]))
    plt.imshow(FT.inverse(ftRep).real.sum(-1))
    plt.colorbar()
    plt.subplot('122')
#     plt.plot(abs(ftRep / ftCalc)[492:534, int(sz[1] / 2), int(sz[2] / 2)].real)
#     plt.plot(abs(ftCalc[246:267, int(sz[1] / 2), int(sz[2] / 2)]))
    plt.imshow(FT.inverse(ftCalc).real.sum(-1))
    plt.colorbar()
    plt.show()

#     plt.figure()
#     plt.subplot('231')
#     plt.imshow(volRep)
#     plt.subplot('232')
#     plt.imshow(FT.inverse(ftCalc).real)
#     plt.subplot('233')
#     plt.imshow(np.log10(abs(ftRep - ftCalc)))
#     plt.colorbar()
#
#     plt.subplot('234')
#     plt.imshow(volRep)
#     plt.subplot('235')
#     plt.imshow(FT.inverse(vCalc).real)
#     plt.subplot('236')
#     plt.imshow(np.log10(abs(ftRep - vCalc)))
#     plt.colorbar()
#
#     plt.show(block=True)
