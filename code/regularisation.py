'''
Created on 5 Apr 2018

@author: Rob Tovey
'''
from code.bin.manager import context
from numpy import log, prod, zeros, exp, isscalar, infty, concatenate


class Regulariser:
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, x):
        return NotImplemented

    def grad(self, x, axis=0):
        return NotImplemented


class null(Regulariser):
    def __call__(self, atoms):
        return 0

    def grad(self, atoms, axis=0):
        c = context()
        if axis == 0 or axis == 'I':
            d = 0 * c.asarray(atoms.I)
        elif axis == 1 or axis == 'x':
            d = 0 * c.asarray(atoms.x)
        elif axis == 2 or axis == 'r':
            d = 0 * c.asarray(atoms.r)
        return d

    def hess(self, atoms):
        return 0


class Joubert(Regulariser):
    def __init__(self, dim, intensity, location, radius):
        # The larger intensity, the more uniform weights are
        # The larger location, the closer to the origin atoms are
        # radius[0]>0, minimal energy volume
        # The larger radius[1], the more uniform volumes are
        self.dim = dim
        self.param = (intensity, location,
                      radius[1], radius[1] * radius[0], prod(radius) * (1 - log(radius[0])))

    def __call__(self, atoms):
        # atoms.I, atoms.x, atoms.r
        # reg(atoms) = -a\log(I) + b/2|x|^2 + c|r| - d\log(|r|)
        c = context()
        I, x, r = c.asarray(atoms.I), c.asarray(atoms.x), c.asarray(atoms.r)

        if I.min() <= 0:
            return infty
        d = -self.param[0] * log(I).sum()
        d += (self.param[1] / 2) * (x * x).sum()

        if r.min() <= 0:
            return infty
        if atoms.space.isotropic:
            r = r**self.dim
        else:
            r = prod(r[:, :self.dim], axis=1)
        d += self.param[2] * r.sum() - self.param[3] * log(r).sum()
        d -= r.size * self.param[4]

        return d

    def grad(self, atoms, axis=0):
        # reg(atoms) = -a\log(I) + b/2|x|^2 + c|r| - d\log(|r|)
        c = context()
        I, x, r = c.asarray(atoms.I), c.asarray(atoms.x), c.asarray(atoms.r)

        if axis == 0 or axis == 'I':
            d = -self.param[0] / I
        elif axis == 1 or axis == 'x':
            d = self.param[1] * x
        elif axis == 2 or axis == 'r':
            if atoms.space.isotropic:
                d = (self.dim * self.param[2]) * \
                    r**(self.dim - 1) - (self.dim * self.param[3]) / r
            else:
                d = zeros(r.shape)
                ind = (slice(None), slice(self.dim))
                d[ind] = prod(r[ind], axis=1).reshape(-1, 1)
                d[ind] = (self.param[2] * d[ind] - self.param[3]) / r[ind]

        return d

    def hess(self, atoms):
        pass

    def __mul__(self, scal):
        if isscalar(scal):
            return Joubert(self.dim,
                           self.param[0] * scal,
                           self.param[1] * scal,
                           (self.param[3] / self.param[2], self.param[2] * scal))

    def __rmul__(self, scal):
        return self.__mul__(scal)


class Mass(Regulariser):
    def __init__(self, dim, m, k=2):
        # Volume is I/det(r)
        # reg = (I/det(r)-m)^{-k}
        self.dim = dim
        self.param = (m, k)

    def __call__(self, atoms):
        # atoms.I, atoms.x, atoms.r
        # reg(atoms) = (I/det(r)-m)^{-k}
        c = context()
        I, r = c.asarray(atoms.I), c.asarray(atoms.r)

        if I.min() <= 0:
            return infty
        if atoms.space.isotropic:
            V = I * r**self.dim
        else:
            V = I / prod(r[:, :self.dim], axis=1)
        V = abs(V)
        d = (V - self.param[0])**(-self.param[1])

        if V.min() <= self.param[0]:
            return infty

        return d.sum()

    def grad(self, atoms, axis=None):
        # reg(atoms) = (I/det(r)-m)^{-k} = (I*R-m)^{-k}
        c = context()
        I, x, r = c.asarray(atoms.I), c.asarray(atoms.x), c.asarray(atoms.r)

        if atoms.space.isotropic:
            R = r**self.dim
        else:
            R = 1 / prod(r[:, :self.dim], axis=1)

        dI = R
        dx = 0 * x
        if atoms.space.isotropic:
            dr = I * self.dim * r**(self.dim - 1)
        else:
            dr = zeros(r.shape)
            ind = (slice(None), slice(self.dim))
            dr[ind] = -(I * R).reshape(-1, 1) / r[ind]

        if axis == 0 or axis == 'I':
            d = dI
        elif axis == 1 or axis == 'x':
            d = dx
        elif axis == 2 or axis == 'r':
            d = dr
        else:
            d = concatenate((dI.reshape(-1), dx.reshape(-1), dr.reshape(-1)))

        return -self.param[1] * d * (I * R - self.param[0])**(-self.param[1] - 1)

    def hess(self, atoms):
        # reg(atoms) = (I/det(r)-m)^{-k} = (I*R-m)^{-k}
        # d_y = d(IR)_y*(-k(V-m)^{-k-1})
        # dd_{yz} = dd(IR)_{yz}*(-k(V-m)^{-k-1}) +
        #             d(IR)_y*d(IR)_z*(k(k+1)(V-m)^{-k-2})
        c = context()
        I, r = c.asarray(atoms.I), c.asarray(atoms.r)
        D = self.dim

        if atoms.space.isotropic:
            R = r**D
            dd = zeros((D + 2, D + 2))
        else:
            R = 1 / prod(r[:, :D], axis=1)
            dd = zeros((int(1 + D + (D * (D + 1)) / 2),
                        int(1 + D + (D * (D + 1)) / 2)))
        s = (-self.param[1] * (I * R - self.param[0])**(-self.param[1] - 1), self.param[1]
             * (self.param[1] + 1) * (I * R - self.param[0])**(-self.param[1] - 2))

        # dd_{II}
        dd[0, 0] = R * R * s[1]
        # dd_{Ix}
        dd[0, 1:1 + D], dd[1:1 + D, 0] = 0, 0
        # dd_{xx}
        dd[1:1 + D, 1:1 + D] = 0
        # dd_{xr}
        dd[1:1 + D, 1 + D:], dd[1 + D:, 1:1 + D] = 0, 0

        if atoms.space.isotropic:
            # dd_{Ir}
            dd[0, 1 + D] = (D * r**(D - 1)) * (s[0] + R * I * s[1])
            dd[1 + D, 0] = (D * r**(D - 1)) * (s[0] + R * I * s[1])
            # dd_{rr}
            dd[1 + D, 1 + D] = I * D * (D - 1) * r**(D - 2) * \
                s[0] + (I * D * r**(D - 1))**2 * s[1]
        else:
            # dd_{Ir}
            for i in range(D):
                dd[0, 1 + D + i] = -R / r[:, i] * (s[0] + I * R * s[1])
                dd[1 + D + i, 0] = -R / r[:, i] * (s[0] + I * R * s[1])
            # dd_{rr}
            for i in range(D):
                for j in range(i - 1):
                    dd[1 + D + i, 1 + D + j] = I * R / \
                        (r[:, i] * r[:, j]) * (s[0] + I * R * s[1])
                    dd[1 + D + j, 1 + D + i] = dd[1 + D + i, 1 + D + j]
                dd[1 + D + i, 1 + D + i] = I * R / \
                    (r[:, i] * r[:, i]) * (2 * s[0] + I * R * s[1])

        return dd


class Radius(Regulariser):
    def __init__(self, dim, s):
        # reg = s*log(I/det(r))
        self.dim = dim
        self.s = s

    def __call__(self, atoms):
        # reg = s*log(I/det(r))
        c = context()
        I, r = c.asarray(atoms.I), c.asarray(atoms.r)

        if I.min() <= 0 or r[:, :self.dim].min() <= 0:
            return infty
        if atoms.space.isotropic:
            d = -log(I) - self.dim * log(r)
        else:
            d = -log(I) + log(r[:, :self.dim]).sum(1)

        return self.s * d.sum()

    def grad(self, atoms, axis=None):
        # reg = s*log(I/det(r))
        c = context()
        I, x, r = c.asarray(atoms.I), c.asarray(atoms.x), c.asarray(atoms.r)

        dI = -1 / I
        dx = 0 * x
        if atoms.space.isotropic:
            dr = -self.dim / r
        else:
            dr = +1 / r
            dr[:, self.dim:] = 0

        if axis == 0 or axis == 'I':
            d = dI
        elif axis == 1 or axis == 'x':
            d = dx
        elif axis == 2 or axis == 'r':
            d = dr
        else:
            d = concatenate((dI.reshape(-1), dx.reshape(-1), dr.reshape(-1)))

        return self.s * d

    def hess(self, atoms):
        # reg(atoms) = s*log(I/det(r))
        # d_y = \pm s/y
        # dd_{yz} = \mp s/y^2
        c = context()
        I, r = c.asarray(atoms.I), c.asarray(atoms.r)
        D = self.dim

        if atoms.space.isotropic:
            dd = zeros((D + 2, D + 2))
        else:
            dd = zeros((int(1 + D + (D * (D + 1)) / 2),
                        int(1 + D + (D * (D + 1)) / 2)))

        # dd_{II}
        dd[0, 0] = 1 / (I * I)
        # dd_{xx}
        dd[1:1 + D, 1:1 + D] = 0

        if atoms.space.isotropic:
            # dd_{rr}
            dd[1 + D, 1 + D] = D / (r * r)
        else:
            # dd_{rr}
            for i in range(D):
                dd[1 + D + i, 1 + D + i] = -1 / (r[:, i] * r[:, i])

        return self.s * dd
