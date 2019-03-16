'''
Created on 5 Apr 2018

@author: Rob Tovey
'''
from GaussDictCode.bin.manager import context
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
            r = r ** self.dim
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
                    r ** (self.dim - 1) - (self.dim * self.param[3]) / r
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

    def __init__(self, dim, m, k=2, s=1):
        # Volume is I/det(r)
        # reg = (I/det(r)-m)^{k}
        self.dim = dim
        self.param = (m, k, s)

    def __call__(self, atoms):
        # reg(atoms) = s*(I/det(r)-m)^{k}
        c = context()
        I, r = c.asarray(atoms.I), c.asarray(atoms.r)

        if I.min() < 0:
            return infty
        if atoms.space.isotropic:
            V = I * r ** self.dim
        else:
            V = I / prod(r[:, :self.dim], axis=1)
        V = abs(V)
        d = (V - self.param[0]) ** (self.param[1])

        return self.param[2] * d.sum()

    def grad(self, atoms, axis=None):
        # reg(atoms) = s(I/det(r)-m)^k = s(I*R-m)^k
        c = context()
        I, x, r = c.asarray(atoms.I), c.asarray(atoms.x), c.asarray(atoms.r)

        if atoms.space.isotropic:
            R = r ** self.dim
        else:
            R = 1 / prod(r[:, :self.dim], axis=1)

        dI = R
        dx = 0 * x
        if atoms.space.isotropic:
            dr = I * self.dim * r ** (self.dim - 1)
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

        return (self.param[2] * self.param[1]) * d * (I * R - self.param[0]) ** (self.param[1] - 1)

    def hess(self, atoms):
        # reg(atoms) = s(I/det(r)-m)^k = s(I*R-m)^k
        # d_y = d(IR)_y*(k(V-m)^{k-1})
        # dd_{yz} = dd(IR)_{yz}*(k(V-m)^{k-1}) +
        #             d(IR)_y*d(IR)_z*(k(k-1)(V-m)^{k-2})
        c = context()
        I, r = c.asarray(atoms.I), c.asarray(atoms.r)
        D = self.dim

        if atoms.space.isotropic:
            R = r ** D
            dd = zeros((D + 2, D + 2))
        else:
            R = 1 / prod(r[:, :D], axis=1)
            dd = zeros((int(1 + D + (D * (D + 1)) / 2),
                        int(1 + D + (D * (D + 1)) / 2)))
        s = (self.param[1] * (I * R - self.param[0]) ** (self.param[1] - 1), self.param[1]
             * (self.param[1] - 1) * (I * R - self.param[0]) ** (self.param[1] - 2))

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
            dd[0, 1 + D] = (D * r ** (D - 1)) * (s[0] + R * I * s[1])
            dd[1 + D, 0] = (D * r ** (D - 1)) * (s[0] + R * I * s[1])
            # dd_{rr}
            dd[1 + D, 1 + D] = I * D * (D - 1) * r ** (D - 2) * \
                s[0] + (I * D * r ** (D - 1)) ** 2 * s[1]
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

        return self.param[2] * dd


class Volume(Regulariser):

    def __init__(self, dim, r0, k=2, s=1):
        # reg = s*|det(r)-r0^dim|^k
        self.dim = dim
        self.__p = (abs(r0) ** dim, k, s)

    def __call__(self, atoms):
        # reg = s*|det(r)-r0^dim|^k
        c = context()
        r = c.asarray(atoms.r)

        if r[:, :self.dim].min() <= 0:
            return infty
        if atoms.space.isotropic:
            d = r ** -self.dim - self.__p[0]
        else:
            d = r[:, :self.dim].prod(1) - self.__p[0]

        return self.__p[2] * (abs(d) ** self.__p[1]).sum()

    def grad(self, atoms, axis=None):
        # reg = s*|det(r)-r0^dim|^k
        # dr_i = sk*(det(r)-r0^dim)|det(r)-r0^dim|^{k-2}*prod(r_{-i})
        c = context()
        I, x, r = c.asarray(atoms.I), c.asarray(atoms.x), c.asarray(atoms.r)

        dI = 0 * I
        dx = 0 * x
        if atoms.space.isotropic:
            d = r ** -self.dim - self.__p[0]
            dr = d * abs(d) ** (self.__p[1] - 2) * \
                (-self.__p[2] * self.dim * r ** (-self.dim - 1))
        else:
            R = r[:, :self.dim].prod(1, keepdims=True)
            d = R - self.__p[0]
            d = self.__p[2] * d * abs(d) ** (self.__p[1] - 2)

            dr = zeros(r.shape)
            ind = (slice(None), slice(self.dim))
            dr[ind] = d * R / r[ind]

        if axis == 0 or axis == 'I':
            d = dI
        elif axis == 1 or axis == 'x':
            d = dx
        elif axis == 2 or axis == 'r':
            d = dr
        else:
            d = concatenate((dI.reshape(-1), dx.reshape(-1), dr.reshape(-1)))

        return d

    def hess(self, atom):
        # reg = s*|det(r)-r0^dim|^k
        # dr_i = sk*sign(R)|R|^{k-1}*prod(r_{-i})
        # drr_{ij} = sk*sign(R)[ (k-1)sign(R)|R|^{k-2}prod(r_{-i})prod(r_{-j})
        #                        + |R|^{k-1}prod(r_{-ij})delta_{ij}]
        #          = sk|R|^{k-2}[ (k-1)prod(r_{-i})prod(r_{-j})
        #                        + R*prod(r_{-ij})delta_{ij}]
        # |r^{-d}-r0|^k -> -dk|r^{-d}-r0|^{k-1}r^{-d-1}
        #       -> dk|R|^{k-2}[d(k-1)r^{-2(d-1)} +(d+1)Rr^{-d-2}]
        c = context()
        r = c.asarray(atom.r)
        D = self.dim

        if atom.space.isotropic:
            dd = zeros((D + 2, D + 2))
        else:
            dd = zeros((int(1 + D + (D * (D + 1)) / 2),
                        int(1 + D + (D * (D + 1)) / 2)))

        # dd_{II}
        dd[0, 0] = 0
        # dd_{xx}
        dd[1:1 + D, 1:1 + D] = 0

        if atom.space.isotropic:
            # dd_{rr}
            R = r ** -self.dim - self.__p[0]
            scale = self.__p[2] * self.dim * self.__p[1] * R ** (self.__p[1] - 2)

            dd[1 + D, 1 + D] = scale * (
                self.dim * (self.__p[1] - 1) * r ** (-2 * (self.dim - 1))
                +(self.dim + 1) * R * r ** (-self.dim - 2))
        else:
            # dd_{rr}
            for i in range(D):
                dd[1 + D + i, 1 + D + i] = 1 / (r[:, i] * r[:, i])

        return self.s * dd


class Radius(Regulariser):

    def __init__(self, dim, r0, s, k):
        # reg = sum s*(r-r_0)^{-k}
        self.dim = dim
        self.r0 = r0
        self.s = s
        self.k = -k

    def __call__(self, atoms):
        # reg = sum s*(r-r_0)^{-k}
        c = context()
        r = c.asarray(atoms.r)

        if r[:, :self.dim].min() <= self.r0:
            return infty
        if atoms.space.isotropic:
            d = self.dim * (r - self.r0) ** self.k
        else:
            d = ((r[:, :self.dim] - self.r0) ** self.k).sum(axis=1)

        return self.s * d.sum()

    def grad(self, atoms, axis=None):
        # reg = sum s*(r-r_0)^{-k}
        c = context()
        I, x, r = c.asarray(atoms.I), c.asarray(atoms.x), c.asarray(atoms.r)

        dI = 0 * I
        dx = 0 * x
        if atoms.space.isotropic:
            dr = self.dim * self.k * (r - self.r0) ** (self.k - 1)
        else:
            dr = r.copy()
            dr[:, :self.dim] = self.k * \
                (r[:, :self.dim] - self.r0) ** (self.k - 1)
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
        # reg = sum s*(r-r_0)^{-k}
        c = context()
        r = c.asarray(atoms.r)
        D = self.dim

        if atoms.space.isotropic:
            dd = zeros((r.shape[0], D + 2, D + 2))
        else:
            dd = zeros((r.shape[0],
                        int(1 + D + (D * (D + 1)) / 2),
                        int(1 + D + (D * (D + 1)) / 2)))

        if atoms.space.isotropic:
            # dd_{rr}
            dd[:, 1 + D, 1 + D] = D * self.k * \
                (self.k - 1) * (r - self.r0) ** (self.k - 2)
        else:
            # dd_{rr}
            for i in range(D):
                dd[:, 1 + D + i, 1 + D + i] = self.k * \
                    (self.k - 1) * (r[:, i] - self.r0) ** (self.k - 2)

        return self.s * dd


class Iso(Regulariser):

    def __init__(self, dim, s):
        self.dim = dim
        self.s = s

    def __call__(self, atoms):
        r = context().asarray(atoms.r)

        if r.shape[0] == r.size:
            return 0
        else:
            d = abs(r[:, self.dim:])
            d = d * d
            return self.s * d.sum() / 2

    def grad(self, atoms, axis=None):
        c = context()
        I, x, r = c.asarray(atoms.I), c.asarray(atoms.x), c.asarray(atoms.r)

        dI = 0 * I
        dx = 0 * x
        if atoms.space.isotropic:
            dr = 0 * r
        else:
            dr = zeros(r.shape)
            dr[:, self.dim:] = r[:, self.dim:]

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
        # reg(atoms) = s(I/det(r)-m)^k = s(I*R-m)^k
        # d_y = d(IR)_y*(k(V-m)^{k-1})
        # dd_{yz} = dd(IR)_{yz}*(k(V-m)^{k-1}) +
        #             d(IR)_y*d(IR)_z*(k(k-1)(V-m)^{k-2})
        c = context()
        r = c.asarray(atoms.r)
        D = self.dim

        if atoms.space.isotropic:
            dd = zeros((D + 2, D + 2))
        else:
            dd = zeros((int(1 + D + (D * (D + 1)) / 2),
                        int(1 + D + (D * (D + 1)) / 2)))

        if not atoms.space.isotropic:
            # dd_{rr}
            for i in range(D, r.shape[1]):
                dd[1 + D + i, 1 + D + i] = self.s

        return dd
