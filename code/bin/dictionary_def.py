'''
Created on 4 Jan 2018

@author: Rob Tovey
'''
from numpy import zeros, concatenate, ones, asarray, tile, isscalar, squeeze,\
    array, prod
from numpy.random import rand, seed as np_seed
from .manager import context
import odl


class Dictionary:
    '''
    A Dictionary is a map which can project Dictionary elements
    to their projection space. 

    Properties:
        ElementSpace
        ProjectionSpace
        isLinear
        isInvertible
        inverse

    Note that inverse is a Dictionary object which projects 
    from projection space to element space. The default behaviour
    is to project every element to the null element. If a 
    Dictionary is not invertible then the "inverse" can be any
    pseudo-inverse type projection.
    '''

    def __init__(self, Element, Projection, isLinear=False,
                 isInvertible=False, inverse=None):

        self.ElementSpace = Element
        self.ProjectionSpace = Projection
        self.isLinear = isLinear
        self.isInvertible = isInvertible
        if inverse is None:
            self.inverse = Dictionary(Projection, Element, inverse=self)
        else:
            self.inverse = inverse

    def __call__(self, e):
        '''
        This function should take in an element, e, in
        self.ElementSpace and return its projection in
        self.ProjectionSpace.
        '''
        return self.ProjectionSpace.null()


class Space:
    def __init__(self, dtype):
        self.isLinear = False
        self.dtype = dtype

    def null(self, Slice=slice(None)):
        return Element(self)

    def copy(self):
        return Space(self.dtype)


class Element:
    __array_ufunc__ = None
    # This is necessary to prevent numpy calling __array_wrap__
    __array_priority__ = 100

    def __init__(self, space):
        self.space = space
        self.array = None

    def asarray(self):
        return context().asarray(self.array)

    def copy(self):
        return Element(self.space)

    def prnt(self):
        print(super(self))

    def __neg__(self):
        return type(self)(self.space, context().mul(self.array, -1))

    def __add__(self, other):
        c = context()
        if isinstance(other, type(self)):
            return type(self)(
                self.space,
                c.add(self.array, other.array)
            )
        else:
            return type(self)(
                self.space,
                c.add(self.array, asarray(other))
            )

    def __radd__(self, other):
        return type(self)(
            self.space,
            context().add(self.array, asarray(other))
        )

    def __iadd__(self, other):
        c = context()
        if isinstance(other, type(self)):
            c.add(self.array, other.array, self.array)
        else:
            c.add(self.array, asarray(other), self.array)
        return self

    def __mul__(self, s):
        if isinstance(s, type(self)):
            return type(self)(
                self.space,
                context().mul(self.array, s.array)
            )
        else:
            return type(self)(
                self.space,
                context().mul(self.array, asarray(s))
            )

    def __rmul__(self, s):
        return self.__mul__(s)

    def __imul__(self, s):
        if isinstance(s, type(self)):
            context().mul(self.array, s.array, self.array)
        else:
            context().mul(self.array, asarray(s), self.array)
        return self

    def __sub__(self, other):
        if isinstance(other, type(self)):
            return type(self)(
                self.space,
                context().sub(self.array, other.array)
            )
        else:
            return type(self)(
                self.space,
                context().sub(self.array, asarray(other))
            )

    def __isub__(self, other):
        if isinstance(other, type(self)):
            context().sub(self.array, other.array, self.array)
        else:
            context().sub(self.array, asarray(other), self.array)
        return self


class AtomSpace(Space):
    def __init__(self, dim, isotropic=True):
        self.dim = dim
        self.isLinear = True
        self.isotropic = isotropic

    def null(self):
        return AtomElement(self, zeros((0, self.dim)), zeros(0), zeros(0))

    def zero(self):
        return AtomElement(self, zeros((0, self.dim)), zeros(0), zeros(0))

    def random(self, n, seed=None):
        if seed is not None:
            np_seed(seed)
        if self.isotropic:
            r = 1
        else:
            r = (self.dim * (self.dim + 1)) // 2
        if r == 1:
            return AtomElement(self, 2 * rand(n, self.dim) - 1, rand(n, r) / 20, rand(n))
        else:
            return AtomElement(self, 2 * rand(n, self.dim) - 1, rand(n, r) * 30, rand(n))

    def copy(self):
        return AtomSpace(self.dim, self.isotropic)


class AtomElement(Element):
    def __init__(self, space, x, r, I):
        Element.__init__(self, space)

        c = context()
        # This will copy to host even if it should stay on device
        x, r, I = c.asarray(x).copy(), c.asarray(r).copy(), c.asarray(I).copy()

        self.x = x.reshape(-1, space.dim)

        if r.size == 1:
            if space.isotropic:
                r = r * ones(self.x.shape[0])
            else:
                r = r * \
                    ones((self.x.shape[0], (space.dim * (space.dim + 1)) // 2))
                r[:, space.dim:] = 0
        elif (not space.isotropic) and (r.size == self.x.shape[0]):
            r = tile(r.reshape(-1, 1), (space.dim * (space.dim + 1)) // 2)
            r[:, space.dim:] = 0
        self.r = r.reshape(self.x.shape[0], -1)
        if space.isotropic:
            if self.r.size != self.x.shape[0]:
                raise ValueError
        else:
            if 2 * self.r.shape[1] != space.dim * (space.dim + 1):
                raise ValueError

        if I.size == 1:
            I = I * ones(self.x.shape[0])
        self.I = I.reshape(-1)
        if self.I.size != self.x.shape[0]:
            raise ValueError

        self.x = c.cast(self.x)
        self.r = c.cast(self.r)
        self.I = c.cast(self.I)

    def copy(self):
        return AtomElement(self.space, self.x, self.r, self.I)

    def __getitem__(self, i):
        return AtomElement(self.space, self.x[i], self.r[i], self.I[i])

    def __setitem__(self, i, atoms):
        c = context()
        if isscalar(i):
            c.set(self.x[i:i + 1], atoms.x)
            c.set(self.r[i:i + 1], atoms.r)
            c.set(self.I[i:i + 1], atoms.I[:])
        else:
            c.set(self.x[i], atoms.x)
            c.set(self.r[i], atoms.r)
            c.set(self.I[i], atoms.I[:])
        return self

    def __getattr__(self, name):
        if name is 'array':
            c = context()
            n = self.x.shape[0]
            return concatenate(c.asarray(self.x), c.asarray(self.r.reshape(n, -1)),
                               c.asarray(self.I.reshape(n, 1)), axis=1)
        else:
            raise AttributeError

    def __add__(self, other):
        if isinstance(other, AtomElement):
            c = context()
            return AtomElement(
                self.space,
                c.cat(self.x, other.x, axis=0),
                c.cat(self.r, other.r, axis=0),
                c.cat(self.I, other.I, axis=0)
            )
        else:
            raise ValueError

    def __iadd__(self, other):
        c = context()
        if isinstance(other, AtomElement):
            self.x = c.cat(self.x, other.x, axis=0)
            self.r = c.cat(self.r, other.r, axis=0)
            self.I = c.cat(self.I, other.I, axis=0)
        else:
            raise ValueError
        return self

    def __sub__(self, other):
        if isinstance(other, AtomElement):
            c = context()
            return AtomElement(
                self.space,
                c.cat(self.x, other.x, axis=0),
                c.cat(self.r, other.r, axis=0),
                c.cat(self.I, -other.I, axis=0)
            )
        else:
            raise ValueError

    def __isub__(self, other):
        c = context()
        if isinstance(other, AtomElement):
            self.x = c.cat(self.x, other.x, axis=0)
            self.r = c.cat(self.r, other.r, axis=0)
            self.I = c.cat(self.I, -other.I, axis=0)
        else:
            raise ValueError
        return self

    def __mul__(self, s):
        c = context()
        return AtomElement(
            self.space,
            self.x,
            self.r,
            c.mul(self.I, asarray(s))
        )

    def __rmul__(self, s):
        return self.__mul__(s)

    def __imul__(self, s):
        context().mul(self.I, asarray(s), self.I)
        return self

    def __len__(self):
        return self.x.shape[0]


class VolSpace(Space):
    def __init__(self, odlSpace):
        self.odlSpace = odlSpace
        c = context()
        self.grid = _getcoord_vectors(odlSpace)
        self.grid = [c.cast(g.reshape(-1)) for g in self.grid]
        self.shape = [g.size for g in self.grid]
        self.isLinear = True

    def null(self):
        return VolElement(self, 0)

    def zero(self):
        return VolElement(self, 0)

    def random(self):
        return VolElement(self, rand(*self.array.shape))

    def copy(self):
        return VolSpace(self.odlSpace)

    def __getitem__(self, Slice=slice(None)):
        if hasattr(self.odlSpace, 'grid'):
            tmp = sliceGrid(self.odlSpace.grid.coord_vectors, Slice)
        else:
            tmp = sliceGrid(self.odlSpace.coord_vectors, Slice)

        tmp = odl.uniform_partition_fromgrid(
            odl.RectGrid(*odl.RectGrid(tmp)))

        return VolSpace(tmp)


class VolElement(Element):
    def __init__(self, space, u):
        Element.__init__(self, space)
        if isscalar(u) or (u.size == 1):
            u = u * ones(space.shape)
        self.array = context().cast(u)
        if not tuple(self.array.shape) == tuple(self.space.shape):
            raise ValueError
        self.shape = self.array.shape

    def plot(self, ax, *args, Slice=None, Sum=-1, T=True, **kwargs):
        arr = self.asarray().real
        if Slice is not None:
            arr = squeeze(arr[Slice])
        if arr.ndim == 2:
            if T:
                arr = arr.T
            return ax.imshow(arr.T, *args, origin='lower', **kwargs)
        elif arr.ndim == 3:
            # Sum in third spatial dimension to project
            arr = squeeze(arr.sum(Sum))
            if T:
                arr = arr.T
            return ax.imshow(arr.T, *args, origin='lower', **kwargs)

    def copy(self):
        return VolElement(self.space, self.array)


class ProjSpace(Space):
    def __init__(self, angles=None, detector=None, orientations=None, ortho=None):
        from .atomFuncs import theta_to_vec, perp
        c = context()
        if angles is not None:
            angles = _getcoord_vectors(angles)
            self.orientations = theta_to_vec(angles)
            self.ortho = perp(angles)
        elif orientations is not None and ortho is not None:
            self.orientations = orientations
            self.ortho = ortho
        else:
            raise ValueError
        dim = self.orientations.shape[-1]
        self.orientations = c.cast(self.orientations.reshape(-1, dim))
        self.ortho = tuple(c.cast(o.reshape(-1, dim)) for o in self.ortho)

        self.detector = _getcoord_vectors(detector)
        self.detector = [c.cast(g.reshape(-1)) for g in self.detector]
        self.shape = [self.orientations.shape[0], ] + \
            [g.size for g in self.detector]
        self.isLinear = True

    def null(self, Slice=slice(None)):
        if Slice == slice(None):
            return ProjElement(self, 0)
        else:
            return ProjElement(self[Slice], 0)

    def zero(self, Slice=slice(None)):
        if Slice == slice(None):
            return ProjElement(self, 0)
        else:
            return ProjElement(self[Slice], 0)

    def random(self, Slice=slice(None)):
        if Slice == slice(None):
            return ProjElement(self, rand(*self.shape))
        else:
            tmp = self[Slice]
            return ProjElement(tmp, rand(*tmp.shape))

    def copy(self):
        c = context()
        return ProjSpace(orientations=c.copy(self.orientations),
                         ortho=tuple(c.copy(g) for g in self.ortho),
                         detector=tuple(c.copy(g) for g in self.detector))

    def __getitem__(self, Slice=slice(None)):
        c = context()
        orientations = c.toarray(self.orientations)[Slice]
        ortho = c.toarray(self.ortho)[Slice]

        return ProjSpace(orientations=orientations, ortho=ortho, detector=self.detector)

    def volume(self):
        return self.orientations.shape[0] * abs(prod([p[-1] - p[0] for p in self.detector]))


class ProjElement(Element):

    def __init__(self, space, u):
        Element.__init__(self, space)
        if isscalar(u) or (u.size == 1):
            u = u * ones(space.shape)
        self.array = context().cast(u)
        if not tuple(self.array.shape) == tuple(self.space.shape):
            raise ValueError('Shape of array does not match shape of space')
        self.shape = self.array.shape

    def copy(self):
        return ProjElement(self.space, self.array)

    def plot(self, ax, origin='lower', aspect='auto', Slice=None, *args, **kwargs):
        arr = self.asarray().real
        if Slice is not None:
            arr = squeeze(arr[Slice])
        if arr.ndim == 2:
            return ax.imshow(arr.T, *args, origin=origin,
                             aspect=aspect, **kwargs)
        elif arr.ndim == 3:
            # Concatenate along orientations to project
            arr = concatenate(arr, axis=0)
            return ax.imshow(arr.T, *args, origin=origin, aspect=aspect, **kwargs)
        elif arr.ndim == 4:
            # Compress orientations into single axis
            new = arr.reshape(-1, arr.shape[2], arr.shape[3])
            new = concatenate(new, axis=0)
            # Sum in second spatial dimension to project
#             new = new.sum(-1)
            return ax.imshow(new.T, *args, origin=origin, aspect=aspect, **kwargs)


def sliceGrid(grid, Slice):
    # Calc dim:
    if type(Slice) is slice or isscalar(Slice):
        dim = 1
    elif hasattr(Slice, '__getitem__'):
        if isscalar(Slice[0]):
            dim = 1
        else:
            dim = len(Slice)
    else:
        raise ValueError

    # Slice grid:
    grid = _getcoord_vectors(grid)
    if dim == 1:
        newgrid = (grid[0][Slice],) + grid[1:]
    else:
        newgrid = (grid[i][Slice[i]] for i in range(dim)) + grid[dim:]

    return newgrid


def _getcoord_vectors(thing):
    if hasattr(thing, 'grid'):
        return thing.grid.coord_vectors
    elif hasattr(thing, 'coord_vectors'):
        return thing.coord_vectors
    elif hasattr(thing, '__getitem__'):
        if isscalar(thing[0]):
            return (thing,)
        else:
            return tuple(array(t) for t in thing)
    else:
        raise ValueError
