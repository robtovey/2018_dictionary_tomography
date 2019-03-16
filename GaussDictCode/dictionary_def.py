'''
Created on 4 Jan 2018

@author: Rob Tovey
'''
from numpy import zeros, concatenate, ones, tile, isscalar, squeeze, \
    array, prod
from numpy.linalg import norm
from numpy.random import rand, seed as np_seed
from .bin.manager import context
from odl import (Operator, uniform_discr, ComplexNumbers, DiscreteLp, DiscreteLpElement,
                 )
from odl.space import FunctionSpace
from odl.discr.discretization import tspace_type
from odl.set.space import LinearSpace, LinearSpaceElement


class DictionaryOp(Operator):
    '''
    A DictionaryOp is a map which can performs Operator actions
    from domain to range but where it is understood that the 
    domain is a non-standard representation of a general space.
    
    Example:
    We have the Fourier transform, F\colon L^2([0,1]) \to L^2(C).. 
    The standard representation is to discretise [0,1] and represent
    L^2([0,1]) \supset X_n, piece-wise constant functions with n 
    partitions. This generates the standard ODL Operator: 
            f\colon X_n\to Z_n 
    Suppose we wish to represent L^2([0,1]) \supset Y_m in a Wavelet 
    basis. There is clearly some natural definition of 
    F\colon Y_m\to C^n, say
            g\colon Y_m\to Z_n,
    but we also know there is an embedding of X_n in Y_m, say 
            A\colon Y_m\to X_n
    These properties generate an approximate equivalence:
            g(y) \approx f(A(y))
    g is the DictionaryOp object that we define here.

    Properties:
        domain (Y = space of dictionary elements)
        embedding (X = high level embedding space, probably R^n)
        range (Z = image of both Operators) 
        discretise (A\colon Y\to X is the embedding operator)
        linear (boolean, describes map Y\to Z)
    '''

    def __init__(self, domain, embedding, range, linear=False):

        Operator.__init__(self, domain=domain, range=range, linear=linear)
        self.embedding = embedding

    def discretise(self, e, out=None):
        '''
        This function should take in an element, e\in self.domain,
        and return its representation in self.embedding.
        '''
        raise NotImplementedError

# class Space:
# 
#     def __init__(self, dtype):
#         self.isLinear = False
#         self.dtype = dtype
# 
#     def zero(self):
#         return Element(self)
# 
#     def copy(self):
#         return Space(self.dtype)
# 
# 
# class Element:
#     __array_ufunc__ = None
#     # This is necessary to prevent numpy calling __array_wrap__
#     __array_priority__ = 100
# 
#     def __init__(self, space):
#         self.space = space
#         self.array = None
# 
#     def asarray(self):
#         return context().asarray(self.array)
# 
#     def copy(self):
#         return Element(self.space)
# 
#     def prnt(self):
#         print(super(self))
# 
#     def __neg__(self):
#         return type(self)(self.space, context().mul(self.array, -1))
# 
#     def __add__(self, other):
#         c = context()
#         if isinstance(other, type(self)):
#             return type(self)(
#                 self.space,
#                 c.add(self.array, other.array)
#             )
#         else:
#             return type(self)(
#                 self.space,
#                 c.add(self.array, asarray(other))
#             )
# 
#     def __radd__(self, other):
#         return type(self)(
#             self.space,
#             context().add(self.array, asarray(other))
#         )
# 
#     def __iadd__(self, other):
#         c = context()
#         if isinstance(other, type(self)):
#             c.add(self.array, other.array, self.array)
#         else:
#             c.add(self.array, asarray(other), self.array)
#         return self
# 
#     def __mul__(self, s):
#         if isinstance(s, type(self)):
#             return type(self)(
#                 self.space,
#                 context().mul(self.array, s.array)
#             )
#         else:
#             return type(self)(
#                 self.space,
#                 context().mul(self.array, asarray(s))
#             )
# 
#     def __rmul__(self, s):
#         return self.__mul__(s)
# 
#     def __imul__(self, s):
#         if isinstance(s, type(self)):
#             context().mul(self.array, s.array, self.array)
#         else:
#             context().mul(self.array, asarray(s), self.array)
#         return self
# 
#     def __sub__(self, other):
#         if isinstance(other, type(self)):
#             return type(self)(
#                 self.space,
#                 context().sub(self.array, other.array)
#             )
#         else:
#             return type(self)(
#                 self.space,
#                 context().sub(self.array, asarray(other))
#             )
# 
#     def __isub__(self, other):
#         if isinstance(other, type(self)):
#             context().sub(self.array, other.array, self.array)
#         else:
#             context().sub(self.array, asarray(other), self.array)
#         return self


class AtomSpace(LinearSpace):

    def __init__(self, dim, isotropic=True):
        LinearSpace.__init__(self, ComplexNumbers())
        self.dim = dim
        self.isotropic = isotropic
        self.__size = 1 + dim
        if isotropic:
            self.__size += 1
        elif dim == 2:
            self.__size += 3
        else:
            self.__size += 6

    def element(self, *args, **kwargs):
        if len(args) == 1:
            args = args[0]
            if hasattr(args, 'ndim') and args.ndim == self.__size:
                return AtomElement(self, args[:, 1:self.dim + 1], args[:, self.dim + 1:], args[:, :1], **kwargs)
            else:
                raise ValueError('input could not be interpetted')
        else:
            return AtomElement(self, *args, **kwargs)
        
    def _lincomb(self, a, x1, b, x2, out):
        c = context()
        n, m = (x1.size, x2.size), out.size
        if n[0] == 0:
            n = n[1], n[0]
            a, x1, b, x2 = b, x2, a, x1

        if sum(n) == m:
            if n[1] == 0:
                c.mul(a, x1.I, out.I)
                c.set(out.x, x1.x)
                c.set(out.r, x1.r)
            else:
                out.I = c.mul(a, x1.I)
                out.x = c.copy(x1.x)
                out.r = c.copy(x1.r)
        else:
            if n[1] == 0:
                out.I = c.mul(a, x1.I)
                out.x = c.copy(x1.x)
                out.r = c.copy(x1.r)
            else:
                out.I = c.cat(c.mul(a, x1.I), c.mul(b, x2.I), axis=0)
                out.x = c.cat(x1.x, x2.x, axis=0)
                out.r = c.cat(x1.r, x2.r, axis=0)
        
    def _dist(self, x1, x2): return self.norm(x1 - x2)

    def _norm(self, x): return sum(norm(t.reshape(-1)) for t in (x.I, x.x, x.r))
    
    def zero(self): return AtomElement(self, zeros((0, self.dim)), zeros(0), zeros(0))

    def empty(self): return AtomElement(self, zeros((0, self.dim)), zeros(0), zeros(0))

    def random(self, n, seed=None):
        if seed is not None:
            np_seed(seed)
        if self.isotropic:
            r = rand(n, 1) + 1
        else:
            r = rand(n, (self.dim * (self.dim + 1)) // 2) + 1
            r[:, self.dim:] /= 4
        
        return AtomElement(self, .8 * (2 * rand(n, self.dim) - 1), r, 2 * rand(n))

    def copy(self): return AtomSpace(self.dim, self.isotropic)

    def __eq__(self, other):
        return (self.dim == getattr(other, 'dim', None)) and (self.isotropic == getattr(other, 'isotropic', None))


class AtomElement(LinearSpaceElement):

    def __init__(self, space, x=None, r=None, I=None):
        # TODO: this should be kwargs and I, x, r
        LinearSpaceElement.__init__(self, space)

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
        elif 2 * self.r.shape[1] != space.dim * (space.dim + 1):
                raise ValueError

        if I.size == 1:
            I = I * ones(self.x.shape[0])
        self.I = I.reshape(-1)
        if self.I.size != self.x.shape[0]:
            raise ValueError

        self.x = c.cast(self.x)
        self.r = c.cast(self.r)
        self.I = c.cast(self.I)
        self.size = self.I.size

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

    def asarray(self): return self.data

    @property
    def data(self):
        c = context()
        n = self.x.shape[0]
        return concatenate(c.asarray(self.x), c.asarray(self.r.reshape(n, -1)),
                           c.asarray(self.I.reshape(n, 1)), axis=1)
        
    def __len__(self): return self.size


class VolSpace(DiscreteLp):
    
    def __init__(self, partition, impl='numpy', **kwargs):
        c = context()
        dtype = c.fType
        fspace = FunctionSpace(partition.set, out_dtype=dtype)
        ds_type = tspace_type(fspace, impl, dtype)
        dtype = ds_type.default_dtype() if dtype is None else dtype 

        exponent = kwargs.pop('exponent', 2.0)
        tspace = ds_type(partition.shape, dtype, exponent=exponent,
                         weighting=1)
        DiscreteLp.__init__(self, fspace, partition, tspace, **kwargs)
        self.coord_vectors = tuple(c.cast(g) for g in self.grid.coord_vectors)
        
    def element(self, arr): return VolElement(self, arr)


class VolElement(DiscreteLpElement):

    def __init__(self, space, u):
        if isscalar(u) or (u.size == 1):
            u = u * ones(space.shape)
        DiscreteLpElement.__init__(self, space,
                                   space.tspace.element(context().cast(u)))

    def plot(self, ax, *args, Slice=None, Sum=-1, T=True, **kwargs):
        arr = self.data.real
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

    def copy(self): return VolElement(self.space, self.data)


class ProjSpace(LinearSpace):

    def __init__(self, angles=None, detector=None, orientations=None, ortho=None):
        LinearSpace.__init__(self, ComplexNumbers())
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

    def element(self, arr=None): return ProjElement(self, 0 if arr is None else arr)

    def _lincomb(self, a, x1, b, x2, out):
        c = context()
        c.add(c.mul(a, x1.data), c.mul(b, x2.data), out.data)

    def _dist(self, x1, x2): return self.norm(x1 - x2)

    def _norm(self, x): return norm(x.data.reshape(-1))

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

        return ProjSpace(orientations=orientations, ortho=ortho,
                         detector=self.detector)

    def __eq__(self, other):
        shape = getattr(other, 'shape', [None])
        if len(shape) != len(self.shape):
            return False
        else:
            return all(shape[i] == self.shape[i] for i in range(len(shape)))

    @property
    def volume(self):
        return self.orientations.shape[0] * abs(prod([p[-1] + p[1] - 2 * p[0] for p in self.detector]))


class ProjElement(LinearSpaceElement):

    def __init__(self, space, u):
        LinearSpaceElement.__init__(self, space)
        if isscalar(u) or (u.size == 1):
            u = u * ones(space.shape)
        self.data = context().cast(u)

        if not tuple(self.data.shape) == tuple(self.space.shape):
            raise ValueError('Shape of array does not match shape of space')
        self.shape = self.data.shape

    def asarray(self): return self.data

    def copy(self): return ProjElement(self.space, self.data)

    def plot(self, ax, origin='lower', aspect='auto', Slice=None, *args, **kwargs):
        arr = self.data.real
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

# def sliceGrid(grid, Slice):
#     # Calc dim:
#     if type(Slice) is slice or isscalar(Slice):
#         dim = 1
#     elif hasattr(Slice, '__getitem__'):
#         if isscalar(Slice[0]):
#             dim = 1
#         else:
#             dim = len(Slice)
#     else:
#         raise ValueError
# 
#     # Slice grid:
#     grid = _getcoord_vectors(grid)
#     if dim == 1:
#         newgrid = (grid[0][Slice],) + grid[1:]
#     else:
#         newgrid = (grid[i][Slice[i]] for i in range(dim)) + grid[dim:]
# 
#     return newgrid


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
