'''
Created on 20 Dec 2017

Store for first attempt at atomic Radon operators. These work as an 
integral in the orthoganal space which is not the right extension 
into more that 2 dimensions. 

@author: Rob Tovey
'''
from numpy import pi, exp
import numba


def __Radon_2D(I, x, r, w, p):
    '''
An atom is going to be defined by a position x = [x1,x2,...] and radius,
r such that density is:
    u(y) = exp(-|y-x|^2/(2r^2))
Radon transform becomes:
    Ru(w,p) = \int_{y\cdot w = p} u(y)
            = (2\pi r^2)^{(dim-1)/2} exp(-|p-x\cdot w|^2/(2r^2))
    D_xRu(w,p) = \frac{p-x\cdot w}{r^2}Ru(w,p)x
    D_rRu(w,p) = ((dim-1)/r + |p-x\cdot w|^2/r^3)Ru(w,p)
    '''

    # shape of everything: [proj,(projcoord+res),atom, (volcoord)]
    n = x.shape[-2]  # number of atoms
    dim = x.shape[-1]  # dimension of volume space
    sz = [[w.shape[0]], [len(P) for P in p], [x.shape[-2]], [x.shape[-1]]]
    # These two take values in volume space
    x.shape = [1] + [1] * (dim - 1) + [n, dim]
    w.shape = [-1] + [1] * (dim - 1) + [1, dim]
    # These two are scalars over atoms
    I.shape = [1] + [1] * (dim - 1) + [n]
    r.shape = [1] + [1] * (dim - 1) + [n]

    # Here cast that dim=2, i.e. dim(p) = 1
    p = p[0]
    p.shape = [1] + sz[1] + [1]

    r2 = r**2
    rM2 = 1 / r2

    s = I * ((2 * pi) * r2)**((dim - 1) / 2)
    interior = -(p - (x * w).sum(-1))**2 * (rM2 / 2)
    R = (s * exp(interior)).sum(-1)  # summation over atoms
    # R[w,p] = (s*exp(interior)).sum(atoms)
    # R.shape = sz[0]+sz[1]
    # s.shape = [1]+[1,...]+[atoms]
    # interior.shape = sz[0]+sz[1]+[atoms]
    # p.shape = [1]+sz[1]+[1]
    # (x*w).sum(-1).shape = sz[0]+[1,...]+[atoms]

    return R


@numba.jit("void(f4[:],f4[:,:],f4[:],f4[:,:],f4[:],f4[:,:])", target='cpu', nopython=True)
def __Radon_2D_CPU(I, x, r, w, p, R):
    s1 = I * (2 * pi * r**2)**(1 / 2)
    s2 = -1 / (2 * r**2)
    for jj in range(w.shape[0]):
        for kk in range(p.shape[0]):
            tmp = 0
            for ii in range(x.shape[0]):
                inter = x[ii, 0] * w[jj, 0] + x[ii, 1] * w[jj, 1]
                tmp += s1[ii] * exp((p[kk] - inter)**2 * s2[ii])
            R[jj, kk] = tmp
