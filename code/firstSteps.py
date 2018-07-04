'''
Created on 18 Dec 2017

@author: Rob Tovey
'''

# An atom is going to be defined by a position x = [x1,x2,...,xn] and radius,
# r such that density is:
#     u(y) = exp(-|y-x|^2/(2r^2))
# Radon transform becomes:
#     Ru(w,p) = \int_{y\cdot w = p} u(y)
#             = (2\pi r^2)^{(n-1)/2} exp(-|p-x\cdot w|^2/(2r^2))
#     D_xRu(w,p) = \frac{p-x\cdot w}{r^2}Ru(w,p)x
#     D_rRu(w,p) = ((n-1)/r + |p-x\cdot w|^2/r^3)Ru(w,p)


import numpy as np
from numpy import pi, linspace, cos, sqrt, exp
from matplotlib import pyplot as plt


# Reconstruction space variables
dim = 2
vol_res = [128, 128]
vol = [linspace(0, 1, vol_res[k], endpoint=True) for k in range(dim)]

# Projection space variables
theta_res = 180
theta = [linspace(0, pi, theta_res) for _ in range(dim - 1)]


def theta_to_vec(theta):
    n = [len(t) for t in theta]
    dim = len(n) + 1
    C = [cos(t) for t in theta]
    S = [sqrt(1 - c**2) for c in C]
    w = np.empty([np.prod(n), dim])

    if dim == 2:
        for i in range(n[0]):
            w[i, 0] = C[0][i]
            s = S[0][i]
            w[i, 1] = s
    elif dim == 3:
        for i in range(n[0]):
            for j in range(n[1]):
                w[n[1] * i + j, 0] = C[0][i]
                s = S[0][i]
                w[n[1] * i + j, 1] = s * C[1][j]
                s *= S[1][j]
                w[n[1] * i + j, 2] = s
    elif dim == 4:
        for i in range(n[0]):
            for j in range(n[1]):
                for k in range(n[2]):
                    w[n[2] * (n[1] * i + j) + k, 0] = C[0][i]
                    s = S[0][i]
                    w[n[2] * (n[1] * i + j) + k, 1] = s * C[1][j]
                    s *= S[1][j]
                    w[n[2] * (n[1] * i + j) + k, 2] = s * C[2][k]
                    s *= S[2][k]
                    w[n[2] * (n[1] * i + j) + k, 3] = s

    return w


w = theta_to_vec(theta)
p_res = 128
p = [linspace(-sqrt(dim), sqrt(dim), p_res, endpoint=True)]


# Define atoms
n_atoms = 4
x = 2 * np.array([np.random.rand(dim) for _ in range(n_atoms)]) - 1
r = (1 + 2 * np.random.rand(n_atoms)) / 50
I = 1 + 0 * 2 * np.random.rand(n_atoms)


def Radon(I, x, r, w, p):
    # shape of everything: [proj,(projcoord+res),atom, (volcoord)]
    n = x.shape[1]
    sz = [[w.shape[0]], [len(P) for P in p], x.shape[0], n]
    # These two take values in volume space
    x.shape = [1] + [1] * (n - 1) + [-1, n]
    w.shape = [-1] + [1] * (n - 1) + [1, n]
    # These two are scalars over atoms
    I.shape = [1] + [1] * (n - 1) + [-1]
    r.shape = [1] + [1] * (n - 1) + [-1]

    p = np.concatenate(np.meshgrid(*p, indexing='ij'), axis=-1)
    # Here cast that n=2, i.e. dim(p) = 1
    p.shape = [1] + sz[1] + [1]

    r2 = r**2
    rM2 = 1 / r2

    s = ((2 * pi) * r2)**((n - 1) / 2)
    interior = -(p - (x * w).sum(-1))**2 * (rM2 / 2)
    R = (s * exp(interior)).sum(-1)  # summation over atoms
    # R[w,p] = (s*exp(interior)).sum(atoms)
    # R.shape = sz[0]+sz[1]
    # s.shape = [1]+[1,...]+[atoms]
    # interior.shape = sz[0]+sz[1]+[atoms]
    # p.shape = [1]+sz[1]+[1]
    # (x*w).sum(-1).shape = sz[0]+[1,...]+[atoms]

    return R


R = Radon(I, x, r, w, p)

x.shape = x.shape[-2:]
plt.plot(x[:, 0], x[:, 1], 'bo')
plt.figure()

if theta_res == 6:
    for k in range(6):
        plt.subplot(2, 3, k + 1)
        plt.plot(p[0], R[k])
    #     plt.imshow(R[k])
        plt.title(str(w[k]))
else:
    plt.imshow(R.T)
plt.show()

print('Script complete')
