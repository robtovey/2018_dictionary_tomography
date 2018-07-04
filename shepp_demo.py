'''
Created on 20 Dec 2017

@author: Rob Tovey
'''
import odl
from numpy import pi, array, sqrt, ones, meshgrid, zeros
from matplotlib import pyplot as plt
from atomFuncs import theta_to_vec, GaussRadon_2D_CPU, GaussVol_2D_CPU

# Reconstruction space variables
dim = 2
vol_res = [128, 128]
vol = odl.uniform_discr([-1, -1], [1, 1], vol_res)
vol_grid = vol.grid.meshgrid

# Projection space variables
theta_res = 180
theta = odl.uniform_discr(0, pi, theta_res)
w = theta_to_vec(theta.grid.coord_vectors)
p_res = 128
p = odl.uniform_discr(-sqrt(dim), sqrt(dim), p_res)
p = p.grid.coord_vectors

# Define atoms
small_vol = odl.uniform_discr([-1, -1], [1, 1], [128, 128])
phantom = odl.phantom.shepp_logan(small_vol, modified=True)
x0 = meshgrid(*small_vol.grid.meshgrid, indexing='ij')
x0 = array([x.reshape(-1) for x in x0], order='F').T
r = ones(x0.shape[0]) / 40
I = phantom.asarray().reshape(-1)
x0 = x0[I.nonzero(), :].squeeze()
r = r[I.nonzero()]
I = I[I.nonzero()]

# Cast to single precision
x0 = x0.astype('float32', order='C')
r = r.astype('float32', order='C')
I = I.astype('float32', order='C')
w = w.astype('float32', order='C')
p = [P.astype('float32', order='C') for P in p]
vol_grid = [P.astype('float32', order='C') for P in vol_grid]

# Compute Radon transform and volume view
R = zeros((w.shape[0], len(p[0])), dtype='float32', order='C')
GaussRadon_2D_CPU(I, x0, r, w, p[0], R)
u = zeros((vol_grid[0].size, vol_grid[1].size), dtype='float32', order='C')
GaussVol_2D_CPU(I, x0, r, vol_grid[0].reshape(-1), vol_grid[1].reshape(-1), u)

# Show two representations
plt.subplot(121)
plt.imshow(phantom.asarray().T, origin='lower')
plt.subplot(122)
plt.imshow(u.T, origin='lower')

# Take sinogram
plt.figure()
plt.imshow(R.T)


plt.show()
print('Shepp Logan demo complete')
