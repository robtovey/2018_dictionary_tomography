'''
Created on 4 Jan 2018

@author: Rob Tovey
'''
import odl
from numpy import pi, array, sqrt, ones, meshgrid
from matplotlib import pyplot as plt
from dictionary_def import AtomSpace, AtomElement, VolSpace, ProjSpace
from atomFuncs import GaussTomo, GaussVolume, BallRadon, BallVolume

# Reconstruction space variables
dim = 2
vol_res = [128, 128]
vol = VolSpace(odl.uniform_discr([-1, -1], [1, 1], vol_res))

# Projection space variables
theta = odl.uniform_discr(0, pi, 180)
p = odl.uniform_discr(-sqrt(dim), sqrt(dim), 128)
proj = ProjSpace(theta, p)

# Define atoms
small_vol = odl.uniform_discr([-1, -1], [1, 1], [64] * 2)
phantom = odl.phantom.shepp_logan(small_vol, modified=True)
x0 = meshgrid(*small_vol.grid.meshgrid, indexing='ij')
x0 = array([x.reshape(-1) for x in x0], order='F').T
r = ones(x0.shape[0]) / 40
I = phantom.asarray().reshape(-1)
x0, r, I = x0[I.nonzero(), :].squeeze(), r[I.nonzero()], I[I.nonzero()]
atom_space = AtomSpace(dim)
atoms = AtomElement(atom_space, x0, r, I)

# Compute Radon transform and volume view
#####
#     These lines use Gaussian intensities
Radon = GaussTomo(atom_space, proj)
view = GaussVolume(atom_space, vol)
#     These lines use binary intensities
# Radon = BallRadon(atom_space, proj)
# view = BallVolume(atom_space, vol)
#####
R = Radon(atoms)
u = view(atoms)

# Show two representations
plt.subplot(121)
plt.imshow(phantom.asarray().T, origin='lower')
plt.title('Original phantom')
plt.subplot(122)
u.plot(plt)
plt.title('Atomic phantom')

# Take sinogram
plt.figure()
R.plot(plt)
plt.title('Computed Sinogram')

plt.show()
print('Shepp Logan demo complete')
