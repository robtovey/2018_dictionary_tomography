'''
Created on 29 Mar 2018

@author: Rob Tovey
'''
from code.transport_loss import l2_squared_loss
from os.path import join
from GD_lib import linesearch as GD
RECORD = join('store', 'Bipyramid_small_20')
RECORD = None
import odl
from code.bin.dictionary_def import VolSpace, ProjSpace, AtomSpace, ProjElement
from code.bin.atomFuncs import GaussTomo, GaussVolume
from numpy import sqrt, zeros, random, arange, loadtxt, asarray, ascontiguousarray,\
    log10, log, pi
from matplotlib import pyplot as plt, animation as mv
from code.bin.manager import myManager
from PIL import Image
from code.regularisation import Joubert, null

with myManager(device='cpu', order='C', fType='float32', cType='complex64') as c:
    # Import data:
    angles = loadtxt(join('Materials', 'p10arot90_hdr0_5.rawtlt')) * pi / 180
    data = Image.open(join('Materials', 'p10arot90_hdr0_5.tif'))
    data = [asarray(data).T for i in range(
        data.n_frames) if data.seek(i) is None]
    # First dimension is angle, second is width, third is slice
    data = asarray(data)
    # Space settings:
    dim = 3
    ASpace = AtomSpace(dim, isotropic=False)
    vol = odl.uniform_partition(
        [-1] * 3, [1] * 3, (data.shape[1], data.shape[1], data.shape[2]))
    data = ascontiguousarray(data[2:-1, ::16, ::16], dtype='float32')
    angles = angles[2:-1]
    vol = vol[::16, ::16, ::16]
#     for i in range(0, len(angles)):
#         plt.gca().clear()
#         plt.imshow(data[i].T)
#         plt.title(str(i))
#         plt.show(block=False)
#         plt.pause(.3)
#     plt.show()
#     exit()

    PSpace = ProjSpace(angles, odl.uniform_partition(
        vol.min_pt[1:], vol.max_pt[1:], data.shape[1:]))
    vol = VolSpace(odl.uniform_discr_frompartition(vol, dtype='float32'))

    # Initiate Recon:
    #####
    def newAtoms(n, seed=None):
        tmp = ASpace.random(n, seed=seed)
        c.set(tmp.r[:], 10)
#         c.set(tmp.r, 10, (slice(None), slice(None, 3)))
#         c.set(tmp.r, 0, (slice(None), slice(3, None)))
        c.set(tmp.I[:], 1)
        return tmp
    nAtoms = 100
    recon = newAtoms(nAtoms, 1)
    #####
    Radon = GaussTomo(ASpace, PSpace, device='GPU')
    view = GaussVolume(ASpace, vol, device='GPU')
    data = ProjElement(PSpace, data / data.max())
    R = Radon(recon)
    # Reconstruction:
    fidelity = l2_squared_loss(dim)
    reg = Joubert(dim, 1e-2, 1e-2, (1e+1**dim, 1e-4))
#     reg = Joubert(dim, 1e-2, 1e-2, (1e-1**dim, 1e-1))
    reg = null(dim)

#     def guess(d, a): return doKL_ProjGDStep_2Diso(d, a, 1e-0, Radon)
#     def guess(d, a): return doKL_LagrangeStep_2Diso(d, a, 1e-3, Radon)
    def guess(d, a): return a

    GD(recon, data, [100, 1], fidelity, reg, Radon, view,
       dim='xrI', guess=guess, RECORD=RECORD)
