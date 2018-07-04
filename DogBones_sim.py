'''
Created on 15 Mar 2018

@author: Rob Tovey
'''
from code.regularisation import null
'''
Created on 20 Feb 2018

@author: Rob Tovey
'''
from code.transport_loss import l2_squared_loss, Transport_loss
from os.path import join
from GD_lib import linesearch as GD
RECORD = join('store', 'DogBones_sim_small')
RECORD = None
import odl
from code.bin.dictionary_def import VolSpace, ProjSpace, AtomSpace, VolElement,\
    ProjElement
from code.bin.atomFuncs import GaussTomo, GaussVolume
from numpy import sqrt, pi, zeros, minimum, maximum, random, arange,\
    ascontiguousarray, loadtxt, log10
from matplotlib import pyplot as plt, animation as mv
from scipy.io import loadmat
from code.bin.manager import myManager

with myManager(device='cpu', order='C', fType='float32', cType='complex64') as c:
    # Space settings:
    dim = 3
    device = 'GPU'  # CPU or GPU
    ASpace = AtomSpace(dim, isotropic=False)

    angles = loadtxt(
        join('DogBones', 'Sample_A2_Tilt_Series_tiltcorr_cut.rawtlt'))
    angles = odl.RectPartition(
        odl.IntervalProd(-pi / 2, pi / 2), odl.RectGrid((pi / 180) * angles))
    gt = ascontiguousarray(loadmat(join('store', 'DogBones_sim'))[
                           'x'][:, :, :44], dtype='float32')
    # First axis is depth, second is width, third is slice

    vol = odl.uniform_partition([-1] * 3, [1] * 3, [1024] * 3)
    vol = odl.uniform_discr_frompartition(
        vol[:, :, 500:544], dtype='float32')
    PSpace = (angles, odl.uniform_partition(
        vol.min_pt[1:], vol.max_pt[1:], (1024 // 16, 44)))
    Radon = odl.tomo.RayTransform(
        vol, odl.tomo.Parallel3dAxisGeometry(*PSpace))
#     tmp = Radon(gt).__array__()
#     for i in range(0, 71):
#         plt.gca().clear()
#         plt.imshow(tmp[i].T)
#         plt.title(str(i))
#         plt.pause(.1)
#     plt.show()
#     exit()

    vol = odl.uniform_partition([-1] * 3, [1] * 3, [1024] * 3)
    vol = odl.uniform_discr_frompartition(
        vol[::4, ::4, 500:544], dtype='float32')
    box = [vol.min_pt, vol.max_pt]
    # First coord, height. Second width
    PSpace = ProjSpace(*PSpace)
    gt_sino = ProjElement(PSpace, Radon(gt).__array__())
    vol = VolSpace(vol)

    # Initiate Recon:
    #####
    def newAtoms(n, seed=None):
        tmp = ASpace.random(n, seed=seed)
        c.set(tmp.r, 10, (slice(None), slice(None, 3)))
        c.set(tmp.r, 0, (slice(None), slice(3, None)))
        c.set(tmp.I[:], 1e-1)
#         c.set(tmp.x, 0, (slice(None), [0]))
        c.set(tmp.x, 0, (slice(None), [2]))
#         c.div(tmp.x, 2, tmp.x)
        return tmp
    nAtoms = 100
    recon = newAtoms(nAtoms, nAtoms)
    # First dim is slice, second width, third depth
#     recon.x[:] = c.asarray([[0, -.5, 0], [0, .5, 0]])
    #####
    Radon = GaussTomo(ASpace, PSpace, device=device)
    view = GaussVolume(ASpace, vol, device=device)
    gt_view = VolElement(vol, gt[::4, ::4, :])
    R = Radon(recon)

#     V = view(recon)
#     V.plot(plt.subplot(121), Slice=[slice(None), slice(None), 128])
#     gt_view.plot(plt.subplot(122), Slice=[slice(None), slice(None), 128])
#     V.plot(plt.subplot(121), Sum=2)
#     gt_view.plot(plt.subplot(122), Sum=2)
#     plt.show()
#     for i in range(71):
#         plt.subplot(121).clear()
#         plt.subplot(122).clear()
#         R.plot(plt.subplot(121), Slice=[i, slice(None), slice(None)])
#         gt_sino.plot(plt.subplot(122), Slice=[i, slice(None), slice(None)])
#         plt.title(str(i))
#         plt.pause(.1)
#     exit()

    # Reconstruction:
    fidelity = l2_squared_loss(dim)
#     fidelity = Transport_loss(dim, device=device)

    reg = null(dim)

#     def guess(d, a): return doKL_ProjGDStep_2Diso(d, a, 1e-0, Radon)
#     def guess(d, a): return doKL_LagrangeStep_2Diso(d, a, 1e-3, Radon)
    def guess(d, a): return a

    GD(recon, gt_sino, [100, 1], fidelity, reg, Radon, view,
       gt=gt_view, dim='xrI', guess=guess, RECORD=RECORD)
