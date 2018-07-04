'''
Created on 10 Mar 2018

@author: Rob Tovey
'''
from code.transport_loss import l2_squared_loss, Transport_loss
from os.path import join
from code.regularisation import null
from KL_GaussRadon import doKL_ProjGDStep_iso
RECORD = join('store', 'DogBones_small_20')
RECORD = None
import odl
from code.bin.dictionary_def import VolSpace, ProjSpace, AtomSpace, ProjElement
from code.bin.atomFuncs import GaussTomo, GaussVolume
from numpy import pi, loadtxt, asarray, ascontiguousarray
from code.bin.manager import myManager
from PIL import Image

with myManager(device='cpu', order='C', fType='float32', cType='complex64') as c:
    # Import data:
    angles = loadtxt(
        join('DogBones', 'Sample_A2_Tilt_Series_tiltcorr_cut.rawtlt'))
    angles = odl.RectPartition(
        odl.IntervalProd(-pi / 2, pi / 2), odl.RectGrid((pi / 180) * angles))
    # angles = odl.uniform_partition(-(pi / 180) * 69, (pi / 180) * 73, 71)
    data = Image.open(
        join('DogBones', 'Sample A2 Tilt Series_tiltcorr_cut.tif'))
    data = [asarray(data).T for i in range(
        data.n_frames) if data.seek(i) is None]
    # First dimension is angle, second is width, third is slice
    data = asarray(data)
    # Space settings:
    dim = 3
    ASpace = AtomSpace(dim, isotropic=False)
    vol = odl.uniform_partition(
        [-1] * 3, [1] * 3, (data.shape[1], data.shape[1], data.shape[2]))
#     data = ascontiguousarray(data[:, ::16, 500:564:4], dtype='float32')
#     vol = vol[::8, ::8, 500:564:4]
    data = ascontiguousarray(data[:, ::16, ::16], dtype='float32')
    vol = vol[::16, ::16, ::16]
#     for i in range(0, 71):
#         plt.gca().clear()
#         plt.imshow(data[i].T)
#         plt.title(str(i))
#         plt.pause(.1)
#     plt.show()
#     exit()

    box = [vol.min_pt, vol.max_pt]
    PSpace = ProjSpace(angles, odl.uniform_partition(
        vol.min_pt[1:], vol.max_pt[1:], data.shape[1:]))
    vol = VolSpace(odl.uniform_discr_frompartition(vol, dtype='float32'))

    # Initiate Recon:
    #####
    def newAtoms(n, seed=None):
        tmp = ASpace.random(n, seed=seed)
        c.set(tmp.r, 10, (slice(None), slice(None, 3)))
        c.set(tmp.r, 0, (slice(None), slice(3, None)))
        c.set(tmp.I[:], 1e-2)
#         c.div(tmp.x, 2, tmp.x)
        return tmp
    nAtoms = 20
    recon = newAtoms(nAtoms, 1)
    #####
    Radon = GaussTomo(ASpace, PSpace, device='GPU')
    view = GaussVolume(ASpace, vol, device='GPU')
    data = ProjElement(PSpace, data / 151786)
    dsum = c.sum(data.asarray())
#     c.mul(recon.I, dsum / c.sum(Radon(recon).asarray()), recon.I)
    R = Radon(recon)

#     view(recon).plot(plt, Sum=1)
#     plt.show(block=True)
#     exit()

    # Reconstruction:
    fidelity = l2_squared_loss(dim)
#     fidelity = Transport_loss(dim, device='GPU')

    reg = null(dim)

    def guess(d, a): return doKL_ProjGDStep_iso(d, a, 1e-0, Radon)
#     guess = None

    from NewtonGaussian import linesearch as GD
#     from NewtonGaussian import linesearch_block as GD
#     from GD_lib import linesearch as GD
    from GD_lib import linesearch_block as GD
    GD(recon, data, [100, 1], fidelity, reg, Radon, view,
       dim='xrI', guess=guess, RECORD=RECORD)
