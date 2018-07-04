'''
Created on 28 Jun 2018

@author: Rob Tovey
'''
from os.path import join
from code.bin.manager import myManager
from code.bin.dictionary_def import AtomSpace, VolSpace, ProjSpace, ProjElement,\
    VolElement
from code.bin.atomFuncs import GaussTomo, GaussVolume
from code.transport_loss import l2_squared_loss
from code.regularisation import null
from KL_GaussRadon import doKL_ProjGDStep_iso
RECORD = join('store', 'polyII_newton_rand100_iso')
# RECORD = None
if RECORD is not None:
    import matplotlib
    matplotlib.use('Agg')
import odl
from odl.contrib import mrc
from numpy import sqrt, ascontiguousarray, pi, pad

# Import data:
angles = odl.uniform_partition(-60 * pi / 180,
                               60 * pi / 180, 61, nodes_on_bdry=True)
with mrc.FileReaderMRC(join('PolyII', 'rna_phantom.mrc')) as f:
    f.read_header()
    gt = f.read_data()
    gt -= gt.min()
gt = pad(gt, ((2, 3), (0, 0), (10, 10)), 'constant')


# Space settings:
vol = list(gt.shape)
vol = odl.uniform_partition([-1] * 3, [1] * 3, vol)

vol = odl.uniform_discr_frompartition(vol[::1, ::1, ::1], dtype='float32')
gt = ascontiguousarray(gt[::1, ::1, ::1], dtype='float32')

PSpace = (angles,
          odl.uniform_partition([-sqrt(3)] * 2, [sqrt(3)] * 2, gt.shape[1:]))
PSpace = odl.tomo.Parallel3dAxisGeometry(*PSpace, axis=[0, 0, 1])

# Operators
Radon = odl.tomo.RayTransform(vol, PSpace)
data = Radon(gt)

with myManager(device='cpu', order='C', fType='float32', cType='complex64') as c:
    # Space settings:
    dim = 3
    device = 'GPU'
    ASpace = AtomSpace(dim, isotropic=True)
    vol = VolSpace(odl.uniform_discr([-1] * 3, [1] * 3, gt.shape))

    # Projection settings:
    PSpace = ProjSpace(odl.uniform_discr([-pi / 3, 0], [pi / 3, pi / 2], [61, 1],
                                         nodes_on_bdry=True),
                       odl.uniform_discr([-sqrt(dim)] * 2,
                                         [sqrt(dim)] * 2, data.shape[1:]))

    nAtoms = 100
    recon = ASpace.random(nAtoms, seed=1)
    c.set(recon.x[:], c.mul(recon.x, 1 / 3))
    c.set(recon.r, 10, (slice(None), slice(None, 3)))
    c.set(recon.r, 0, (slice(None), slice(3, None)))
    nAtoms = recon.I.shape[0]
    Radon = GaussTomo(ASpace, PSpace, device=device)
    view = GaussVolume(ASpace, vol, device=device)
    gt = VolElement(vol, gt)
    R = Radon(recon)
    data = ProjElement(PSpace, data.asarray())

#     from matplotlib import pyplot as plt
#     while True:
#         for i in range(data.shape[0]):
#             plt.gca().clear()
#             data.plot(plt, Slice=[i])
#             plt.title(str(i))
#             plt.pause(.1)
#     exit()

    # Reconstruction:
    fidelity = l2_squared_loss(dim)
    reg = null(dim)

    def guess(d, a): return doKL_ProjGDStep_iso(d, a, 1e-0, Radon)
    guess = None

    from NewtonGaussian import linesearch as GD
#     from NewtonGaussian import linesearch_block as GD
#     from GD_lib import linesearch as GD
    GD(recon, data, [1100, 1, 100], fidelity, reg, Radon, view,
       gt=gt, guess=guess, RECORD=RECORD, tol=1e-6, min_iter=2000)
