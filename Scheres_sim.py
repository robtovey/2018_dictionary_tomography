'''
Created on 18 Jul 2018

@author: Rob Tovey
'''
from os.path import join
RECORD = join('store', 'Scheres_sim_N_rand50')
RECORD = None
from odl.contrib import mrc
import odl
from numpy import linspace, pi, ascontiguousarray
from code.bin.manager import myManager
from code.bin.dictionary_def import AtomSpace, VolSpace, ProjSpace, VolElement,\
    ProjElement
from code.bin.atomFuncs import GaussTomo, GaussVolume
from code.transport_loss import l2_squared_loss
from code.regularisation import null
from KL_GaussRadon import doKL_ProjGDStep_iso

with mrc.FileReaderMRC(join('store', 'jasenko_1p8A_nonsharpened_absscale.mrc')) as f:
    _, gt = f.read()
    gt[gt < 0] = 0
    gt /= gt.max() / 2
angles = odl.RectGrid(linspace(-pi / 2, pi / 2, 40), linspace(0, pi / 2, 20))
angles = odl.uniform_partition_fromgrid(angles)

vol = list(gt.shape)
vol = odl.uniform_partition([-1] * 3, [1] * 3, vol)

vol = odl.uniform_discr_frompartition(vol, dtype='float32')
gt = ascontiguousarray(gt, dtype='float32')

PSpace = (angles,
          odl.uniform_partition([-1] * 2, [1] * 2, [64] * 2))
PSpace = odl.tomo.Parallel3dEulerGeometry(*PSpace)

# Operators
Radon = odl.tomo.RayTransform(vol, PSpace)
data = Radon(gt)

with myManager(device='cpu', order='C', fType='float32', cType='complex64') as c:
    # Space settings:
    dim = 3
    device = 'GPU'
    ASpace = AtomSpace(dim, isotropic=False)
    vol = VolSpace(odl.uniform_discr([-1] * 3, [1] * 3, gt.shape))

    # Projection settings:
    PSpace = ProjSpace(angles,
                       odl.uniform_discr([-1] * 2, [1] * 2, [64] * 2))

    nAtoms = 50
    recon = ASpace.random(nAtoms, seed=1)
    c.set(recon.x[:], c.mul(recon.x, 1 / 3))
    c.set(recon.r, 10, (slice(None), slice(None, 3)))
    c.set(recon.r, 0, (slice(None), slice(3, None)))
    nAtoms = recon.I.shape[0]
    Radon = GaussTomo(ASpace, PSpace, device=device)
    view = GaussVolume(ASpace, vol, device=device)
    gt = VolElement(vol, gt)
    R = Radon(recon)
    data = ProjElement(PSpace, data.asarray().reshape(PSpace.shape))

    # Reconstruction:
    fidelity = l2_squared_loss(dim)
    reg = null(dim)

    def guess(d, a): return doKL_ProjGDStep_iso(d, a, 1e-0, Radon)
    guess = None

    from NewtonGaussian import linesearch as GD
    GD(recon, data, [100, 1, 100], fidelity, reg, Radon, view,
       gt=gt, guess=guess, RECORD=RECORD, tol=1e-6, min_iter=10)
