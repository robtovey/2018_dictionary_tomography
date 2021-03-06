'''
Created on 17 May 2018

@author: Rob Tovey
'''
from GaussDictCode.transport_loss import l2_squared_loss, Transport_loss
from KL_GaussRadon import doKL_LagrangeStep_iso, doKL_ProjGDStep_iso
from GD_lib import linesearch as GD
import odl
from GaussDictCode.dictionary_def import VolSpace, ProjSpace, AtomSpace, AtomElement
from GaussDictCode.atomFuncs import GaussTomo
from numpy import sqrt, pi
from GaussDictCode.bin.manager import myManager
from GaussDictCode.regularisation import Joubert, null

RECORD = 'multi_aniso_atoms_2D'
RECORD = None

with myManager(device='cpu', order='C', fType='float32', cType='complex64') as c:
    # Space settings:
    dim = 2
    device = 'GPU'  # CPU or GPU
    ASpace = AtomSpace(dim, isotropic=True)
    vol = VolSpace(odl.uniform_discr([-1] * 2, [1] * 2, [128] * 2))

    # Projection settings:
    PSpace = ProjSpace(odl.uniform_discr(0, pi, 30),
                       odl.uniform_discr(-1.5 * sqrt(dim),
                                         1.5 * sqrt(dim), 128))

    # Initiate Data:
    #####
    # #   These lines initiate the 2 atom demo
#     gt = AtomElement(ASpace, x=[[-.5, 0], [.5, 0]], r=[.30, .10], I=1)
#     recon = AtomElement(ASpace, [[.2, .5], [-.2, -.5]], [.18, .18], 1)
#     recon = ASpace.random(10, seed=1)
    # #   These lines generate random atoms
    nAtoms = 5  # 5
    gt = ASpace.random(nAtoms, seed=2)  # 6, 10
    recon = ASpace.random(nAtoms)
    c.set(gt.r, .1, (slice(None), slice(None, 2)))
    c.set(recon.r, .1, (slice(None), slice(None, 2)))
#     c.set(recon.r, 0, (slice(None), slice(2, None)))
#     c.set(recon.I[:], 1)
#     c.set(gt.I[:], 1)
#     recon[nAtoms - 1] = gt[nAtoms - 1]
    #####
    nAtoms = recon.I.shape[0]
    Radon = GaussTomo(ASpace, vol, PSpace, device=device)
    data = Radon(gt)
    gt_view = Radon.discretise(gt)
    R = Radon(recon)

    # Reconstruction:
    fidelity = l2_squared_loss(dim)
#     reg = Joubert(dim, 1e-2, 3e-1, (1e-1**dim, 1e-3))
    reg = null(dim)

    def guess(d, a): return doKL_ProjGDStep_iso(d, a, 1e-0, Radon)
#     def guess(d, a): return doKL_LagrangeStep_iso(d, a, 1e-3, Radon)
#     def guess(d, a): return a

    GD(recon, data, [100, 1], fidelity, reg, Radon,
       gt=gt_view, dim='xrI', guess=guess, RECORD=RECORD)
