'''
Created on 8 Jan 2018

@author: Rob Tovey
'''
from KL_GaussRadon import doKL_ProjGDStep_iso
RECORD = 'store/2_atoms_3D_L2'
RECORD = None
import odl
from GD_lib import linesearch as GD
from code.bin.dictionary_def import VolSpace, ProjSpace, AtomSpace, AtomElement
from code.bin.atomFuncs import GaussTomo, GaussVolume
from code.transport_loss import l2_squared_loss, Transport_loss
from numpy import sqrt, pi
from code.bin.manager import myManager
from code.regularisation import Joubert, null

with myManager(device='cpu', order='C', fType='float32', cType='complex64') as c:
    # Space settings:
    dim = 3
    device = 'GPU'  # CPU or GPU
    ASpace = AtomSpace(dim)
    vol = VolSpace(odl.uniform_discr([-1] * 3, [1] * 3, [64] * 3))

    # Projection settings:
    PSpace = ProjSpace(odl.uniform_discr([0] * 2, [pi, pi / 2], [20, 1]),
                       odl.uniform_discr([-sqrt(dim)] * 2,
                                         [sqrt(dim)] * 2, [8] * 2))
#     PSpace = ProjSpace(odl.uniform_discr([0], [pi], [50]),
#                        odl.uniform_discr([-sqrt(dim)] * 2,
#                                          [sqrt(dim)] * 2, [64] * 2))

    # Initiate Data:
    #####
    # # These lines initiate the 2 atom demo, seed=200 is a good one
    gt = AtomElement(
        ASpace, x=[[-.5, 0, 0], [.5, 0, 0]], r=[1 / 30, 1 / 10], I=1)
    recon = AtomElement(
        ASpace, [[.2, .5, 0], [-.2, -.5, 0]], [1 / 30, 1 / 10], 1)
    # # These lines generate random atoms
#     nAtoms = 10
#     gt = ASpace.random(nAtoms, seed=5)  # 5
#     recon = ASpace.random(nAtoms)
#     recon.I, recon.r = gt.I, gt.r
    #####
    nAtoms = gt.r.shape[0]
    Radon = GaussTomo(ASpace, PSpace, device=device)
    view = GaussVolume(ASpace, vol, device=device)
    gt_sino = Radon(gt)
    gt_view = view(gt)
    R = Radon(recon)

#     from code.bin.atomFuncs import test_grad
#     test_grad(ASpace, Radon, [10**-(k + 1) for k in range(6)])
#     exit()

    # Reconstruction:
    fidelity = l2_squared_loss(dim)
    reg = Joubert(dim, 1e2, 1e-1, (.5, 1e2))
    # fidelity = Transport_loss(dim, device=device)
#     reg = Joubert(dim, 1e2, 1e+2, (.5, 1e2))
    reg = null(dim)

    def guess(d, a): return doKL_ProjGDStep_iso(d, a, 1e-0, Radon)
#     def guess(d, a): return doKL_LagrangeStep_2Diso(d, a, 1e-3, Radon)
#     def guess(d, a): return a

    GD(recon, gt_sino, [100, 1], fidelity, reg, Radon, view,
       gt=gt_view, dim='xrI', guess=guess, RECORD=RECORD)
