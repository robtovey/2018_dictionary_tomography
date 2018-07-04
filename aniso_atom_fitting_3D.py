'''
Created on 20 Feb 2018

@author: Rob Tovey
'''
from code.transport_loss import l2_squared_loss, Transport_loss
from GD_lib import linesearch as GD
from KL_GaussRadon import doKL_ProjGDStep_iso
RECORD = 'store/5_atoms_3D_OT_+I'
RECORD = None
import odl
from code.bin.dictionary_def import VolSpace, ProjSpace, AtomSpace, AtomElement
from code.bin.atomFuncs import GaussTomo, GaussVolume
from numpy import sqrt, pi
from code.bin.manager import myManager
from code.regularisation import Joubert, null

with myManager(device='cpu', order='C', fType='float32', cType='complex64') as c:
    # Space settings:
    dim = 3
    device = 'GPU'  # CPU or GPU
    ASpace = AtomSpace(dim, isotropic=False)
    vol = odl.uniform_discr([-1] * 3, [1] * 3, [64] * 3)
    box = [vol.min_pt, vol.max_pt]
    vol = VolSpace(vol)

    # Projection settings:
#     PSpace = ProjSpace(odl.uniform_discr([0] * 2, [pi, pi / 2], [20, 10]),
#                        odl.uniform_discr([-sqrt(dim)] * 2,
#                                          [sqrt(dim)] * 2, [64] * 2))
    PSpace = ProjSpace(odl.uniform_discr([0], [pi], [50]),
                       odl.uniform_discr([-sqrt(dim)] * 2,
                                         [sqrt(dim)] * 2, [64] * 2))

    # Initiate Data:
    #####
    # # These lines initiate the 2 atom demo, seed=200 is a good one
#     gt = AtomElement(ASpace, x=[[-.5, 0, 0], [.5, 0, 0]], r=[30, 10], I=1)
    gt = AtomElement(ASpace, x=[[0.64, 0.78, -0.34],
                                [0.29, -0.82, -0.78]], r=[[10, 10, 10, 0, 0, 0], [10, 7, 5, 0, 0, 0]], I=[2, 1])
#     recon = AtomElement(ASpace, [[.2, .5, 0], [-.2, -.5, 0]], [30, 10], 1)
    # # These lines generate random atoms
    nAtoms = 1
#     gt = ASpace.random(nAtoms, seed=6)  # 0,1,3,6,8
    recon = ASpace.random(nAtoms)
#     c.set(recon.r, 10, (slice(None), slice(None, 3)))
#     c.set(recon.r, 0, (slice(None), slice(3, None)))
#     c.set(recon.I[:], 1)
#     c.set(gt.I[:], 1)
    #####
    nAtoms = recon.r.shape[0]
    Radon = GaussTomo(ASpace, PSpace, device=device)
    view = GaussVolume(ASpace, vol, device=device)
    gt_sino = Radon(gt)
    gt_view = view(gt)
    R = Radon(recon)

#     from matplotlib import pyplot as plt
#     for i in range(gt_sino.shape[0]):
#         plt.gca().clear()
#         gt_sino.plot(plt, Slice=[i])
#         plt.title(str(i))
#         plt.pause(.1)
#     exit()
#     from code.bin.atomFuncs import test_grad
#     test_grad(ASpace, Radon, [10**-(k + 1) for k in range(6)])
#     exit()
#     gt_sino.plot(plt.subplot(121))
#     gt_view.plot(plt.subplot(122))
#     plt.show()
#     exit()

    # Reconstruction:
    fidelity = l2_squared_loss(dim)
#     fidelity = Transport_loss(dim, device=device)
#     reg = Joubert(dim, 1e-1, L3e-2, (1e1**dim, 1e-3))
#     fidelity = Transport_loss(dim, device=device)
#     reg = Joubert(dim, 0 * 1e2, 1e2, (1e1**dim, 1e-1))
    reg = null(dim)

    def guess(d, a): return doKL_ProjGDStep_iso(d, a, 1e-0, Radon)
#     def guess(d, a): return doKL_LagrangeStep_2Diso(d, a, 1e-3, Radon)
#     def guess(d, a): return a
#     guess = None

    GD(recon, gt_sino, [200, 1, 100], fidelity, reg, Radon, view,
       gt=gt_view, dim='xrI', guess=guess, RECORD=RECORD, tol=1e-6, min_iter=30)
