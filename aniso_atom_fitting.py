'''
Created on 15 Feb 2018

@author: Rob Tovey
'''
from code.transport_loss import l2_squared_loss, Transport_loss
from code import standardGaussTomo
RECORD = 'multi_aniso_atoms_2D'
RECORD = None
import odl
from code.dictionary_def import VolSpace, ProjSpace, AtomSpace, AtomElement
from code.atomFuncs import GaussTomo, GaussVolume
from numpy import sqrt, pi
from code.bin.manager import myManager
from code.regularisation import Joubert, null

with myManager(device='cpu', order='C', fType='float32', cType='complex64') as c:
    Radon, view, fidelity, _, ASpace, PSpace, params = standardGaussTomo(
        dim=2, device='GPU', isotropic=False,
        angle_range=(0, pi), angle_num=30,
        vol_box=[-1, 1], vol_size=64, det_box=[-1.5, 1.5], det_size=128,
        fidelity='l2_squared', reg=None,
        solver='Newton'
    )
    reg, GD = params
    vol = view.ProjectionSpace

    # Initiate Data:
    #####
    # #   These lines initiate the 2 atom demo
    gt = AtomElement(ASpace, x=[[-.5, 0], [.5, 0]], r=[3, 1], I=1)
    recon = AtomElement(ASpace, [[.2, .5], [-.2, -.5]], [1.8, 1.8], 1)
    recon = ASpace.random(10, seed=1)
    # #   These lines generate random atoms
    # nAtoms = 5
    # gt = ASpace.random(nAtoms, seed=2)  # 6, 10
    # gt.x *= .8
    # recon = ASpace.random(nAtoms)
    # recon.x *= .8
    c.set(recon.r, 1, (slice(None), slice(None, 2)))
    c.set(recon.r, 0, (slice(None), slice(2, None)))
    c.set(recon.I[:], .01)
    #####
    nAtoms = recon.I.shape[0]
    gt_sino = Radon(gt)
    gt_view = view(gt)
    R = Radon(recon)

#     from atomFuncs import test_grad
#     test_grad(ASpace, Radon, [10**-(k + 0) for k in range(6)])
#     exit()
#     gt_sino.plot(plt.subplot(121))
#     gt_view.plot(plt.subplot(122))
#     plt.show()
#     exit()

#     def guess(d, a): return doKL_ProjGDStep_2Diso(d, a, 1e-0, Radon)
#     def guess(d, a): return doKL_LagrangeStep_2Diso(d, a, 1e-3, Radon)
#     def guess(d, a): return a
    guess = None

    GD(recon, gt_sino, [100, 1], fidelity, reg, Radon, view,
       gt=gt_view, guess=guess, RECORD=RECORD)
print('Reconstruction complete')
