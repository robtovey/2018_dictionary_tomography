'''
Created on 5 Jan 2018

@author: Rob Tovey
'''
from code.bin.manager import myManager
from code import standardGaussTomo
RECORD = '2_atoms_2D_OT'
RECORD = None
from code.dictionary_def import AtomElement
from numpy import sqrt, pi

with myManager(device='cpu', order='C', fType='float32', cType='complex64') as c:
    Radon, view, fidelity, _, ASpace, PSpace, params = standardGaussTomo(
        dim=2, device='GPU', isotropic=False,
        angle_range=(0, pi), angle_num=30,
        vol_box=[-1, 1], vol_size=64, det_box=[-1.4, 1.4], det_size=128,
        fidelity='l2_squared', reg=None,
        solver='Newton'
    )
    reg, GD = params
    vol = view.ProjectionSpace

    # Initiate Data:
    #####
    # #   These lines initiate the 2 atom demo
    gt = AtomElement(ASpace, x=[[-.5, 0], [.5, 0]], r=[1, 3], I=1)
    recon = AtomElement(ASpace, [[.2, .5], [-.2, -.5]], [2, 2], .01)
    # #   These lines generate random atoms
    # nAtoms = 5
    # gt = ASpace.random(nAtoms, seed=6)  # 6, 10
    # recon = ASpace.random(nAtoms)
    # recon.I, recon.r = gt.I, gt.r
    #####
    nAtoms = len(gt.r)
    gt_sino = Radon(gt)
    gt_view = view(gt)
    R = Radon(recon)

    # from atomFuncs import test_grad
    # test_grad(ASpace, Radon, [10**-(k + 0) for k in range(6)])
    # exit()

    GD(recon, gt_sino, [100, 1], fidelity, reg, Radon, view,
       gt=gt_view, guess=None, RECORD=RECORD, tol=1e-4, min_iter=20)
