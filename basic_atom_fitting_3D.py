'''
Created on 8 Jan 2018

@author: Rob Tovey
'''
from code import standardGaussTomo
RECORD = 'store/2_atoms_3D_L2'
RECORD = None
import odl
from code.dictionary_def import AtomElement
from numpy import sqrt, pi
from code.bin.manager import myManager

with myManager(device='cpu', order='C', fType='float32', cType='complex64') as c:
    Radon, view, fidelity, _, ASpace, PSpace, params = standardGaussTomo(
        dim=3, device='GPU', isotropic=False,
        angle_range=([0] * 2, [pi, pi / 2]), angle_num=[20, 1],
        vol_box=[-1, 1], vol_size=32, det_box=[-1.4, 1.4], det_size=32,
        fidelity='l2_squared', reg=None,
        solver='Newton'
    )
    reg, GD = params
    vol = view.ProjectionSpace

    # Initiate Data:
    #####
    # # These lines initiate the 2 atom demo, seed=200 is a good one
    gt = AtomElement(
        ASpace, x=[[-.5, 0, 0], [.5, 0, 0]], r=[1, 3], I=1)
    recon = AtomElement(
        ASpace, [[.2, .5, 0], [-.2, -.5, 0]], [1, 1], .1)
    # # These lines generate random atoms
#     nAtoms = 10
#     gt = ASpace.random(nAtoms, seed=5)  # 5
#     recon = ASpace.random(nAtoms)
#     recon.I, recon.r = gt.I, gt.r
    #####
    nAtoms = gt.r.shape[0]
    gt_sino = Radon(gt)
    gt_view = view(gt)
    R = Radon(recon)

#     from code.atomFuncs import test_grad
#     test_grad(ASpace, Radon, [10**-(k + 1) for k in range(6)])
#     exit()

#     def guess(d, a): return doKL_ProjGDStep_iso(d, a, 1e-0, Radon)
#     def guess(d, a): return doKL_LagrangeStep_2Diso(d, a, 1e-3, Radon)
#     def guess(d, a): return a

    GD(recon, gt_sino, [100, 1], fidelity, reg, Radon, view,
       gt=gt_view, guess=None, RECORD=RECORD)
