'''
Created on 20 Feb 2018

@author: Rob Tovey
'''
from KL_GaussRadon import doKL_ProjGDStep_iso
from GaussDictCode import standardGaussTomo
RECORD = 'store/5_atoms_3D_OT_+I'
RECORD = None
from GaussDictCode.dictionary_def import AtomElement
from numpy import sqrt, pi
from GaussDictCode.bin.manager import myManager

with myManager(device='cpu', order='C', fType='float32', cType='complex64') as c:
    Radon, fidelity, _, ASpace, PSpace, params = standardGaussTomo(
        dim=3, device='GPU', isotropic=False,
        angle_range=(0, pi), angle_num=50,
        vol_box=[-1, 1], vol_size=32, det_box=[-1.4, 1.4], det_size=64,
        fidelity='l2_squared', reg=None,
        solver='Newton'
    )
    reg, GD = params
    vol = Radon.embedding

    # Initiate Data:
    #####
    # # These lines initiate the 2 atom demo, seed=200 is a good one
#     gt = AtomElement(ASpace, x=[[-.5, 0, 0], [.5, 0, 0]], r=[30, 10], I=1)
    gt = AtomElement(ASpace, x=[[0.24, 0.48, 0],
                                [-0.2, -0.52, -0.1]], r=[[1, 1, 1, 0, 0, 0], [1, .7, .5, 0, 0, 0]], I=[2, 1])
    nAtoms = 3
#     gt = ASpace.random(nAtoms, seed=6)  # 0,1,3,6,8
    recon = ASpace.random(nAtoms)
#     c.set(recon.r, 1.5, (slice(None), slice(None, 3)))
#     c.set(recon.r, 0, (slice(None), slice(3, None)))
    c.set(recon.I[:], .01)
#     c.set(gt.I[:], 1)
    #####
    nAtoms = recon.r.shape[0]
    gt_sino = Radon(gt)
    gt_view = Radon.discretise(gt)
    R = Radon(recon)

#     from matplotlib import pyplot as plt
#     for i in range(gt_sino.shape[0]):
#         plt.gca().clear()
#         gt_sino.plot(plt, Slice=[i])
#         plt.title(str(i))
#         plt.pause(.1)
#     exit()
#     from GaussDictCode.atomFuncs import test_grad
#     test_grad(ASpace, Radon, [10**-(k + 1) for k in range(6)])
#     exit()
#     gt_sino.plot(plt.subplot(121))
#     gt_view.plot(plt.subplot(122))
#     plt.show()
#     exit()

    def guess(d, a): return doKL_ProjGDStep_iso(d, a, 1e-0, Radon)

#     def guess(d, a): return doKL_LagrangeStep_2Diso(d, a, 1e-3, Radon)
#     def guess(d, a): return a
    guess = None

    GD(recon, gt_sino, [100, 1, 1], fidelity, reg, Radon,
       gt=gt_view, guess=guess, RECORD=RECORD, tol=1e-6, min_iter=30)
