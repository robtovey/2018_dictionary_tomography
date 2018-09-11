'''
Created on 18 Jun 2018

@author: Rob Tovey
'''
from KL_GaussRadon import doKL_ProjGDStep_iso
from os.path import join
from Fourier_Transform import doLoc_L2Step
from code import standardGaussTomo
RECORD = join('store', 'mesh_rand30_gd')
# RECORD = None
from code.bin.dictionary_def import AtomElement
from numpy import sqrt, pi
from code.bin.manager import myManager


with myManager(device='cpu', order='C', fType='float32', cType='complex64') as c:
    Radon, view, fidelity, _, ASpace, PSpace, params = standardGaussTomo(
        dim=3, device='GPU', isotropic=False,
        angle_range=(0, pi), angle_num=50,
        vol_box=[-1, 1], vol_size=32, det_box=[-1.4, 1.4], det_size=64,
        fidelity='l2_squared', reg=None, solver='other'
    )
    reg, GD = params
    vol = view.ProjectionSpace

    # Initiate Data:
    #####
    # # These lines initiate the 2 atom demo, seed=200 is a good one
#     gt = AtomElement(ASpace, x=[[-.5, 0, 0], [.5, 0, 0]], r=[30, 10], I=1)
#     gt = AtomElement(ASpace, x=[[0.64, 0.78, -0.34],
#                                 [0.29, -0.82, -0.78]], r=[[10, 10, 10, 0, 0, 0], [10, 7, 5, 0, 0, 0]], I=[2, 1])
#     gt = AtomElement(ASpace, x=[[0, 0, 0]], r=[
#                      [10, 10, 10, 0, 0, 0]], I=[1])

    from numpy import meshgrid, linspace, concatenate
    x = concatenate([x.reshape(-1, 1) for x in
                     meshgrid(*([linspace(-.8, .8, 3), ] * 3))], axis=1)
    gt = AtomElement(ASpace, x, 20, 1)
#     recon = AtomElement(ASpace, [[.2, .5, 0], [-.2, -.5, 0]], [30, 10], 1)
    # # These lines generate random atoms
    nAtoms = 30
#     gt = ASpace.random(nAtoms, seed=6)  # 0,1,3,6,8
    recon = ASpace.random(nAtoms, seed=1)
    c.set(recon.r, 5, (slice(None), slice(None, 3)))
    c.set(recon.r, 0, (slice(None), slice(3, None)))
    c.set(recon.I[:], 0.01)
#     c.set(gt.I[:], 1)
    #####
    nAtoms = recon.I.shape[0]
    gt_sino = Radon(gt)
    gt_view = view(gt)
    R = Radon(recon)

#     from GD_lib import _get3DvolPlot
#     from matplotlib import pyplot as plt
#     _get3DvolPlot(None, gt_view.asarray(), (15, 25), 0.03)
#     plt.title('GT mesh')
#     plt.show()
#     exit()
#     while True:
#         for i in range(gt_sino.shape[0]):
#             plt.gca().clear()
#             gt_sino.plot(plt, Slice=[i])
#             plt.title(str(i))
#             plt.pause(.1)
#     exit()


#     #####
#     from code.bin.atomFuncs import test_grad
#     test_grad(ASpace, Radon, [10**-(k + 1) for k in range(6)])
#     exit()
#     #####

#     def guess(d, a): return doKL_ProjGDStep_iso(d, a, 1e-0, Radon)
#     def guess(d, a): return doKL_LagrangeStep_2Diso(d, a, 1e-3, Radon)
#     def guess(d, a): return a
    guess = None

    GD(recon, gt_sino, [200, 1, 100], fidelity, reg, Radon, view,
       gt=gt_view, guess=guess, RECORD=RECORD, tol=1e-6, min_iter=100,
       angles=((15, 25), (15, 115)), thresh=.03)

#     from Fourier_Transform import GaussFT, GaussFTVolume
#     gFT = GaussFT(ASpace)
#     dFT = GaussFT(PSpace)
#     FT = GaussFTVolume(ASpace, PSpace)
#
#     def vview(a): return view(gFT.inverse(a))
#     GD(gFT(recon), dFT(gt_sino), [100, 1, 100], fidelity, reg, FT, vview,
#        gt=gt_view, guess=guess, RECORD=RECORD, tol=1e-6, min_iter=10,
#        myderivs=FT.derivs, angles=((15, 25), (15, 115)), thresh=.03)
