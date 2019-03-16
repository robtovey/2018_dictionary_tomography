'''
Created on 18 Jun 2018

@author: Rob Tovey
'''
from KL_GaussRadon import doKL_ProjGDStep_iso
from os.path import join
from GaussDictCode import standardGaussTomo
RECORD = join('store', 'mesh_rand30_radius')
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

#####
    # Initiate Data:
    #####
    # # These lines initiate the 2 atom demo, seed=200 is a good one
#     gt = AtomElement(ASpace, x=[[-.5, .5, 0], [.5, -.5, 0]], r=[3, 1], I=1)
#     gt = AtomElement(ASpace, x=[[0.64, 0.78, -0.34],
#                                 [0.29, -0.82, -0.78]], r=[[10, 10, 10, 0, 0, 0], [10, 7, 5, 0, 0, 0]], I=[2, 1])
#     gt = AtomElement(ASpace, x=[[0, 0, 0]], r=[
#                      [10, 10, 10, 0, 0, 0]], I=[1])
# 
    from numpy import meshgrid, linspace, concatenate
    x = concatenate([x.reshape(-1, 1) for x in
                     meshgrid(*([linspace(-.8, .8, 3), ] * 3))], axis=1)
    gt = AtomElement(ASpace, x, 2.5, 1)
    # # These lines generate random atoms
    nAtoms = 27
#     gt = ASpace.random(nAtoms, seed=6)  # 0,1,3,6,8
    recon = ASpace.random(nAtoms, seed=1)
    c.set(recon.r, 1.5, (slice(None), slice(None, 3)))
    c.set(recon.r, 0, (slice(None), slice(3, None)))
    c.set(recon.I[:], 0.01)
    #####
    nAtoms = recon.I.shape[0]
    gt_sino = Radon(gt)
    gt_view = Radon.discretise(gt)
    R = Radon(recon)

#     from numpy import random
#     random.seed(0)
#     gt_sino.data += .001**.5 * random.randn(*gt_sino.shape)
#     gt_sino.data += .01**.5 * random.randn(*gt_sino.shape)
#     gt_sino.data += .05**.5 * random.randn(*gt_sino.shape)

#     from matplotlib import pyplot as plt
#     from GD_lib import _get3DvolPlot
#     _get3DvolPlot(None, Radon.discretise(gt).asarray(), (15, 25), 0.03)
#     plt.title('Ground Truth', {'fontsize': 26})
# #     plt.savefig('store/mesh_gt.eps', format='eps', dpi=600)
#     plt.show()
#     exit()

#     #####
#     from GaussDictCode.atomFuncs import test_grad
#     test_grad(ASpace, Radon, [10**-(k + 1) for k in range(6)])
#     exit()
#     #####

#     def guess(d, a): return doKL_ProjGDStep_iso(d, a, 1e-0, Radon)
#     def guess(d, a): return doKL_LagrangeStep_2Diso(d, a, 1e-3, Radon)
#     def guess(d, a): return a
    guess = None

    GD(recon, gt_sino, [100, 1, 100], fidelity, reg, Radon,
       gt=gt_view, guess=guess, RECORD=RECORD, tol=1e-6, min_iter=100,
       angles=((15, 25), (15, 115)), thresh=.03)

#     from GD_lib import _get3DvolPlot
#     _get3DvolPlot(None, Radon.discretise(recon).asarray(), (15, 25), 0.03)
#     plt.title('Reconstruction with Noise', {'fontsize': 26})
#     plt.savefig('store/mesh_noise.eps', format='eps', dpi=600)
#     plt.show()
#     exit()

#     from Fourier_Transform import GaussFT, GaussFTVolume
#     gFT = GaussFT(ASpace)
#     dFT = GaussFT(PSpace)
#     FT = GaussFTVolume(ASpace, PSpace)
#
#     def vview(a): return Radon.discretise(gFT.inverse(a))
#     GD(gFT(recon), dFT(gt_sino), [100, 1, 100], fidelity, reg, FT, view=vview,
#        gt=gt_view, guess=guess, RECORD=RECORD, tol=1e-6, min_iter=10,
#        myderivs=FT.derivs, angles=((15, 25), (15, 115)), thresh=.03)
