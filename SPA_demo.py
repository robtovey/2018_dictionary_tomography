'''
Created on 11 Mar 2019

@author: Rob Tovey
'''
dim = 2
from GaussDictCode import standardSingleParticleGaussTomo, standardGaussTomo
from os.path import join
RECORD = join('store', 'mesh_rand30_radius')
RECORD = None
from GaussDictCode.dictionary_def import AtomElement
from numpy import pi, prod
from GaussDictCode.bin.manager import myManager

with myManager(device='cpu', order='C', fType='float32', cType='complex64') as c:
    a_range = (0, pi) if dim == 2 else ([0] * 2, [pi] * 2)
    a_num = (10,) if dim == 2 else (10, 10)
    Radon, fidelity, _, ASpace, PSpace, params = standardSingleParticleGaussTomo(
        dim=dim, device='GPU', isotropic=False,
        angle_range=a_range, angle_num=a_num,
        vol_box=[-1, 1], vol_size=32, det_box=[-1.4, 1.4], det_size=128,
        fidelity='l2_squared', reg=None,
        solver='Newton'
    )
    scale = 1, 1
#     Radon, fidelity, _, ASpace, PSpace, params = standardGaussTomo(
#         dim=dim, device='GPU', isotropic=False,
#         angle_range=a_range, angle_num=a_num,
#         vol_box=[-1, 1], vol_size=32, det_box=[-1.4, 1.4], det_size=128,
#         fidelity='l2_squared', reg=None,
#         solver='Newton'
#     )
#     # Scale is so pointspread(x) = scale*pointwise(x)
#     scale = Radon.range.volume / prod(Radon.range.shape) , Radon.embedding.cell_volume
    reg, GD = params

#####
    # Initiate Data:
    #####
    if dim == 2:
        gt = AtomElement(ASpace, x=[[-.5, 0], [.5, 0]], r=[3, 1], I=1)
    else:
        gt = AtomElement(ASpace, x=[[-.5, .5, 0], [.5, -.5, 0]], r=[3, 1], I=1)

    recon = ASpace.random(10, seed=1)
    c.set(recon.r, 1, (slice(None), slice(None, dim)))
    c.set(recon.r, 0, (slice(None), slice(dim, None)))
    c.set(recon.I[:], .01)
        
    # # These lines generate random atoms
    nAtoms = 10
#     gt = ASpace.random(nAtoms, seed=6)  # 0,1,3,6,8
    recon = ASpace.random(nAtoms, seed=1)
    c.set(recon.r, 1.5, (slice(None), slice(None, dim)))
    c.set(recon.r, 0, (slice(None), slice(dim, None)))
    c.set(recon.I[:], 0.01)
    #####
    nAtoms = recon.I.shape[0]
    gt_sino = Radon(gt)
    gt_view = Radon.discretise(gt)
    R = Radon(recon)
    
#     from numpy import set_printoptions; set_printoptions(1, suppress=False, sign='+')
#     tmp = (Radon(recon[0]).asarray() ** 2).sum() / 2, Radon.L2_derivs(recon[0], 0 * Radon(recon[0]), order=2)
#     print()
#     print(gt_view.asarray().sum() * scale[1])
#     print(tmp[0] * scale[0] ** 2)
#     print(tmp[1][0] * scale[0] ** 2)
#     print((tmp[1][1] / abs(tmp[1][1]).max()).round(1))
#     print()
#     print((tmp[1][2] / abs(tmp[1][2]).max()).round(1))
#     exit()
 
#     tmp = Radon.L2_derivs(recon[0], .01 * gt_sino, order=2)
#     print()
#     print(tmp[0] * scale[0] ** 2)
#     print(tmp[1] * scale[0] ** 2)
#     print()
#     print(tmp[2] * scale[0] ** 2)
#     print()
#     print(tmp[1] / abs(tmp[1]))
#     print()
#     print(tmp[2] / abs(tmp[2]))
#     exit()

#     from numpy import random
#     random.seed(0)
#     gt_sino.data += .001**.5 * random.randn(*gt_sino.shape)
#     gt_sino.data += .01**.5 * random.randn(*gt_sino.shape)
#     gt_sino.data += .05**.5 * random.randn(*gt_sino.shape)

#     from matplotlib import pyplot as plt
# #     from GD_lib import _get3DvolPlot
#     plt.imshow(gt_sino.asarray(), aspect='auto')
# #     tmp = Radon.discretise(gt).asarray()
# #     _get3DvolPlot(None, Radon.discretise(gt).asarray(), (15, 25), 0.03 * scale[1])
# #     plt.title('Ground Truth', {'fontsize': 26})
# # #     plt.savefig('store/mesh_gt.eps', format='eps', dpi=600)
#     plt.show()
#     exit()

#     #####
#     from GaussDictCode.atomFuncs import test_grad
#     test_grad(ASpace, Radon, [10**-(k + 1) for k in range(6)])
#     exit()
#     #####

    GD(recon, gt_sino, [50, 1, 50], fidelity, reg, Radon,
       gt=gt_view, RECORD=RECORD, tol=1e-6, min_iter=100,
       angles=((15, 25), (15, 115)), thresh=.03 * scale[1])

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
