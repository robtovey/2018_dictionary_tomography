'''
Created on 28 Jun 2018

@author: Rob Tovey
'''
from os.path import join
from code.bin.manager import myManager
from code.dictionary_def import VolElement, AtomElement
from KL_GaussRadon import doKL_ProjGDStep_iso
from code import standardGaussTomo
RECORD = join('store', 'polyII_rand200_Evren')
RECORD = None
from odl.contrib import mrc
from numpy import sqrt, ascontiguousarray, pi, pad
from scipy.io import savemat, loadmat

# Import data:
with mrc.FileReaderMRC(join('PolyII', 'rna_phantom.mrc')) as f:
    f.read_header()
    gt = f.read_data()
    gt -= gt.min()
gt = pad(gt, ((2, 3), (0, 0), (10, 10)), 'constant')
gt = ascontiguousarray(gt[30:70, 30:70, 30:70], dtype='float32')

# from GD_lib import _get3DvolPlot, __makeVid
# smallVid = None if 0 else join('store', 'polyII_TVvsGauss')
# if smallVid is not None:
#     __makeVid()
# from matplotlib import pyplot as plt
# x = loadmat(join('store', 'polyII_recon'))['view']
# y = loadmat(join('store', 'polyII_TV'))['view']
# clim = [0, gt.max()]
# plt.figure(figsize=(20, 10))
# ax = plt.subplot(231), plt.subplot(232), plt.subplot(
#     233), plt.subplot(234), plt.subplot(236)
# if smallVid is not None:
#     writer = __makeVid(plt.gcf(), smallVid, stage=1, fps=30)
# i, j = 0, 0
# # _get3DvolPlot(None, gt, (15, 0), 1.7)
# # ax = plt.gca(),
# # ax[0].set_title('PolyII sample')
# while j < 2:
#     for a in ax:
#         a.clear()
#     ax[0].set_title('TV recon')
#     ax[1].set_title('GT')
#     ax[2].set_title('Gauss recon')
#     ax[3].set_title('TV residual')
#     ax[4].set_title('Gauss residual')
#
#     ax[0].imshow(y[:, :, i], clim=clim)
#     ax[1].imshow(gt[:, :, i], clim=clim)
#     ax[2].imshow(x[:, :, i], clim=clim)
#     ax[3].imshow(abs(y[:, :, i] - gt[:, :, i]), clim=clim)
#     ax[4].imshow(abs(x[:, :, i] - gt[:, :, i]), clim=clim)
#
#     i += 1
#     if i == gt.shape[2]:
#         j += 1
#         i -= gt.shape[2]
#
# #     ax[0].view_init(15, i)
# #     i += 2.5
# #     print(i)
# #     if i >= 360:
# #         break
#
#     plt.draw()
#     if smallVid is None:
#         plt.show(block=False)
#         plt.pause(.2)
#     else:
#         __makeVid(writer, plt, stage=2)
# if smallVid is not None:
#     __makeVid(writer, plt, stage=3)
#
# # _get3DvolPlot(None, gt, (280, 90), 1.7)
# # ax = plt.gca()
# # i = 0
# # for j in range(720):
# #     ax.view_init(i * 10, 0)
# #     i = (i + 1) % 36
# #     ax.set_title('Rotation ' + str(i * 10))
# #     print(j)
# #     plt.pause(.01)
# #     plt.show(block=False)
# # plt.show()
# exit()

with myManager(device='cpu', order='C', fType='float32', cType='complex64') as c:

    Radon, view, fidelity, data, ASpace, PSpace, params = standardGaussTomo(
        gt=gt, dim=3, device='GPU', isotropic=False,
        angle_range=([-pi / 3, 0], [pi / 3, pi / 2]), angle_num=[61, 1],
        vol_box=[-1, 1], det_box=[-sqrt(3), sqrt(3)], noise=0 * 0.1,
        fidelity='l2_squared', reg=None, solver='Newton'
    )
#     Radon, view, fidelity, data, ASpace, PSpace, params = standardGaussTomo(
#         gt=gt, dim=3, device='GPU', isotropic=False,
#         angle_range=([-pi / 3, 0], [pi / 3, pi / 2]), angle_num=[61 * 4, 1],
#         vol_box=[-1, 1], det_box=[-sqrt(3), sqrt(3)], det_size=int(40 / 2),
#         fidelity='l2_squared', reg=None, solver='Newton'
#     )
    reg, GD = params
#     recon = loadmat(join('store', 'polyII_recon'))
#     x, recon = recon['view'], AtomElement(
#         ASpace, recon['X'], recon['R'], recon['I'])
#     print('GT RMSE = %f, data RMSE = %f' %
#           ((abs(gt - x)**2).sum() / gt.size, (abs((data - Radon(recon)).asarray())**2).sum() / data.asarray().size))
#     exit()

    nAtoms = 20
    recon = ASpace.random(nAtoms, seed=1)
    c.set(recon.x[:], c.mul(recon.x, 1 / 3))
    c.set(recon.r, 1, (slice(None), slice(None, 3)))
    c.set(recon.r, 0, (slice(None), slice(3, None)))
    c.set(recon.I, 0.01)
#     c.set(recon.r, 2, (slice(None), slice(None, 3)))
#     c.set(recon.I, 0.01)
    nAtoms = recon.I.shape[0]
    gt = VolElement(view.ProjectionSpace, gt)
    R = Radon(recon)

#     from matplotlib import pyplot as plt
#     while True:
#         for i in range(data.shape[0]):
#             plt.gca().clear()
#             data.plot(plt, Slice=[i])
#             plt.title(str(i))
#             plt.pause(.1)
#     exit()

#     def guess(d, a): return doKL_ProjGDStep_iso(d, a, 1e-0, Radon)
#     def guess(d, a): return a
    guess = None

    GD(recon, data, [100, 1, 100], fidelity, reg, Radon, view,
        gt=gt, guess=guess, RECORD=RECORD, tol=1e-6, min_iter=2000,
       thresh=1.5, angles=((20, 45), (100, 90)))

#     from Fourier_Transform import GaussFT, GaussFTVolume
#     gFT = GaussFT(ASpace)
#     dFT = GaussFT(PSpace)
#     FT = GaussFTVolume(ASpace, PSpace)
#
#     def vview(a): return view(gFT.inverse(a))
#     GD(gFT(recon), dFT(data), [100, 1, 100], fidelity, reg, FT, vview,
#        gt=gt, guess=guess, RECORD=RECORD, tol=1e-6, min_iter=10,
#        myderivs=FT.derivs, thresh=1.5, angles=((20, 45), (100, 90)))

    print('GT RMSE = %f, data RMSE = %f' % 
          ((abs((gt - view(recon)).asarray()) ** 2).sum() / gt.asarray().size, (abs((data - Radon(recon)).asarray()) ** 2).sum() / data.asarray().size))

#     savemat(join('store', 'polyII_recon'), {'view': view(recon).asarray(),
#                                             'X': c.asarray(recon.x),
#                                             'R': c.asarray(recon.r),
#                                             'I': c.asarray(recon.I), })

#     savemat(join('store', 'polyII_recons', 'Gauss_' + str(nAtoms)), {'view': view(recon).asarray(),
#                                                                      'X': c.asarray(recon.x),
#                                                                      'R': c.asarray(recon.r),
#                                                                      'I': c.asarray(recon.I), })

exit()


def r2V(r):
    V = zeros((r.shape[0], 3, 3))
    for i in range(3):
        V[:, i, i] = r[:, i]
    for i in range(2):
        V[:, i, i + 1] = r[:, i + 3]
    V[:, 0, 2] = r[:, 5]

    V = matmul(V.transpose(0, 2, 1), V)
    V = linalg.inv(V)

    return V


def getIPs(x, r):
    n = x.shape[0]
    V = r2V(r)
    detV = linalg.det(V)
    detV = sqrt(detV.reshape(1, n) * detV.reshape(n, 1))
    sumV = V.reshape(1, n, 3, 3) + V.reshape(n, 1, 3, 3)
    sumX = x.reshape(1, n, 3, 1) - x.reshape(n, 1, 3, 1)
    IPs = exp(-.5 * (matmul(linalg.inv(sumV), sumX) * sumX).sum(axis=(2, 3)))
    IPs *= sqrt(8 * detV / linalg.det(sumV))
    return IPs


recon = loadmat(join('store', 'polyII_recon'))
recon = loadmat(join('store', 'Bipyramid_recon_200'))
recon = AtomElement(ASpace, recon['X'], recon['R'], recon['I'])
from numpy import zeros, matmul, linalg, exp
from matplotlib import pyplot as plt
IPs = getIPs(recon.x, recon.r)
IPs[IPs > .999] = 0
plt.imshow(IPs)
plt.title('Orthogonality plot')
plt.colorbar()
plt.figure()
plt.plot(IPs.sum(1))
plt.show(block=True)
