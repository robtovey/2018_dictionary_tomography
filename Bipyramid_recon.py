'''
Created on 29 Mar 2018

@author: Rob Tovey
'''
from os.path import join
from KL_GaussRadon import doKL_ProjGDStep_iso
from GaussDictCode import standardGaussTomo
RECORD = join('store', 'Bipyramid_rand200_noise')
RECORD = None
from numpy import loadtxt, asarray, ascontiguousarray, pi, sqrt
from GaussDictCode.bin.manager import myManager
from PIL import Image
from scipy.io import savemat, loadmat

# x = loadmat(join('store', 'Bipyramid_recon_TV'))['view']
# y = loadmat(join('store', 'Bipyramid_recon'))['view']
# n = 1
# from GD_lib import _get3DvolPlot, __makeVid
# # _get3DvolPlot(None, y[::n, ::n, ::n], (-168, -158), .07)
# smallVid = None if False else join('store', 'Bipyramid_TVvsGauss')
# if smallVid is not None:
#     __makeVid()
# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# plt.figure(figsize=(20, 10))
# if smallVid is not None:
#     writer = __makeVid(plt.gcf(), smallVid, stage=1, fps=30)
# ax = (plt.gcf().add_subplot(121, projection='3d'),
#       plt.gcf().add_subplot(122, projection='3d'))
# _get3DvolPlot(ax[0], x[::n, ::n, ::n], (-168, -158), .07)
# _get3DvolPlot(ax[1], y[::n, ::n, ::n], (-168, -158), .07)
# ax[0].set_title('TV reconstruction')
# ax[1].set_title('Gaussian reconstruction')
# i, j = 0, 0
# while j < 2:
#     ax[0].view_init(-168, i)
#     ax[1].view_init(-168, i)
#     i += 2.5
#     print(i)
#     if i >= 360:
#         break
#
#     plt.draw()
#     if smallVid is None:
#         plt.show(block=False)
#         plt.pause(.2)
#     else:
#         __makeVid(writer, plt, stage=2)
# if smallVid is not None:
#     __makeVid(writer, plt, stage=3)
# # plt.show()
# exit()

with myManager(device='cpu', order='C', fType='float32', cType='complex64') as c:
    # Import data:
    angles = loadtxt(join('Materials', 'p10arot90_hdr0_5.rawtlt')) * pi / 180
    data = Image.open(join('Materials', 'p10arot90_hdr0_5.tif'))
    data = [asarray(data).T for i in range(
        data.n_frames) if data.seek(i) is None]
    # First dimension is angle, second is width, third is slice
    data = asarray(data)
    # Space settings:
    data = ascontiguousarray(
        data[2:-1, 100:-100:16, 100:-100:16], dtype='float32')
    angles = angles[2:-1]

#     from GD_lib import __makeVid
#     smallVid = None if True else join('store', 'Bipyramid_gt')
#     if smallVid is not None:
#         __makeVid()
#     from matplotlib import pyplot as plt
#     clim = [0, data.max()]
#     plt.figure(figsize=(10, 10))
#     if smallVid is not None:
#         writer = __makeVid(plt.gcf(), smallVid, stage=1, fps=5)
#     i, j, n = 0, 0, data.shape[0]
#     ax = plt.gca()
#     while j < 1:
#         ax.clear()
#         ax.set_title('Bi-pyramid Data')
#         if i < n:
#             ax.imshow(data[i].T, clim=clim)
#         else:
#             ax.imshow(data[2 * n - i - 1].T, clim=clim)
#         i += 1
#         if i == 2 * n - 1:
#             i, j = 0, j + 1
#         plt.draw()
#         if smallVid is None:
#             plt.show(block=False)
#             plt.pause(.2)
#         else:
#             __makeVid(writer, plt, stage=2)
#     if smallVid is not None:
#         __makeVid(writer, plt, stage=3)
#     exit()

    if True:
        Radon, fidelity, data, ASpace, PSpace, params = standardGaussTomo(
            data=data / data.max(), dim=3, device='GPU', isotropic=False,
            vol_box=[-1, 1], vol_size=(data.shape[1], data.shape[1], data.shape[2]),
            angles=angles, det_box=[-1, 1],
            fidelity='l2_squared', reg=None, solver='Newton'
        )
        reg, GD = params

        # Initiate Recon:
        #####
        def newAtoms(n, seed=None):
            tmp = ASpace.random(n, seed=seed)
            c.set(tmp.r, 10, (slice(None), slice(None, 3)))
            c.set(tmp.r, 0, (slice(None), slice(3, None)))
            c.set(tmp.x[:], c.mul(tmp.x, .5))
            c.set(tmp.I[:], 4 / n)
            return tmp

        nAtoms = 200
        recon = newAtoms(nAtoms, 1)
        #####
        R = Radon(recon)

        def guess(d, a): return doKL_ProjGDStep_iso(d, a, 1e-0, Radon)

        guess = None

        GD(recon, data, [200, 1, 100], fidelity, reg, Radon,
           guess=guess, RECORD=RECORD, thresh=.07, angles=((-123, -40), (-164, -123)))

    #     from Fourier_Transform import GaussFT, GaussFTVolume
    #     gFT = GaussFT(ASpace)
    #     dFT = GaussFT(PSpace)
    #     FT = GaussFTVolume(ASpace, PSpace)
    #
    #     def vview(a): return Radon.discretise(gFT.inverse(a))
    #     GD(gFT(recon), dFT(data), [200, 1, 100], fidelity, reg, FT, view=vview,
    #        guess=guess, RECORD=RECORD, tol=1e-6, min_iter=10,
    #        myderivs=FT.derivs)

        savemat(join('store', 'Bipyramid_recon_' + str(nAtoms)),
                {'view': Radon.discretise(recon).asarray(),
                 'X': c.asarray(recon.x),
                 'R': c.asarray(recon.r),
                 'I': c.asarray(recon.I), })
    else:
        import odl
        vol = (data.shape[1], data.shape[1], data.shape[2])
        Radon, _, fidelity, data, vol, PSpace, params = standardGaussTomo(
            data=data / data.max(), dim=3, solver='odl', fidelity='l2_squared', reg=['TV', 1e-3],
            vol_box=([-v / sqrt(128) for v in vol], [v / sqrt(128) for v in vol]), vol_size=vol,
            angles=angles,
            det_box=([-v / sqrt(128) for v in vol[1:]], [v / sqrt(128) for v in vol[1:]]), det_size=vol[1:]
        )

        op_norm = 1.001 * odl.power_method_opnorm(params[2])
        niter = 500  # Number of iterations
        tau = 1 / op_norm  # Step size for the primal variable
        sigma = 1 / (tau * op_norm ** 2)  # Step size for the dual variable
        callback = (odl.solvers.CallbackPrintIteration(step=50) & 
                    odl.solvers.CallbackShow(step=50))
        x = params[2].domain.zero()
        odl.solvers.pdhg(x, params[0], params[1], params[2], tau=tau, sigma=sigma, niter=niter,
                         callback=callback)

#         x.show(title='TV reconstruction', force_show=True)
        x = x.asarray()
        savemat(join('store', 'Bipyramid_recon_TV'), {'view': x})

# from matplotlib import pyplot as plt
# y = loadmat(join('store', 'Bipyramid_recon'))
# n, y = 2, y['view']
# from GD_lib import _get3DvolPlot
# plt.close()
# plt.figure()
# _get3DvolPlot(None, x[::n, ::n, ::n], (-168, -158), .07)
# # plt.figure()
# # _get3DvolPlot(None, y[::n, ::n, ::n], (-168, -158), .05)
# plt.show()
# exit()
