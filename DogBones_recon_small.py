'''
Created on 10 Mar 2018

@author: Rob Tovey
'''
from os.path import join
from KL_GaussRadon import doKL_ProjGDStep_iso
from GaussDictCode import standardGaussTomo
RECORD = join('store', 'DogBones_rand100_iso')
RECORD = None
from numpy import loadtxt, asarray, ascontiguousarray, pi
from GaussDictCode.bin.manager import myManager
from PIL import Image

with myManager(device='cpu', order='C', fType='float32', cType='complex64') as c:
    # Import data:
    angles = loadtxt(
        join('DogBones', 'Sample_A2_Tilt_Series_tiltcorr_cut.rawtlt')) * pi / 180
    data = Image.open(
        join('DogBones', 'Sample A2 Tilt Series_tiltcorr_cut.tif'))
    data = [asarray(data).T for i in range(
        data.n_frames) if data.seek(i) is None]
    # First dimension is angle, second is width, third is slice
    data = asarray(data)
    # Space settings:

#     data = ascontiguousarray(data[:, ::16, 500:564:4], dtype='float32')
    data = ascontiguousarray(data[:, 300:-300:5, 500:600:5], dtype='float32')

#     from matplotlib import pyplot as plt
# #     plt.imshow(data[35])
# #     plt.colorbar()
# #     plt.show()
# #     exit()
#     for i in range(0, 71):
#         plt.gca().clear()
#         plt.imshow(data[i].T)
#         plt.title(str(i))
#         plt.pause(.1)
#     plt.show()
#     exit()

    Radon, fidelity, data, ASpace, PSpace, params = standardGaussTomo(
        data=(data - data.min()) / 151786, dim=3, device='GPU', isotropic=False,
        vol_box=[-1, 1], vol_size=(data.shape[1], data.shape[1], data.shape[2]),
        angles=angles, det_box=[-1, 1],
        fidelity='l2_squared', reg=None, solver='Newton'
    )
    reg, GD = params

    # Initiate Recon:
    #####
    def newAtoms(n, seed=None):
        tmp = ASpace.random(n, seed=seed)
        c.set(tmp.r, 1, (slice(None), slice(None, 3)))
        c.set(tmp.r, 0, (slice(None), slice(3, None)))
        c.set(tmp.I[:], 1e-2)
#         c.set(tmp.x, 0, (slice(None), [2]))
        return tmp

    nAtoms = 20
    recon = newAtoms(nAtoms, 1)
    #####
    dsum = c.sum(data.asarray())
#     c.mul(recon.I, dsum / c.sum(Radon(recon).asarray()), recon.I)
    R = Radon(recon)

#     Radon.discretise(recon).plot(plt, Sum=1)
#     plt.show(block=True)
#     exit()

    def guess(d, a): return doKL_ProjGDStep_iso(d, a, 1e-0, Radon)

    guess = None

    GD(recon, data, [100, 1, 100], fidelity, reg, Radon,
       thresh=.1, guess=guess, RECORD=RECORD)

#     from Fourier_Transform import GaussFT, GaussFTVolume
#     gFT = GaussFT(ASpace)
#     dFT = GaussFT(PSpace)
#     FT = GaussFTVolume(ASpace, PSpace)
#
#     def vview(a): return Radon.discretise(gFT.inverse(a))
#     GD(gFT(recon), dFT(data), [300, 1, 100], fidelity, reg, FT, view=vview,
#        guess=guess, RECORD=RECORD, tol=1e-6, min_iter=10,
#        myderivs=FT.derivs)
