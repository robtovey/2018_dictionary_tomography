'''
Created on 16 Apr 2018

@author: Rob Tovey
'''
from code.transport_loss import l2_squared_loss, Transport_loss
from KL_GaussRadon import doKL_LagrangeStep_iso, doKL_ProjGDStep_iso
from GD_lib import linesearch as GD
RECORD = 'multi_aniso_atoms_2D'
RECORD = None
import odl
from code.bin.dictionary_def import VolSpace, ProjSpace, AtomSpace, AtomElement
from code.bin.atomFuncs import GaussTomo, GaussVolume
from numpy import sqrt, pi, zeros, random, arange, log10
from matplotlib import pyplot as plt, animation as mv
from code.bin.manager import myManager
from code.regularisation import Joubert, null

with myManager(device='cpu', order='C', fType='float32', cType='complex64') as c:
    # Space settings:
    dim = 2
    device = 'GPU'  # CPU or GPU
    ASpace = AtomSpace(dim, isotropic=True)
    vol = VolSpace(odl.uniform_discr([-1] * 2, [1] * 2, [128] * 2))

    # Projection settings:
    PSpace = ProjSpace(odl.uniform_discr(0, pi, 30),
                       odl.uniform_discr(-1.5 * sqrt(dim),
                                         1.5 * sqrt(dim), 128))

    # Initiate Data:
    #####
    # #   These lines initiate the 2 atom demo
    gt = AtomElement(ASpace, x=[[-.5, 0], [.5, 0]], r=[.30, .10], I=1)
#     recon = AtomElement(ASpace, [[.2, .5], [-.2, -.5]], [.18, .18], 1)
    recon = ASpace.random(10, seed=1)
    # #   These lines generate random atoms
#     nAtoms = 1  # 5
#     gt = ASpace.random(nAtoms, seed=2)  # 6, 10
#     recon = ASpace.random(nAtoms)
#     c.set(recon.r, 1, (slice(None), slice(None, 2)))
#     c.set(recon.r, 0, (slice(None), slice(2, None)))
#     c.set(recon.I[:], 1)
#     c.set(gt.I[:], 1)
    #####
    nAtoms = recon.I.shape[0]
    Radon = GaussTomo(ASpace, PSpace, device=device)
    view = GaussVolume(ASpace, vol, device=device)
    gt_sino = Radon(gt)
    gt_view = view(gt)
    R = Radon(recon)

    # Reconstruction:
    fidelity = l2_squared_loss(dim)
#     reg = Joubert(dim, 1e0, 3e-1, (1e1**dim, 1e-3))
    reg = null(dim)

    fig, ax = plt.subplots(2, 3)
    if RECORD is not None:
        writer = mv.writers['ffmpeg'](fps=5, metadata={'title': RECORD})
        writer.setup(fig, RECORD + '.mp4', dpi=100)

    max_iter = 10
    F = zeros(max_iter * nAtoms + 1)
    E = zeros(max_iter * nAtoms + 1)
    F[0] = fidelity(R, gt_sino)
    E[0] = F[0] + reg(recon)
    jj = 0
    for i in range(max_iter):
        for j in random.permutation(arange(nAtoms)):
            #             recon[j] = doKL_LagrangeStep_iso(
            #                 gt_sino - R, recon[j], 1e-3, Radon)
            recon[j] = doKL_ProjGDStep_iso(
                gt_sino - R, recon[j], 1e-4, Radon)
            R = Radon(recon)
            jj += 1
            F[jj] = fidelity(R, gt_sino)
            E[jj] = F[jj] + reg(recon)

#         print(abs(recon.x - gt.x).max(), abs(recon.r -
# gt.r).max(), abs(recon.I - gt.I).max())
#             print((i + 1) / max_iter)

        ax[0, 0].clear()
        ax[0, 1].clear()
        ax[1, 0].clear()
        ax[1, 1].clear()

        view(recon).plot(ax[0, 0])
        ax[0, 0].set_title('Reconstruction')
        gt_view.plot(ax[0, 1])
        ax[0, 1].set_title('GT')
        R.plot(ax[1, 0], aspect='auto')
        ax[1, 0].set_title('Reconstructed Sinogram')
        (R - gt_sino).plot(ax[1, 1], aspect='auto')
        ax[1, 1].set_title('Sinogram Error')
        ax[0, 2].plot(log10(F[:jj]))
        ax[0, 2].set_title('log10(Fidelity)')
        ax[1, 2].plot(log10(E[:jj]))
        ax[1, 2].set_title('log10(Energy)')
        try:
            plt.pause(.1)
            if RECORD is not None:
                writer.grab_frame()
        except NameError:
            exit()

    if RECORD is not None:
        writer.finish()
    plt.show(block=True)
print('Reconstruction complete')
