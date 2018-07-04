'''
Created on 5 Jan 2018

@author: Rob Tovey
'''
from code.transport_loss import l2_squared_loss, Transport_loss
from code.bin.manager import myManager
from GD_lib import linesearch as GD
from KL_GaussRadon import doKL_ProjGDStep_iso
RECORD = '2_atoms_2D_OT'
RECORD = None
import odl
from code.bin.dictionary_def import VolSpace, ProjSpace, AtomSpace, AtomElement
from code.bin.atomFuncs import GaussTomo, GaussVolume
from numpy import sqrt, pi, zeros, random, arange
from matplotlib import pyplot as plt, animation as mv
from code.regularisation import Joubert, null

with myManager(device='cpu', order='C', fType='float32', cType='complex64') as c:
    # Space settings:
    dim = 2
    device = 'GPU'  # CPU or GPU
    ASpace = AtomSpace(dim)
    vol = VolSpace(odl.uniform_discr([-1] * 2, [1] * 2, [128] * 2))

    # Projection settings:
    PSpace = ProjSpace(odl.uniform_discr(0, pi, 30),
                       odl.uniform_discr(-1.5 * sqrt(dim),
                                         1.5 * sqrt(dim), 128))

    # Initiate Data:
    #####
    # #   These lines initiate the 2 atom demo
    gt = AtomElement(ASpace, x=[[-.5, 0], [.5, 0]], r=[1 / 30, 1 / 10], I=1)
    recon = AtomElement(ASpace, [[.2, .5], [-.2, -.5]], [1 / 30, 1 / 10], 1)
    # #   These lines generate random atoms
    # nAtoms = 5
    # gt = ASpace.random(nAtoms, seed=6)  # 6, 10
    # recon = ASpace.random(nAtoms)
    # recon.I, recon.r = gt.I, gt.r
    #####
    nAtoms = len(gt.r)
    Radon = GaussTomo(ASpace, PSpace, device=device)
    view = GaussVolume(ASpace, vol, device=device)
    gt_sino = Radon(gt)
    gt_view = view(gt)
    R = Radon(recon)

    # from atomFuncs import test_grad
    # test_grad(ASpace, Radon, [10**-(k + 0) for k in range(6)])
    # exit()

    # Reconstruction:
    fidelity = l2_squared_loss(dim)
    reg = Joubert(dim, 1e2, 3e-1, (.1**2, 1e2))
#     fidelity = Transport_loss(dim, device=device)
#     reg = Joubert(dim, 1e2, 1e+2, (.5, 1e2))
    reg = null(dim)

    def guess(d, a): return doKL_ProjGDStep_iso(d, a, 1e-0, Radon)
#     def guess(d, a): return doKL_LagrangeStep_2Diso(d, a, 1e-3, Radon)
#     def guess(d, a): return a

    GD(recon, gt_sino, [100, 1], fidelity, reg, Radon, view,
       gt=gt_view, dim='xrI', guess=guess, RECORD=RECORD, tol=1e-4, min_iter=20)