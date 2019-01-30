'''
Created on 10 Jul 2018

@author: Rob Tovey
'''
from code.transport_loss import l2_squared_loss
from code.regularisation import Joubert, null, Mass, Radius
from os.path import join
from Fourier_Transform import GaussFT, GaussFTVolume
RECORD = join('store', '2_atoms_3D_GD_rand')
RECORD = None
if RECORD is not None:
    import matplotlib
    matplotlib.use('Agg')
import odl
from code.dictionary_def import VolSpace, ProjSpace, AtomSpace, AtomElement
from code.atomFuncs import GaussTomo, GaussVolume
from numpy import sqrt, pi
from code.bin.manager import myManager

with myManager(device='cpu', order='C', fType='float32', cType='complex64') as c:
    # Space settings:
    dim = 3
    device = 'GPU'
    ASpace = AtomSpace(dim, isotropic=False)
    vol = VolSpace(odl.uniform_discr([-1] * 3, [1] * 3, [64] * 3))

    # Projection settings:
#     PSpace = ProjSpace(odl.uniform_discr([0, -pi / 2], [pi, pi / 2], [50, 1]),
#                        odl.uniform_discr([-sqrt(dim)] * 2,
#                                          [sqrt(dim)] * 2, [64] * 2))
    PSpace = ProjSpace(odl.uniform_discr([0], [pi], [50]),
                       odl.uniform_discr([-sqrt(dim)] * 2,
                                         [sqrt(dim)] * 2, [64] * 2))

    # Initiate Data:
    #####
    # # These lines initiate the 2 atom demo, seed=200 is a good one
#     gt = AtomElement(ASpace, x=[[0.64, 0.78, -0.34],
#                                 [0.29, -0.82, -0.78]], r=[[10, 10, 10, 0, 0, 0], [10, 7, 5, 0, 0, 0]], I=[2, 1])
    gt = AtomElement(ASpace, x=[[0, 0, 0]], r=[
                     [10, 10, 10, 0, 0, 0]], I=[1])
    # # These lines generate random atoms
    nAtoms = 1
#     gt = ASpace.random(nAtoms, seed=6)  # 0,1,3,6,8
    recon = ASpace.random(nAtoms, seed=1)
    c.set(recon.r, 4, (slice(None), slice(None, 3)))
    c.set(recon.r, 0, (slice(None), slice(3, None)))
    c.set(recon.I[:], 1)
    c.set(recon.x[:], recon.x / 2)
#     c.set(gt.I[:], 1)
    #####
    nAtoms = recon.I.shape[0]
    Radon = GaussTomo(ASpace, PSpace, device=device)
    gFT = GaussFT(ASpace)
    dFT = GaussFT(PSpace)
    FT = GaussFTVolume(ASpace, PSpace)
    view = GaussVolume(ASpace, vol, device=device)

    def vview(a): return view(gFT.inverse(a))

    gt_sino = Radon(gt)
    gt_view = view(gt)

    # Reconstruction:
    fidelity = l2_squared_loss(dim)
    reg = null(dim)
    guess = None
    from NewtonGaussian import linesearch as GD

    GD(gFT(recon), dFT(gt_sino), [100, 1, 100], fidelity, reg, FT, vview,
       gt=gt_view, guess=guess, RECORD=RECORD, tol=1e-10, min_iter=10)
#     GD(recon, gt_sino, [100, 1, 100], fidelity, reg, Radon, view,
#        gt=gt_view, guess=guess, RECORD=RECORD, tol=1e-10, min_iter=10)
