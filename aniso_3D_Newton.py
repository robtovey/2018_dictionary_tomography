'''
Created on 18 Jun 2018

@author: Rob Tovey
'''
from code.transport_loss import l2_squared_loss, Transport_loss
from code.regularisation import Joubert, null, Mass, Radius
from KL_GaussRadon import doKL_ProjGDStep_iso
from os.path import join
RECORD = join('store', '2_atoms_3D_GD_rand')
# RECORD = None
if RECORD is not None:
    import matplotlib
    matplotlib.use('Agg')
import odl
from code.bin.dictionary_def import VolSpace, ProjSpace, AtomSpace, AtomElement
from code.bin.atomFuncs import GaussTomo, GaussVolume
from numpy import sqrt, pi
from matplotlib import pyplot as plt, animation as mv
from code.bin.manager import myManager

with myManager(device='cpu', order='C', fType='float32', cType='complex64') as c:
    # Space settings:
    dim = 3
    device = 'GPU'
    ASpace = AtomSpace(dim, isotropic=False)
    vol = VolSpace(odl.uniform_discr([-1] * 3, [1] * 3, [64] * 3))

    # Projection settings:
#     PSpace = ProjSpace(odl.uniform_discr([0] * 2, [pi, pi / 2], [20, 10]),
#                        odl.uniform_discr([-sqrt(dim)] * 2,
#                                          [sqrt(dim)] * 2, [64] * 2))
    PSpace = ProjSpace(odl.uniform_discr([0, -pi / 2], [pi, pi / 2], [50, 1]),
                       odl.uniform_discr([-sqrt(dim)] * 2,
                                         [sqrt(dim)] * 2, [64] * 2))

    # Initiate Data:
    #####
    # # These lines initiate the 2 atom demo, seed=200 is a good one
#     gt = AtomElement(ASpace, x=[[-.5, 0, 0], [.5, 0, 0]], r=[30, 10], I=1)
    gt = AtomElement(ASpace, x=[[0.64, 0.78, -0.34],
                                [0.29, -0.82, -0.78]], r=[[10, 10, 10, 0, 0, 0], [10, 7, 5, 0, 0, 0]], I=[2, 1])
#     recon = AtomElement(ASpace, [[.2, .5, 0], [-.2, -.5, 0]], [30, 10], 1)
    # # These lines generate random atoms
    nAtoms = 4
#     gt = ASpace.random(nAtoms, seed=6)  # 0,1,3,6,8
    recon = ASpace.random(nAtoms, seed=1)
    c.set(recon.r, 7, (slice(None), slice(None, 3)))
    c.set(recon.r, 0, (slice(None), slice(3, None)))
    c.set(recon.I[:], 1)
#     c.set(gt.I[:], 1)
    #####
    nAtoms = recon.I.shape[0]
    Radon = GaussTomo(ASpace, PSpace, device=device)
    view = GaussVolume(ASpace, vol, device=device)
    gt_sino = Radon(gt)
    gt_view = view(gt)
    R = Radon(recon)

#     while True:
#     for i in range(gt_sino.shape[0]):
#         plt.gca().clear()
#         gt_sino.plot(plt, Slice=[i])
#         plt.title(str(i))
#         plt.pause(.1)
#     exit()

    # Reconstruction:
    fidelity = l2_squared_loss(dim)
#     fidelity = Transport_loss(dim, device=device)
#     reg = Joubert(dim, 1e-1, 3e-2, (1e1**dim, 1e-3))
#     fidelity = Transport_loss(dim, device=device)
#     reg = Joubert(dim, 0 * 1e2, 1e2, (1e1**dim, 1e-1))
    reg = null(dim)
#     reg = Mass(dim, m=1e-4, k=2)
#     reg = Radius(dim, s=1e-1)

#     #####
#     from numpy import log10, array
#     da = ASpace.random(1)
#     DA = array(list(da.I) + list(da.x[0]) + list(da.r[0]))
#     recon = recon[0]
#     f, df, ddf = derivs(recon, Radon, gt_sino)
#     f = fidelity(Radon(recon), gt_sino)
#     DF = (DA * df).sum() * sqrt(2 * pi)**2
#     DDF = (ddf.dot(DA) * DA).sum() * sqrt(2 * pi)**2
#
#     for e in [.1**j for j in range(-3, 4)]:
#         gt.I[0] = recon.I[0] + e * da.I[0]
#         gt.x[0, :] = recon.x[0] + e * da.x[0]
#         gt.r[0, :] = recon.r[0] + e * da.r[0]
#         f2 = fidelity(Radon(gt[0]), gt_sino)
# #         print(log10(abs(f2 - f - e * DF)), log10(e))
# #         print(abs(f2 - f) / abs(e * DF))
# #         print(log10(abs(f2 - f - e * DF - e * e * DDF)), log10(e))
#         print(abs(f2 - f - e * DF) / abs(e * e * DDF))
#
#     exit()
#     #####

    def guess(d, a): return doKL_ProjGDStep_iso(d, a, 1e-0, Radon)
#     def guess(d, a): return doKL_LagrangeStep_2Diso(d, a, 1e-3, Radon)
#     def guess(d, a): return a
    guess = None

    from NewtonGaussian import linesearch as GD
#     from NewtonGaussian import linesearch_block as GD
#     from GD_lib import linesearch as GD
    from GD_lib import linesearch_block as GD
    GD(recon, gt_sino, [100, 1, 100], fidelity, reg, Radon, view,
       gt=gt_view, guess=guess, RECORD=RECORD, tol=1e-6, min_iter=1)
