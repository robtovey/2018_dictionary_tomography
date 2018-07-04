'''
Created on 16 Apr 2018

@author: Rob Tovey
'''
from os.path import join
RECORD = join('store', 'polyII')
RECORD = None
if RECORD is not None:
    import matplotlib
    matplotlib.use('Agg')
import odl
from odl.contrib import mrc
from numpy import sqrt, ascontiguousarray, log10, pi, pad
from matplotlib import pyplot as plt, animation as mv

# Import data:
angles = odl.uniform_partition(-60 * pi / 180,
                               60 * pi / 180, 61, nodes_on_bdry=True)
with mrc.FileReaderMRC(join('PolyII', 'rna_phantom.mrc')) as f:
    f.read_header()
    gt = f.read_data()
    gt -= gt.min()
gt = pad(gt, ((2, 3), (0, 0), (10, 10)), 'constant')


# Space settings:
vol = list(gt.shape)
vol = odl.uniform_partition([-v / sqrt(128)
                             for v in vol], [v / sqrt(128) for v in vol], vol)

vol = odl.uniform_discr_frompartition(vol[::1, ::1, ::1], dtype='float32')
gt = ascontiguousarray(gt[::1, ::1, ::1], dtype='float32')

PSpace = (angles,
          odl.uniform_partition([-v / sqrt(128) for v in gt.shape[1:]], [v / sqrt(128) for v in gt.shape[1:]], gt.shape[1:]))
PSpace = odl.tomo.Parallel3dAxisGeometry(*PSpace, axis=[0, 0, 1])

# Operators
Radon = odl.tomo.RayTransform(vol, PSpace)
data = Radon(gt)
grad = odl.Gradient(vol)
op = odl.BroadcastOperator(Radon, grad)
g = odl.solvers.functional.default_functionals.IndicatorNonnegativity(
    op.domain)
fidelity = odl.solvers.L2NormSquared(Radon.range).translated(data)
TV = .0001 * odl.solvers.L1Norm(grad.range)
# fidelity = odl.solvers.L1Norm(Radon.range).translated(data)
# TV = .03 * odl.solvers.L1Norm(grad.range)
f = odl.solvers.SeparableSum(fidelity, TV)
data /= abs(Radon.adjoint(data).__array__()).max()

# fbp = odl.tomo.fbp_op(Radon)
# fbp = fbp(data)
# fbp.show(title='FBP', force_show='True')
# exit()

# op_norm = [odl.power_method_opnorm(Radon), odl.power_method_opnorm(grad)]
# print(op_norm, (op_norm[0]**2 + op_norm[1]**2)**.5)

op_norm = 25.7  # 10 slices: 23.2, 200 slices: 23.4
# print('Computing operator norm...')
op_norm = 1.001 * odl.power_method_opnorm(op)
print('Op norm = ', op_norm)
niter = 1000  # Number of iterations
tau = 1 / op_norm  # Step size for the primal variable
sigma = 1 / (tau * op_norm**2)  # Step size for the dual variable
callback = (odl.solvers.CallbackPrintIteration(step=10) &
            odl.solvers.CallbackShow(step=10))
x = op.domain.zero()

odl.solvers.pdhg(x, f, g, op, tau=tau, sigma=sigma, niter=niter,
                 callback=callback)

x.show(title='TV reconstruction', force_show=True)

from scipy.io import savemat
savemat(join('store', 'polyII_recon'), {'x': x.__array__()})
