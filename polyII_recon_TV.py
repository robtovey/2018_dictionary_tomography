'''
Created on 16 Apr 2018

@author: Rob Tovey
'''
from os.path import join
import odl
from odl.contrib import mrc
from numpy import ascontiguousarray, pi, pad, sqrt
from code import standardGaussTomo
from scipy.io import savemat, loadmat

# Import data:
with mrc.FileReaderMRC(join('PolyII', 'rna_phantom.mrc')) as f:
    f.read_header()
    gt = f.read_data()
    gt -= gt.min()
gt = pad(gt, ((2, 3), (0, 0), (10, 10)), 'constant')
gt = ascontiguousarray(gt[30:70, 30:70, 30:70], dtype='float32')
vol = list(gt.shape)

# from matplotlib import pyplot as plt
# from GD_lib import _get3DvolPlot
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# x = loadmat(join('store', 'polyII_TV'))
# x = x['view']
# f = plt.figure()
# _get3DvolPlot(f.add_subplot(221, projection='3d'), gt, (20, 45), 1.5)
# plt.title('GT, view from orientation: ' + str((20, 45)))
# _get3DvolPlot(f.add_subplot(222, projection='3d'), gt, (100, 90), 1.5)
# plt.title('GT, view from orientation: ' + str((100, 90)))
# _get3DvolPlot(f.add_subplot(223, projection='3d'), x, (20, 45), 1.5)
# plt.title('TV recon, view from orientation: ' + str((20, 45)))
# _get3DvolPlot(f.add_subplot(224, projection='3d'), x, (100, 90), 1.5)
# plt.title('TV recon, view from orientation: ' + str((100, 90)))
# plt.show()
# exit()


Radon, _, fidelity, data, vol, PSpace, params = standardGaussTomo(
    gt=gt, dim=3, solver='odl', fidelity='l2_squared', reg=['TV', 3e-6],
    vol_box=([-v / sqrt(128) for v in vol], [v / sqrt(128) for v in vol]), vol_size=vol,
    angle_range=[-60 * pi / 180, 60 * pi / 180], angle_num=61, noise=0.1,
    det_box=([-v / sqrt(128) for v in vol[1:]], [v / sqrt(128) for v in vol[1:]]), det_size=vol[1:]
)

# n = 4
# Radon, _, fidelity, data, vol, PSpace, params = standardGaussTomo(
#     gt=gt, dim=3, solver='odl', fidelity='l2_squared', reg=['TV', 1e-5],
#     vol_box=([-v / n for v in vol], [v / n for v in vol]), vol_size=vol,
#     angle_range=[-pi / 3, pi / 3], angle_num=61 * 4,
#     det_box=([-v / n for v in vol[1:]], [v / n for v in vol[1:]]), det_size=int(40 / 2)
# )

scale = abs(Radon.adjoint(data).__array__()).max()
data /= scale

# fbp = odl.tomo.fbp_op(Radon)
# fbp = fbp(data)
# fbp.show(title='FBP', force_show='True')
# exit()

print('Computing operator norm...')
op_norm = 1.001 * odl.power_method_opnorm(params[2])
print('Op norm = ', op_norm)
niter = 500  # Number of iterations
tau = 1 / op_norm  # Step size for the primal variable
sigma = 1 / (tau * op_norm**2)  # Step size for the dual variable
callback = (odl.solvers.CallbackPrintIteration(step=50) &
            odl.solvers.CallbackShow(step=50))
x = params[2].domain.zero()
odl.solvers.pdhg(x, params[0], params[1], params[2], tau=tau, sigma=sigma, niter=niter,
                 callback=callback)

x.show(title='TV reconstruction', force_show=True)

data *= scale
x = x.asarray() * scale
print('GT RMSE = %f, data RMSE = %f' %
      ((abs(gt - x)**2).sum() / gt.size,
       (abs((data - Radon(x)).asarray())**2).sum() / data.size))

# savemat(join('store', 'polyII_TV'), {'view': x})
