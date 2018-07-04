'''
Created on 10 Mar 2018

@author: Rob Tovey
'''
from os.path import join
RECORD = join('store', 'DogBones_small')
RECORD = None
if RECORD is not None:
    import matplotlib
    matplotlib.use('Agg')
import odl
from numpy import sqrt, loadtxt, asarray, pi, ascontiguousarray
from PIL import Image

# Import data:
angles = loadtxt(
    join('DogBones', 'Sample_A2_Tilt_Series_tiltcorr_cut.rawtlt'))
angles = odl.RectPartition(
    odl.IntervalProd(-pi / 2, pi / 2), odl.RectGrid((pi / 180) * angles))
# angles = odl.uniform_partition(-(pi / 180) * 69, (pi / 180) * 73, 71)
data = Image.open(
    join('DogBones', 'Sample A2 Tilt Series_tiltcorr_cut.tif'))
data = [asarray(data).T for i in range(data.n_frames) if data.seek(i) is None]
# First dimension is angle, second is width, third is slice
data = asarray(data)
# import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt, animation as mv
# writer = mv.writers['ffmpeg'](fps=5, metadata={'title': 'DogBones'})
fig = plt.figure()
# writer.setup(fig, 'DogBones' + '.mp4', dpi=100)
for i in range(data.shape[0]):
    plt.gca().clear()
    plt.imshow(data[i, :, 500:564].T)
    plt.pause(0.1)
#     writer.grab_frame()
# writer.finish()
exit()


# Space variables
vol = list(data.shape)
vol[0] = vol[1]  # recon same as each row width
vol = odl.uniform_partition([-v / sqrt(128)
                             for v in vol], [v / sqrt(128) for v in vol], vol)

vol = odl.uniform_discr_frompartition(vol[:, :, 500:564], dtype='float32')
data = ascontiguousarray(data[:, :, 500:564], dtype='float32')


PSpace = (angles,
          odl.uniform_partition([-v / sqrt(128) for v in data.shape[1:]], [v / sqrt(128) for v in data.shape[1:]], data.shape[1:]))
PSpace = odl.tomo.Parallel3dAxisGeometry(*PSpace, axis=[0, 0, 1])


# Operators
Radon = odl.tomo.RayTransform(vol, PSpace)
grad = odl.Gradient(vol)
op = odl.BroadcastOperator(Radon, grad)
g = odl.solvers.functional.default_functionals.IndicatorNonnegativity(
    op.domain)
fidelity = odl.solvers.L2NormSquared(Radon.range).translated(data)
TV = .0003 * odl.solvers.L1Norm(grad.range)
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
# op_norm = 1.001 * odl.power_method_opnorm(op)
# print('Op norm = ', op_norm)
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
savemat(join('store', 'DogBones_recon'), {'x': x.__array__()})
