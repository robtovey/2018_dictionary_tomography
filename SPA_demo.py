'''
Created on 11 Mar 2019

@author: Rob Tovey


This module is intended as an introduction to using the code for basic
least-squares/gradient descent. The setup of the geometry is particularly
general but more customisation is always possible, mainly by looking in 
GaussDictCode.__init__.py.

List of geometry options:
    dim=2 or 3 should be self-explanatory
    a_range defines the angular range (rectangle) on which to scan
    a_num defines the number of projections on a regular grid
    vol_box/vol_size defines the range/discretisation of the volume to 
        for viewing. Extended to higher dimensions by simple product.
    det_box/det_size defines the range/discretisation of the detector.
    
Choosing operator:
    2 main operators chosen by 'myOperator':
        standardGaussTomo is intended for low number of high-res projections, big atoms
        standardSingleParticleGaussTomo is intended for high number, low res projections
            with a point-spread function and small atoms.
    The attributes here could be mixed on request.
    
Computing gradients:
    There are many ways to compute function values, several to compute gradients,
        and only 1 (useful) way to compute Hessians. These are basically shown
        in the chunk of code: from numpy import set_printoptions...exit()
    There is a Radon.gradient(G) method which wraps Radon.derivs(G,d,order)
        where G is a list of Gaussians, d a 'direction' in Sinogram space
        and order=1 will speed up computations a bit if you don't need the 
        Hessian.
    For computing least squares, there are Radon.L2_derivs(G,res,order) which
        computes derivatives of .5|Radon(G)-res|^2 up to order=order
    All methods are intended atom-wise so if G has more than 1 atom then
        all return values are lists of length len(G). For Radon.gradient/derivs this
        is just a little weird but for the Hessian information this is a block-diagonal 
        approximation. This is really weird for L2_derivs unless you do it 1 at a time,
        somehow the sum of function values is not the function value of the sum, so I may 
        try to update this behaviour at some point.
    
Initialising lists of Gaussians:
    gt=... shows how to construct explicitely
    recon = ASpace.random... shows how to initialise randomly
    Attributes of Gaussian are g(I,x,r)(y) = I\exp(.5*|r(y-x)|^2)
    r is parametrised as an upper triangular matrix by diagonals:
        r = [r_{00},r_{11},r_{22},r_{01},r_{1,2},r_{20}] for dim=3
    isotropic=True/False determines whether radius is scalar or not
    
List of optimisation options:
    solver\in\{'Newton','GD'}. Newton uses the second order derivatives too. 
        Each method is 'stochastic' in the sense that each step selects a 
        random gaussian and increments just that Gaussian.
    '[50,1,1]' is choice of number of iterations, basically 50. The other 
        numbers were there to enable the addition of 1 Gaussian at a time.
        Let me know if you want to look into this. 
    
List of visualisation options:
    If a plot is labelled 'sinogram' then it is the full 2D data projection
    If a plot is labelled 'data' then it is a 2D projection from the dataset
    If the plot label is not explicit then it is a volume render, either 2D or 3D
    All comparative plots are normalised equivalently, i.e. if you see a 
        reconstruction render and a ground truth (gt) render then same colours 
        correspond to equivalent grey-scale values.
    'Fidelity' = 'Energy' + regulariser but for the moment the regulariser is 
        always 0 so these are the same. The only difference is 'Fidelity' is 
        plotted on a log scale.
    RECORD=<filename> or None dictates whether visualisation is shown or saved
    angles is a coordinate on the sphere (in degrees) which is used as a 
        viewing angle in 3D reconstructions.
    thresh determines which level-set to show in 3D renderering
    The chunk of code 'from matplotlib ... exit()' is useful for setting up
        3D angles/threshold. You can rotate the image with the mouse and read
        off angles in bottom corner.

'''
dim = 2
from GaussDictCode import standardSingleParticleGaussTomo, standardGaussTomo
from os.path import join
RECORD = join('store', 'mesh_rand30_radius')
RECORD = None
from GaussDictCode.dictionary_def import AtomElement
from numpy import pi, prod

a_range = (0, pi) if dim == 2 else ([0] * 2, [pi] * 2)
a_num = (100,) if dim == 2 else (20, 20)

myOperator = standardSingleParticleGaussTomo  # standardSingleParticleGaussTomo or standardGaussTomo
Radon, fidelity, _, ASpace, PSpace, params = myOperator(
    dim=dim, device='GPU', isotropic=False,
    angle_range=a_range, angle_num=a_num,
    vol_box=[-1, 1], vol_size=32, det_box=[-1.4, 1.4], det_size=16,
    fidelity='l2_squared', reg=None,
    solver='Newton'
)
reg, GD = params
scale = (Radon.range.volume / prod(Radon.range.shape) , Radon.embedding.cell_volume) if myOperator is standardGaussTomo else (1, 1) 

#####
# Initiate Data:
#####
if dim == 2:
    I = 10 * scale[0]
    gt = AtomElement(ASpace, x=[[-.5, 0], [.5, 0]], r=[3, 1], I=I)
else:
    I = 10 * scale[0]
    gt = AtomElement(ASpace, x=[[-.5, .5, 0], [.5, -.5, 0]], r=[3, 1], I=I)

# # These lines generate random atoms
nAtoms = 10
recon = ASpace.random(nAtoms, seed=1)
recon.r[:, :dim] = 1.5
recon.r[:, dim:] = 0
recon.I[:] = 0.1 * I
#####
nAtoms = recon.I.shape[0]
gt_sino = Radon(gt)
gt_view = Radon.discretise(gt)
R = Radon(recon)

# from numpy import set_printoptions; set_printoptions(3, suppress=False, sign='+')
# res = Radon(recon[0]) - gt_sino
# tmp1 = Radon.L2_derivs(recon[0], gt_sino)  # 0,1,2 order derivatives of |R(g)-gt_sino|^2/2
# tmp2 = Radon.derivs(recon[0], res, order=1)  # 0,1 order derivatives of R(g)\cdot res
#  
# print('Least Squares = E(I,x,r): %2.3e = %2.3e = %2.3e = %2.3e' % (.5 * (res.asarray() ** 2).sum(), fidelity(Radon(recon[0]), gt_sino),
#                                         tmp1[0], .5 * (tmp2[0] - gt_sino.inner(res).real)))
# print('\n\nGradient = [dE/dI, dE/dx, ..., dE/dr,...]: \n%s \n= \n%s \n= \n%s' % (str(tmp1[1]), str(tmp2[1]),
#                                                                               str((Radon.gradient(recon[0])(res)).asarray())))
# print('\n\nHessian = [[d^2E/dI^2, d^2E/dIdx, ..., d^2E/dIdr,...],...]: \n%s\n' % (str(tmp1[2]),))
# exit()

# from matplotlib import pyplot as plt
# from GD_lib import _get3DvolPlot
# _get3DvolPlot(None, gt_view.asarray(), (15, 25), 3e-4 * scale[0] / scale[1])
# plt.title('Ground Truth', {'fontsize': 26})
# #     plt.savefig('store/mesh_gt.eps', format='eps', dpi=600)
# plt.show()
# exit()

GD(recon, gt_sino, [50, 1, 1], fidelity, reg, Radon,
   gt=gt_view, RECORD=RECORD, tol=1e-6,
   angles=((15, 25), (15, 115)), thresh=3e-4 * scale[0] / scale[1])
