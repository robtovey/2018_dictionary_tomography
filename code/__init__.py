'''
Created on 5 Apr 2018

@author: Rob Tovey
'''
import odl
from numpy import isscalar
from code.bin.dictionary_def import AtomSpace, ProjSpace, ProjElement, VolSpace
from code.bin.atomFuncs import GaussVolume, GaussTomo
from code.transport_loss import l2_squared_loss, Transport_loss
from code.regularisation import null, Joubert, Mass, Radius, Iso
from GD_lib import linesearch, linesearch_block
from NewtonGaussian import linesearch as Newton_linesearch, linesearch_block as Newton_linesearch_block
from spyder.app.tests.script import arr


def standardGaussTomo(gt=None, data=None, noise=None, dim=3, device='GPU', isotropic=False,
                      vol_box=None, vol_size=None,
                      angles=None, angle_range=None, angle_num=None, axis=[0, 0, 1],
                      det_box=None, det_size=None,
                      fidelity='l2_squared', reg=None,
                      solver='Newton'):

    # Volume parameters
    if vol_box is None:
        vol_box = [-1, 1]
    if vol_size is None:
        if gt is None:
            raise ValueError('If <gt> is not provided then <vol_size> must.')
        else:
            vol_size = gt.shape
    vol_box = __padshape(vol_box, [dim, dim])
    vol_size = __padshape(vol_size, dim)
    vol = odl.uniform_partition(vol_box[0], vol_box[1], vol_size)

    # Detector properties
    if angles is None:
        if (dim == 3) and not (isscalar(angle_num) or len(angle_num) == 1):
            angle_range = __padshape(angle_range, [dim - 1, dim - 1])
        angles = odl.uniform_partition(
            angle_range[0], angle_range[1], angle_num, nodes_on_bdry=True)
    else:
        angles = __touniformpartition(angles)

    if det_box is None:
        det_box = [-1.3, 1.3]
    if det_size is None:
        if data is not None:
            det_size = data.shape[1:]
        elif gt is not None:
            det_size = gt.shape[1:]
        else:
            raise ValueError(
                'If <det_size> is not provided then either <data> or <gt> must.')
    det_box = __padshape(det_box, [dim - 1, dim - 1])
    det_size = __padshape(det_size, dim - 1)
    detector = odl.uniform_partition(det_box[0], det_box[1], det_size)

    if solver == 'odl' or data is None:
        if dim == 2:
            PSpace = odl.tomo.Parallel2dGeometry(angles, detector)
        elif angles.ndim == 1:
            PSpace = odl.tomo.Parallel3dAxisGeometry(
                angles, detector, axis=axis)
        else:
            PSpace = odl.tomo.Parallel3dEulerGeometry(angles, detector)
        vol = odl.uniform_discr_frompartition(vol, dtype='float32')

        Radon = odl.tomo.RayTransform(vol, PSpace)

        if data is None:
            if gt is None:
                data = Radon.range.zero()
            else:
                data = Radon(gt)
        if noise is not None:
            data = data + noise

    if solver == 'odl':
        if fidelity == 'l2_squared':
            fidelity = odl.solvers.L2NormSquared(Radon.range).translated(data)
        else:
            raise ValueError(
                'At the moment, for odl fidelity must be "l2_squared".')
        g = odl.solvers.IndicatorNonnegativity(Radon.domain)
        if reg is None:
            reg = str(None)
        if reg[0] == 'TV':
            grad = odl.Gradient(vol)
            op = odl.BroadcastOperator(Radon, grad)
            TV = reg[1] * odl.solvers.L1Norm(grad.range)
            f = odl.solvers.SeparableSum(fidelity, TV)

            print(odl.power_method_opnorm(grad),
                  odl.power_method_opnorm(Radon))
        else:
            op = Radon
            f = fidelity

        return Radon, lambda x: x, fidelity, data, vol, PSpace, (f, g, op)

    else:
        vol = VolSpace(vol)
        ASpace = AtomSpace(dim, isotropic=isotropic)
        PSpace = ProjSpace(angles, detector)
        Radon = GaussTomo(ASpace, PSpace, device=device)
        view = GaussVolume(ASpace, vol, device=device)

        if hasattr(data, 'asarray'):
            data = data.asarray()
        data.shape = (-1,) + data.shape[-dim + 1:]
        data = ProjElement(PSpace, data)

        if fidelity == 'l2_squared':
            fidelity = l2_squared_loss(dim)
        elif fidelity == 'Transport':
            fidelity = Transport_loss(dim, device=device)

        if reg is None:
            reg = null(None)
        elif reg[0] == 'Joubert':
            reg = Joubert(dim, *reg[1])
        elif reg[0] == 'Mass':
            reg = Mass(dim, *reg[1])
        elif reg[0] == 'Radius':
            reg = Radius(dim, *reg[1])
        elif reg[0] == 'Iso':
            reg = Iso(dim, *reg[1])
        else:
            raise ValueError(
                'Given <reg>, %s, was not recognised.' % repr(reg))

        if solver == 'Newton':
            GD = Newton_linesearch
        elif solver == 'Newton_block':
            GD = Newton_linesearch_block
        elif solver.endswith('_block'):
            GD = linesearch_block
        else:
            GD = linesearch

        return Radon, view, fidelity, data, ASpace, PSpace, (reg, GD)


def __padshape(arr, shape):
    if isscalar(arr):
        arr = [arr, ]
    if isscalar(shape):
        shape = [shape, ]

    if len(shape) == 1:
        if len(arr) == 1:
            return (arr[0],) * shape[0]
        elif len(arr) == shape[0]:
            return arr
        else:
            raise ValueError(
                'If <shape> is scalar/length 1 then <arr> must be scalar, length 1 or length <shape>.')

    if len(arr) == 1:
        return [(arr[0],) * s for s in shape]
    elif len(arr) == len(shape):
        return [__padshape(arr[i], shape[i]) for i in range(len(arr))]
    else:
        raise ValueError(
            'If <shape> is a vector then either <arr> is scalar or the same length of <shape>.')


def __touniformpartition(arr):
    if isinstance(arr, odl.RectPartition):
        return arr
    elif hasattr(arr, 'grid'):
        arr = arr.grid
    elif isscalar(arr[0]):
        arr = odl.RectGrid(arr)
    else:
        arr = odl.RectGrid(*arr)

    return odl.uniform_partition_fromgrid(arr)
