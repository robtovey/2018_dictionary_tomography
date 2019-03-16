'''
Created on 27 Feb 2018

@author: Rob Tovey
'''
import numpy as np
from numba import cuda
try:
    from numba.cuda.cudadrv.devicearray import DeviceNDArray as GPUArray
except ImportError:
    GPUArray = type(None)
from .numba_cuda_aux import __GPU_reduce_flex_F as _GPU_reduce_flex_F, \
    __GPU_reduce_flex_C as _GPU_reduce_flex_C, __GPU_fill_C as _GPU_fill_C, \
    __GPU_fill_F as _GPU_fill_F, THREADS


class __tmp:
    my_list = []

    def __call__(self):
        return self.my_list[0]

    def append(self, context):
        self.my_list.insert(0, context)

    def remove(self, context):
        self.my_list.remove(context)


context = __tmp()


class myManager:

    def __init__(self, device='cpu', order='C', fType='float32', cType='complex64'):
        self.device = device.lower()
        self.order = order
        self.fType = fType
        self.cType = cType

        Sig = ['void(T', ',T', ',T[:],i4)']
        Func = [
            'def func(x,y,z,sz):\n\ti=cuda.grid(1);\n\tif i<sz:\n\t\tz[i] = x', 'y', ';']
        Sym = ['+', '-', '*', '/']
        Scal = [[1, 1], [1, 0], [0, 1], [0, 0]]

        self.__funcs = {}
        for i in range(4):
            for j in range(4):
                exec(Func[0]
                     +('[i]' if not Scal[j][0] else '')
                     +Sym[i] + Func[1]
                     +('[i]' if not Scal[j][1] else '') + Func[2]
                     )
                name = Sym[i] + str(Scal[j][0]) + str(Scal[j][1]) + 'f'
                self.__funcs[name] = cuda.jit(
                    (Sig[0] + ('[:]' if not Scal[j][0] else '')
                     +Sig[1] + ('[:]' if not Scal[j][1] else '') + Sig[2]).replace('T', fType))(locals()['func'])
                name = Sym[i] + str(Scal[j][0]) + str(Scal[j][1]) + 'c'
                self.__funcs[name] = cuda.jit(
                    (Sig[0] + ('[:]' if not Scal[j][0] else '')
                     +Sig[1] + ('[:]' if not Scal[j][1] else '') + Sig[2]).replace('T', cType))(locals()['func'])

    def __enter__(self):
        context.append(self)
        return self

    def __exit__(self, *other):
        context.remove(self)

    def empty(self, shape, dtype='float'):
        if np.isscalar(shape):
            shape = (shape,)
        if str(dtype)[0] == 'f':
            if self.device == 'gpu':
                tmp = cuda.device_array(
                    shape, dtype=str(self.fType), order=self.order)
            else:
                tmp = np.empty(shape, dtype=self.fType, order=self.order)
        else:
            if self.device == 'gpu':
                tmp = cuda.device_array(
                    shape, dtype=str(self.cType), order=self.order)
            else:
                tmp = np.empty(shape, dtype=self.cType, order=self.order)

        return tmp

    def zeros(self, shape, dtype='float'):
        if np.isscalar(shape):
            shape = (shape,)
        if str(dtype)[0] == 'f':
            tmp = np.zeros(shape, dtype=self.fType, order=self.order)
        else:
            tmp = np.zeros(shape, dtype=self.cType, order=self.order)
        if self.device == 'gpu':
            tmp = cuda.to_device(tmp)
        return tmp

    def rand(self, shape):
        if np.isscalar(shape):
            shape = (shape,)
        tmp = np.random.rand(*shape).astype(self.fType)
        if self.device == 'gpu':
            tmp = cuda.to_device(tmp)
        # Does not generate complex random numbers
        return tmp

    def cast(self, x, isComplex=None):
        if isComplex is None:
            isComplex = _isComplex(x)
        if issubclass(type(x), GPUArray):
            if self.device == 'gpu':
                # Need to check data type
                pass
            else:
                x = x.copy_to_host()
                if isComplex:
                    x = x.astype(self.cType, order=self.order)
                else:
                    x = x.astype(self.fType, order=self.order)
        else:
            x = np.asarray(x)
            if isComplex:
                x = x.astype(self.cType, order=self.order)
            else:
                x = x.astype(self.fType, order=self.order)
            if self.device == 'gpu':
                x = cuda.to_device(x)

        return x

    def cat(self, *lst, axis=0):
        lst = [x.copy_to_host() if isinstance(x, GPUArray)
               else np.asarray(x) for x in lst]
        lst = np.concatenate(lst, axis=axis)
        return self.cast(lst)

    def asarray(self, x):
        if issubclass(type(x), GPUArray):
            return x.copy_to_host()
        elif type(x) in (list, tuple):
            return np.asarray([self.asarray(X) for X in x])
        else:
            return np.asarray(x)

    def copy(self, x):
        if hasattr(x, 'copy'):
            X = x.copy()
        else:
            X = self.empty(x.shape, x.dtype)
            self.set(X, x)

        return self.cast(X)

    def __getparams(self, x, y):
        if np.isscalar(x) or x.size == 1:
            if np.isscalar(y) or y.size == 1:
                shape = (1,)
                string = '11' + \
                    ('c' if (_isComplex(x) or _isComplex(y)) else 'f')
            else:
                shape = y.shape
                string = '10' + \
                    ('c' if (_isComplex(x) or _isComplex(y)) else 'f')
        else:
            shape = x.shape
            string = '0' + str(int(np.isscalar(y) or y.size == 1)) + \
                ('c' if (_isComplex(x) or _isComplex(y)) else 'f')
        return shape, np.prod(shape), string

    def set(self, x, y, Slice=None):
        if Slice is None:
            if issubclass(type(x), GPUArray):
                y = self.cast(y)
                sz = np.asarray(
                    [-((-x.size) // THREADS), x.size], dtype='int32')
                if y.size == 1:
                    y = y[0]
                    if _isComplex(x) or _isComplex(y):
                        _GPU_fill_C[sz[0], THREADS](
                            x.reshape((sz[1],)), np.asarray(y, dtype=x.dtype), sz)
                    else:
                        _GPU_fill_F[sz[0], THREADS](
                            x.reshape((sz[1],)), np.asarray(y, dtype=x.dtype), sz)
                else:
                    if _isComplex(x) or _isComplex(y):
                        _GPU_fill_C[sz[0], THREADS](
                            x.reshape((sz[1],)), y.reshape((sz[1],)), sz)
                    else:
                        _GPU_fill_F[sz[0], THREADS](
                            x.reshape((sz[1],)), y.reshape((sz[1],)), sz)
            elif issubclass(type(y), GPUArray):
                x[:] = y.copy_to_host()
            else:
                x[:] = y
        else:
            if issubclass(type(x), GPUArray):
                X = x.copy_to_host()
                Y = np.asarray(y) if not issubclass(
                    type(y), GPUArray) else y.copy_to_host()
                if Y.shape == X.shape:
                    X[Slice] = Y[Slice]
                else:
                    X[Slice] = Y
                self.set(x, X, None)
                return
            elif issubclass(type(y), GPUArray):
                y = y.copy_to_host()
            y = np.asarray(y)
            if y.shape == x.shape:
                x[Slice] = y[Slice]
            else:
                x[Slice] = y

    def neg(self, x, out=None):
        if self.device == 'cpu':
            if out is None:
                z = -x
            else:
                out[:] = -x
                z = out
        else:
            z = self.mult(x, -1, out)
        return z

    def add(self, x, y, out=None):
        if self.device == 'cpu':
            if out is None:
                z = x + y
            else:
                out[:] = x + y
                z = out
        else:
            shape, size, string = self.__getparams(x, y)
            if hasattr(x, 'ndim'):
                if x.ndim > 1:
                    x = x.reshape(size)
                elif x.ndim == 1 and string[0] == '1':
                    x = x[0]
            if hasattr(y, 'ndim'):
                if y.ndim > 1:
                    y = y.reshape(size)
                elif y.ndim == 1 and string[1] == '1':
                    y = y[0]
            if out is None:
                z = self.empty(size, dtype=string[-1])
            else:
                z = out.reshape(size)
            grid = -((-size) // THREADS)
            self.__funcs['+' + string][grid, THREADS](x, y, z, size)
            z = z.reshape(shape)
        return z

    def sub(self, x, y, out=None):
        if self.device == 'cpu':
            if out is None:
                z = x - y
            else:
                out[:] = x - y
                z = out
        else:
            shape, size, string = self.__getparams(x, y)
            if hasattr(x, 'ndim'):
                if x.ndim > 1:
                    x = x.reshape(size)
                elif x.ndim == 1 and string[0] == '1':
                    x = x[0]
            if hasattr(y, 'ndim'):
                if y.ndim > 1:
                    y = y.reshape(size)
                elif y.ndim == 1 and string[1] == '1':
                    y = y[0]
            if out is None:
                z = self.empty(size, dtype=string[-1])
            else:
                z = out.reshape(size)
            grid = -((-size) // THREADS)
            self.__funcs['-' + string][grid, THREADS](x, y, z, size)
            z = z.reshape(shape)
        return z

    def mul(self, x, y, out=None):
        if self.device == 'cpu':
            if out is None:
                z = x * y
            else:
                out[:] = x * y
                z = out
        else:
            shape, size, string = self.__getparams(x, y)
            if hasattr(x, 'ndim'):
                if x.ndim > 1:
                    x = x.reshape(size)
                elif x.ndim == 1 and string[0] == '1':
                    x = x[0]
            if hasattr(y, 'ndim'):
                if y.ndim > 1:
                    y = y.reshape(size)
                elif y.ndim == 1 and string[1] == '1':
                    y = y[0]
            if out is None:
                z = self.empty(size, dtype=string[-1])
            else:
                z = out.reshape(size)
            grid = -((-size) // THREADS)
            self.__funcs['*' + string][grid, THREADS](x, y, z, size)
            z = z.reshape(shape)
        return z

    def div(self, x, y, out=None):
        #         if abs(self.asarray(y)).min() < 1e-16:
        #             print(self.asarray(y))
        #             raise
        if self.device == 'cpu':
            if out is None:
                z = x / y
            else:
                out[:] = x / y
                z = out
        else:
            shape, size, string = self.__getparams(x, y)
            if hasattr(x, 'ndim'):
                if x.ndim > 1:
                    x = x.reshape(size)
                elif x.ndim == 1 and string[0] == '1':
                    x = x[0]
            if hasattr(y, 'ndim'):
                if y.ndim > 1:
                    y = y.reshape(size)
                elif y.ndim == 1 and string[1] == '1':
                    y = y[0]
            if out is None:
                z = self.empty(size, dtype=string[-1])
            else:
                z = out.reshape(size)
            grid = -((-size) // THREADS)
            self.__funcs['/' + string][grid, THREADS](x, y, z, size)
            z = z.reshape(shape)
        return z

    def sum(self, x):
        if self.device == 'cpu':
            z = x.sum()
        else:
            # Threads per block = THREADS
            # Elements per thread = 1000
            # Number of blocks = sz/(1000*THREADS)
            sz = x.size
            L = 1024 * THREADS
            blocks = -((-sz) // L)
            if _isComplex(x):
                x = x.reshape((sz,))
                buf = cuda.device_array((2, blocks), dtype=x[0].real.dtype)
                _GPU_reduce_flex_C[blocks, THREADS](x, buf, sz)
                _GPU_reduce_flex_F[1, THREADS](buf[0], buf[0], blocks)
                _GPU_reduce_flex_F[1, THREADS](buf[1], buf[1], blocks)
                z = complex(buf[0, 0], buf[1, 0])
            else:
                buf = cuda.device_array((blocks,), dtype=x.dtype)
                _GPU_reduce_flex_F[blocks, THREADS](x.reshape((sz,)), buf, sz)
                _GPU_reduce_flex_F[1, THREADS](buf, buf, blocks)
                z = buf[0]
        return z


def _isComplex(x):
    if hasattr(x, 'dtype'):
        return (x.dtype.str[1] == 'c')
    else:
        return np.iscomplex(x)


context.append(myManager())
