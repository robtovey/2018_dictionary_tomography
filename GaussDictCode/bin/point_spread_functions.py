'''
Created on 30 Jan 2019

@author: Rob Tovey
'''
from numpy import empty, pi, array, zeros, ascontiguousarray, log10
import numba
from numba import cuda, float32 as f4, int32 as i4
from math import exp as c_exp, sqrt as c_sqrt, floor, ceil
from .numba_cuda_aux import THREADS, __GPU_reduce_n


def r2aniso(r, dim):
    if r.shape[1] > 1:
        return r
    else:
        rr = zeros((r.shape[0], 3 if dim == 2 else 6))
        for i in range(dim):
            rr[:, i] = r
        return ascontiguousarray(rr, r.dtype)


@cuda.jit(device=True, inline=True)
def __Radius(r):
    R = min(r[0], r[1])
    if r.size > 3:
        R = min(R, r[2])
    return 1 / R


@cuda.jit(device=True, inline=True)
def __simpleindex(x, p, R, ind):
    # Assume p[i] = p[0] + i*(p[1]-p[0])
    ind[0] = int((x - R - p[0]) / (p[1] - p[0]))
    ind[1] = int((x + R - p[0]) / (p[1] - p[0])) + 1
    
    ind[0] = max(0, ind[0])
    ind[1] = min(p.size, ind[1] + 1)  # [i,j] = range(i,j+1)


@cuda.jit(device=True, inline=True)
def __index(x, y, p, R, ind):
    IP = 0
    n = 0
    for i in range(x.size):
        IP += x[i] * y[i]
        n += y[i] * y[i]
    IP /= n
    R = R / c_sqrt(n)
    
    # Assume p[i] = p[0] + i*(p[1]-p[0])
    ind[0] = int((IP - R - p[0]) / (p[1] - p[0]))
    ind[1] = int((IP + R - p[0]) / (p[1] - p[0])) + 1
    
    ind[0] = max(0, ind[0])
    ind[1] = min(p.size, ind[1] + 1)  # [i,j] = range(i,j+1)


@cuda.jit(device=True, inline=True)
def __total(I, r):
    if r.size == 3:
        return I / (r[0] * r[1]) * c_sqrt(pi / 30)
    else:
        return I / (r[0] * r[1] * r[2]) * (pi / 30)


@cuda.jit(device=True, inline=True)
def __norm(x, DIM):
    tmp = 0
    for i in range(DIM):
        tmp += x[i] * x[i]
    return c_sqrt(tmp)


@cuda.jit(device=True, inline=True)
def __projnorm(x, T, DIM):
    tmp0, tmp1 = 0, 0
    for i in range(DIM):
        tmp0 += x[i] * x[i]
    for i in range(T.size):
        tmp1 += x[i] * T[i]
    tmp0 = tmp0 - tmp1 * tmp1
    if tmp0 > 0:
        return c_sqrt(tmp0)
    else:
        return 0


@numba.jit(cache=True, nopython=True)
def __countbins(x, x0, r, R, Len):
    Bin = [0, 0, 0]
    for j0 in range(x.shape[0]):
        X = x[j0] - x0
        for i in range(3):
            Bin[i] = max(0, int(floor((X[i] - r) / R)))
            X[i] -= Bin[i] * R

        for i in range(2):
            if (X[0] < i * (R - r)) or (Bin[0] + i >= Len.shape[0]): 
                continue
            for j in range(2):
                if (X[1] < j * (R - r)) or (Bin[1] + j >= Len.shape[1]): 
                    continue
                for k in range(2):
                    if (X[2] < k * (R - r)) or (Bin[2] + k >= Len.shape[2]): 
                        continue
                    Len[Bin[0] + i, Bin[1] + j, Bin[2] + k] += 1


@numba.jit(cache=True, nopython=True)
def __rebin(I, x, rr, x0, subI, subx, subr, r, R, Len):
    Bin = [0, 0, 0]
    for j0 in range(x.shape[0]):
        X = x[j0] - x0
        for i in range(3):
            Bin[i] = max(0, int(floor((X[i] - r) / R)))
            X[i] -= Bin[i] * R

        for i in range(2):
            if (X[0] < i * (R - r)) or (Bin[0] + i >= Len.shape[0]): 
                continue
            for j in range(2):
                if (X[1] < j * (R - r)) or (Bin[1] + j >= Len.shape[1]): 
                    continue
                for k in range(2):
                    if (X[2] < k * (R - r)) or (Bin[2] + k >= Len.shape[2]): 
                        continue
                    ind = Len[Bin[0] + i, Bin[1] + j, Bin[2] + k]
                    subI[Bin[0] + i, Bin[1] + j, Bin[2] + k, ind] = I[j0]
                    subx[Bin[0] + i, Bin[1] + j, Bin[2] + k, ind] = x[j0]
                    subr[Bin[0] + i, Bin[1] + j, Bin[2] + k, ind] = rr[j0]
                    Len[Bin[0] + i, Bin[1] + j, Bin[2] + k] += 1

    for b0 in range(subI.shape[0]):
        for b1 in range(subI.shape[1]):
            for b2 in range(subI.shape[2]):
                j0 = Len[b0, b1, b2]
                if j0 < subI.shape[3]:
                    subI[b0, b1, b2, j0] = -1


def rebin(I, x, r, y, R0, R):
    DIM = x.shape[1]
    if DIM == 2:
        y = [y[0], y[1], array([0])]
        X = zeros((x.shape[0], 3), dtype=x.dtype)
        X[:, :2] = x
        x = ascontiguousarray(X)
    xmin = array([Y.item(0) for Y in y], dtype='float32')
    nbins = [1 + max(0, int(floor((y[i].item(-1) - y[i].item(0) + R0) / R)))
             for i in range(3)]

    Len = zeros(nbins, dtype='i4')
    __countbins(x, xmin, R0, R, Len)

    II = zeros(nbins + [Len.max()], dtype='f4')
    xx = zeros(nbins + [Len.max(), x.shape[1]], dtype='f4')
    rr = zeros(nbins + [Len.max(), r.shape[1]], dtype='f4')
    Len[:] = 0
    __rebin(I, x, r, xmin, II, xx, rr, R0, R, Len)
    
    if DIM == 2:
        xx = ascontiguousarray(xx[..., :2])
    return II, xx, rr


def VolProj(atom, y, u):
    DIM = atom.x.shape[1]
    R0 = 1 / atom.r[:, :DIM].min()
    R = max(1 + y[0].size // 8, 1.5 * ceil(R0 / abs(y[0].item(1) - y[0].item(0))))
    I, x, r = rebin(atom.I, atom.x, r2aniso(atom.r, DIM), y,
                    R0, R * abs(y[0].item(1) - y[0].item(0)))
    
    grid = I.shape[:DIM]
    tpb = 4
    if DIM == 2:
        __VolProj2[tuple(-(-g // tpb) for g in grid), (tpb, tpb)
              ](I, x, r, y[0], y[1], R, u)
    else:
        __VolProj3[tuple(-(-g // tpb) for g in grid), (tpb, tpb, tpb)
                  ](I, x, r, *y, R, u)


@cuda.jit(device=True, inline=True)
def __f(x, r, y, DIM):
    Y, rY = cuda.local.array((3,), f4), cuda.local.array((3,), f4)

    for i in range(DIM):
        Y[i] = y[i] - x[i]
    if DIM == 2:
        rY[0] = r[0] * Y[0] + r[2] * Y[1]
        rY[1] = r[1] * Y[1]
    else:
        rY[0] = r[0] * Y[0] + r[3] * Y[1] + r[5] * Y[2]
        rY[1] = r[1] * Y[1] + r[4] * Y[2]
        rY[2] = r[2] * Y[2]

    n = 0
    for i in range(DIM):
        n += rY[i] * rY[i]

    return c_exp(-30 * n) * c_sqrt(30 / pi)


@cuda.jit(device=True, inline=True)
def __if2(I, x, r, y, s):
    X = cuda.local.array((2,), f4)
    DIM = 2
    for i in range(DIM):
        X[i] = x[i] - y[i]

    R, n = __Radius(r), __norm(X, DIM) 
    if n + R < s:
        return __total(I, r)
    elif n - R > s:
        return 0
    
    limits = cuda.local.array((2, 2), f4)
    Y = cuda.local.array((2,), f4)
    for i in range(2):
        limits[i, 0] = max(X[i] - R, -s)
        limits[i, 1] = min(X[i] + R, s)
    
        Y[i] = limits[i, 0]
        limits[i, 0] = .1 * (limits[i, 1] - limits[i, 0])
        
    count, sum = 0, 0
    for _ in range(11):
        Y[0] += limits[0, 0]
        for __ in range(11):
            Y[1] += limits[1, 0]
            sum += __f(X, r, Y, 2)
            count += 1
        Y[1] -= 11 * limits[1, 0]
        
#     return sum
    return I * sum / count * 100 * limits[0, 0] * limits[1, 0]


@cuda.jit(device=True, inline=True)
def __if3(I, x, r, y, s):
    X = cuda.local.array((3,), f4)
    DIM = 3
    for i in range(DIM):
        X[i] = x[i] - y[i]
    
    R, n = __Radius(r), __norm(X, DIM) 
    if n + R < s:
        return __total(I, r)
    elif n - R > s:
        return 0
    
    limits = cuda.local.array((3, 2), f4)
    Y = cuda.local.array((3,), f4)
    for i in range(DIM):
        limits[i, 0] = max(X[i] - R, -s)
        limits[i, 1] = min(X[i] + R, s)

        Y[i] = limits[i, 0]
        limits[i, 0] = .1 * (limits[i, 1] - limits[i, 0])
         
    count, sum = 0, 0
    for _ in range(11):
        Y[0] += limits[0, 0]
        for __ in range(11):
            Y[1] += limits[1, 0]
            for __ in range(11):
                Y[2] += limits[2, 0]
                sum += __f(X, r, Y, 3)
                count += 1
            Y[2] -= 11 * limits[2, 0]
        Y[1] -= 11 * limits[1, 0]
         
#     return sum
    return I * sum / count * 1000 * limits[0, 0] * limits[1, 0] * limits[2, 0]


@cuda.jit
def __VolProj2(I, x, r, y0, y1, rad, u):
    j, k = cuda.grid(2)
    if j >= I.shape[0] or k >= I.shape[1]:
        return
    jj, kk = int(rad * j), int(rad * k)
    s = abs(y1[1] - y1[0]) / 2  # assume uniform grid size

    II, xx, rr = I[j, k, 0], x[j, k, 0], r[j, k, 0]
    y = cuda.local.array((2,), f4)
    for jjj in range(jj, min(jj + int(rad), y0.size)):
        y[0] = y0[jjj]
        for kkk in range(kk, min(kk + int(rad), y1.size)):
            y[1] = y1[kkk]
            tmp = 0
            for ii in range(II.size):
                if II[ii] == -1:
                    break
                tmp += __if2(II[ii], xx[ii], rr[ii], y, s)
            u[jjj, kkk] = tmp


@cuda.jit
def __VolProj3(I, x, r, y0, y1, y2, rad, u):
    j, k, l = cuda.grid(3)
    if j >= I.shape[0] or k >= I.shape[1] or l >= I.shape[2]:
        return
    jj, kk, ll = i4(rad * j), i4(rad * k), i4(rad * l)
    s = abs(y1[1] - y1[0]) / 2  # assume uniform grid size

    II, xx, rr = I[j, k, l], x[j, k, l], r[j, k, l]
    irad = i4(rad)
    y = cuda.local.array((3,), f4)
    for jjj in range(jj, min(jj + irad, y0.size)):
        y[0] = y0[jjj]
        for kkk in range(kk, min(kk + irad, y1.size)):
            y[1] = y1[kkk]
            for lll in range(ll, min(ll + irad, y2.size)):
                y[2] = y2[lll]
                tmp = 0
                for ii in range(II.size):
                    if II[ii] == -1:
                        break
                    tmp += __if3(II[ii], xx[ii], rr[ii], y, s)
                u[jjj, kkk, lll] = tmp


def RadProj(atom, Rad, R):
    S = Rad.range
    t, w, p = S.orientations, S.ortho, S.detector
 
    R[...] = 0
    grid = w[0].shape[0],
    tpb = 8
    if len(p) == 1:  # dim=2
        __RadProj_2D[tuple(-(-g // tpb) for g in grid),
                     (tpb,)](atom.I, atom.x, r2aniso(atom.r, 2), t, w[0], p[0], R)
    else:
        if len(w) == 1:  # dim=2.5
            w = (w[0], empty((1, 1), dtype='f4'))
        __RadProj_3D[tuple(-(-g // tpb) for g in grid),
                     (tpb,)](atom.I, atom.x, r2aniso(atom.r, 3), t, w[0], w[1], p[0], p[1], R)


@cuda.jit(device=True, inline=True)
def __Rf(x, r, y, rT, n, Y, rY, MY, DIM):
    for i in range(DIM):
        Y[i] = y[i] - x[i]
    if DIM == 2:
        rY[0] = r[0] * Y[0] + r[2] * Y[1]
        rY[1] = r[1] * Y[1]
    else:
        rY[0] = r[0] * Y[0] + r[3] * Y[1] + r[5] * Y[2]
        rY[1] = r[1] * Y[1] + r[4] * Y[2]
        rY[2] = r[2] * Y[2]
        
    # Final vector:
    IP = 0
    for i in range(DIM):
        IP += rT[i] * rY[i]
    m = 0
    for i in range(DIM):
        MY[i] = rY[i] - rT[i] * IP
        m += MY[i] * MY[i]

    return c_exp(-30 * m) * n


@cuda.jit(device=True, inline=True)
def __rf2(I, x, r, y, T, W0, s):
    DIM = 2
    X = cuda.local.array((2,), f4)

    for i in range(DIM):
        X[i] = y[i] - x[i]

    R, n = __Radius(r), __projnorm(X, T, DIM) 
    if n + R < s:
        return __total(I, r)
    elif n > R:
        return 0

    rT = cuda.local.array((2,), f4)
    rT[0] = r[0] * T[0] + r[2] * T[1]
    rT[1] = r[1] * T[1]

    # Normalise rT:
    n = 0
    for i in range(DIM):
        n += rT[i] * rT[i]
    n = 1 / c_sqrt(n)
    for i in range(DIM):
        rT[i] *= n

    # Square of points is: x-s -> x+s
    buf0, buf1, buf2 = cuda.local.array((2,), f4), cuda.local.array((2,), f4), cuda.local.array((2,), f4)
    limits, Y = cuda.local.array((2,), f4), cuda.local.array((2,), f4)
    
    # <x,w_i> \not\in (limits[i,0], limits[i,1]) => Rf(x)=0 or x\not\in [-s,s]
    limits[0] = X[0] * W0[0] + X[1] * W0[1]
    limits[1] = min(limits[0] + R, s)
    limits[0] = max(limits[0] - R, -s)
    
    for i in range(DIM):
        Y[i] = limits[0] * W0[i]
    limits[0] = .1 * (limits[1] - limits[0])
           
    count, sum = 0, 0
    for _ in range(11):
        for i in range(DIM):
            Y[i] += limits[0] * W0[i]
        sum += __Rf(X, r, Y, rT, n, buf0, buf1, buf2, 2)
        count += 1
            
    return I * sum / count * 10 * limits[0]


@cuda.jit(device=True, inline=True)
def __rf3(I, x, r, y, T, W0, W1, s):
    DIM, lenT = 3, T.shape[0]
    X = cuda.local.array((3,), f4)

    for i in range(DIM):
        X[i] = y[i] - x[i]
        
    R, n = __Radius(r), __projnorm(X, T, DIM) 
    if n + R < s:
        return __total(I, r)
    elif n - R > s:
        return 0
    
    rT = cuda.local.array((3,), f4)
    if lenT == 2:
        rT[0] = r[0] * T[0] + r[3] * T[1]
        rT[1] = r[1] * T[1]
        rT[2] = 0
    else:
        rT[0] = r[0] * T[0] + r[3] * T[1] + r[5] * T[2]
        rT[1] = r[1] * T[1] + r[4] * T[2]
        rT[2] = r[2] * T[2]

    # Normalise rT:
    n = 0
    for i in range(lenT):
        n += rT[i] * rT[i]
    n = 1 / c_sqrt(n)
    for i in range(lenT):
        rT[i] *= n

    # Square of points is: x-s -> x+s
    buf0, buf1, buf2 = cuda.local.array((3,), f4), cuda.local.array((3,), f4), cuda.local.array((3,), f4)
    limits, Y = cuda.local.array((2, 2), f4), cuda.local.array((3,), f4)

    # <x,w_i> \not\in (limits[i,0], limits[i,1]) => Rf(x)=0 or x\not\in [-s,s]
    limits[0, 0] = X[0] * W0[0] + X[1] * W0[1] + X[2] * W0[2]
    limits[0, 1] = min(limits[0, 0] + R, s)
    limits[0, 0] = max(limits[0, 0] - R, -s)
    limits[1, 0] = X[0] * W1[0] + X[1] * W1[1] + X[2] * W1[2]
    limits[1, 1] = min(limits[1, 0] + R, s)
    limits[1, 0] = max(limits[1, 0] - R, -s)
    
    for i in range(DIM):
        Y[i] = limits[0, 0] * W0[i] + limits[1, 0] * W1[i]
    for i in range(2):
        limits[i, 0] = .1 * (limits[i, 1] - limits[i, 0])
        
    count, sum = 0, 0
    for _ in range(11):
        for i in range(DIM):
            Y[i] += limits[0, 0] * W0[i]
        for __ in range(11):
            for i in range(DIM):
                Y[i] += limits[1, 0] * W1[i]
            sum += __Rf(X, r, Y, rT, n, buf0, buf1, buf2, 3)
            count += 1
        for i in range(DIM):
            Y[i] -= 11 * limits[1, 0] * W1[i]
        
#     return sum
    if count == 0:
        return 0
    else:
        return I * sum / count * 100 * limits[0, 0] * limits[1, 0]


@cuda.jit
def __RadProj_2D(I, x, r, t, w0, p0, R):
    # Single core for each projection
    # Find lower pixel then iterate either side
    jj = cuda.grid(1)
    if jj >= t.shape[0]:
        return
    T, W0 = t[jj], w0[jj]
    s = abs(p0[1] - p0[0]) / 2
    dp = 1 / (p0[1] - p0[0])
    y = cuda.local.array((2,), f4)
    for ii in range(x.shape[0]):
        IP = x[ii, 0] * W0[0] + x[ii, 1] * W0[1]
        rr = __Radius(r[ii])
        bin00 = max(0, int((IP - rr - p0[0]) * dp))
        bin01 = min(int((IP + rr - p0[0]) * dp) + 2, p0.size) 
        bin00, bin01 = 0, p0.size
        
        for k0 in range(bin00, bin01):
            y[0] = p0[k0] * W0[0]
            y[1] = p0[k0] * W0[1]
            R[jj, k0] += __rf2(I[ii], x[ii], r[ii], y, T, W0, s)


@cuda.jit
def __RadProj_3D(I, x, r, t, w0, w1, p0, p1, R):
    # Single core for each projection
    # Find lower pixel then iterate either side
    jj = cuda.grid(1)
    if jj >= t.shape[0]:
        return
    T = t[jj]
    is3D = (w1.size == 1)
    W0, W1 = cuda.local.array((3,), f4), cuda.local.array((3,), f4)
    if is3D:
        W0[0], W0[1], W0[2] = w0[jj, 0], w0[jj, 1], 0
        W1[0], W1[1], W1[2] = 0, 0, 1
    else:
        for i in range(3):
            W0[i] = w0[jj, i]
            W1[i] = w1[jj, i]
        
    s = abs(p0[1] - p0[0]) / 2
    dp = 1 / (p0[1] - p0[0])
    y = cuda.local.array((3,), f4)
    for ii in range(x.shape[0]):
        rr = __Radius(r[ii])

        IP = x[ii, 0] * W0[0] + x[ii, 1] * W0[1] + x[ii, 2] * W0[2]
        bin00 = max(0, int((IP - rr - p0[0]) * dp))
        bin01 = min(int((IP + rr - p0[0]) * dp) + 2, p0.size) 

        IP = x[ii, 0] * W1[0] + x[ii, 1] * W1[1] + x[ii, 2] * W1[2]
        bin10 = max(0, int((IP - rr - p1[0]) * dp))
        bin11 = min(int((IP + rr - p1[0]) * dp) + 2, p1.size)
        
        for k0 in range(bin00, bin01):
            for k1 in range(bin10, bin11):
                for i in range(3):
                    y[i] = p0[k0] * W0[i] + p1[k1] * W1[i]
                R[jj, k0, k1] += __rf3(I[ii], x[ii], r[ii], y, T, W0, W1, s)

 
@cuda.jit(device=True, inline=True)
def __dRf2(I, x, r, y, T, rT, n, Y, rY, MY, R, dR, ddR, order):
    DIM = 2
    index = cuda.local.array((3, 2), dtype=i4)
    index[0, 0], index[0, 1] = 0, 0
    index[1, 0], index[1, 1] = 1, 1
    index[2, 0], index[2, 1] = 0, 1
 
    for i in range(DIM):
        Y[i] = y[i] - x[i]
    rY[0] = r[0] * Y[0] + r[2] * Y[1]
    rY[1] = r[1] * Y[1]
          
    # Final vector:
    IP = 0
    for i in range(DIM):
        IP += rT[i] * rY[i]
    m = 0
    for i in range(DIM):
        MY[i] = rY[i] - rT[i] * IP
        m += MY[i] * MY[i]
  
    # Preliminary derivatives:
    dMydT = cuda.local.array((2, 2), f4)
    # Missing a factor of n
    for i in range(DIM):
        for j in range(DIM):
            dMydT[i, j] = rT[i] * (IP * rT[j] - MY[j])
        dMydT[i, i] -= IP
    dg, dJdy, dJdT = cuda.local.array((3,), f4), cuda.local.array(
        (2,), f4), cuda.local.array((2,), f4)
    dg[0] = c_exp(-30 * m)
    dg[1] = -30 * dg[0]
    dg[2] = 900 * dg[0]
  
    tmp = dg[1] * 2 * n
    for i in range(DIM):
        dJdy[i] = tmp * MY[i]
    tmp = -n * n
    for i in range(DIM):
        dJdT[i] = tmp * (2 * IP * MY[i] * dg[1] + rT[i] * dg[0])
  
    ddJdyy, ddJdyT, ddJdTT = cuda.local.array((2, 2), f4), cuda.local.array(
        (2, 2), f4), cuda.local.array((2, 2), f4)
  
    if order > 1:
        tmp = 2 * n
        for i in range(DIM):
            for j in range(i):
                ddJdyy[i, j] = ddJdyy[j, i]
            ddJdyy[i, i] = tmp * (2 * MY[i] * MY[i] * dg[2] + 
                                  (1 - rT[i] * rT[i]) * dg[1])
            for j in range(i + 1, DIM):
                ddJdyy[i, j] = tmp * (2 * MY[i] * MY[j] * 
                                      dg[2] - rT[i] * rT[j] * dg[1])
        tmp = 2 * n * n
        for i in range(DIM):
            for j in range(DIM):
                ddJdyT[i, j] = tmp * ((dMydT[i, j] - MY[i] * rT[j]) * dg[1]
                                      -2 * IP * MY[i] * MY[j] * dg[2])
        tmp = 2 * n * n * n
        for i in range(DIM):
            for j in range(i):
                ddJdTT[i, j] = ddJdTT[j, i]
            ddJdTT[i, i] = tmp * (
                (1.5 * rT[i] * rT[i] - .5) * dg[0]
                +(2 * IP * MY[i] * rT[i] + IP * rT[i] * MY[i] - 
                   IP * dMydT[i, i] - MY[i] * MY[i]) * dg[1]
                +2 * IP * IP * MY[i] * MY[i] * dg[2])
            for j in range(i + 1, DIM):
                ddJdTT[i, j] = tmp * (
                    1.5 * rT[i] * rT[j] * dg[0]
                    +(2 * IP * MY[i] * rT[j] + IP * rT[i] * MY[j] - 
                       IP * dMydT[i, j] - MY[i] * MY[j]) * dg[1]
                    +2 * IP * IP * MY[i] * MY[j] * dg[2])
  
    # Fill in values:
    R[0] = I * dg[0] * n
  
    # dI
    dR[0] = dg[0] * n
    ddR[0, 0] = 0
    # dIdx
    # ddR[0, 1 + i] = - r_{j,i}dJdy[j]
    ddR[0, 1 + 0] = -r[0] * dJdy[0]
    ddR[0, 1 + 1] = -(r[2] * dJdy[0] + r[1] * dJdy[1])
    # dIdr
    # ddR[0, 3 + (I, i)] = dJdy[I] * Y[i] + dJdT[I] * T[i]
    for i in range(3):
        i0, i1 = index[i]
        ddR[0, 3 + i] = dJdy[i0] * Y[i1] + dJdT[i0] * T[i1]
    # dr, dx
    for i in range(5):
        dR[1 + i] = I * ddR[0, 1 + i]
  
    if order > 1:
        # d^2r
        # ddR[1+(I,i), 1+(J,j)] = I*( ddJdyy[I,J]Y[i]Y[j] + ddJdYT[I,J]Y[i]T[j]
        #                           + ddJdYT[J,I]Y[j]T[i] + ddJdTT[I,J]T[i]T[j])
        for i in range(3):
            i0, i1 = index[i]
            for j in range(i, 3):
                j0, j1 = index[j]
                ddR[3 + i, 3 + j] = I * (ddJdyy[i0, j0] * Y[i1] * Y[j1]
                                         +ddJdyT[i0, j0] * Y[i1] * T[j1]
                                         +ddJdyT[j0, i0] * Y[j1] * T[i1]
                                         +ddJdTT[i0, j0] * T[i1] * T[j1]
                                         )
        # dxdr
        # ddR[1+j, 4+(I,i)] = -I(ddJdyy[I,k]Y[i]r[k,j] + ddJdYT[k,I]T[i]r[k,j]
        #                             + dJdY[I](i==j))
        # 0 -> 0,0
        # 1 -> 1,1
        # 2 -> 2,2
        # 3 -> 0,1
        # 4 -> 1,2
        # 5 -> 0,2
        for i in range(3):
            i0, i1 = index[i]
            ddR[1 + 0, 3 + i] = -I * (
                (ddJdyy[i0, 0] * Y[i1] + ddJdyT[0, i0] * T[i1]) * r[0]
            )
            ddR[1 + 1, 3 + i] = -I * (
                (ddJdyy[i0, 1] * Y[i1] + ddJdyT[1, i0] * T[i1]) * r[1]
                +(ddJdyy[i0, 0] * Y[i1] + ddJdyT[0, i0] * T[i1]) * r[2]
            )
            # j == i1
            ddR[1 + i1, 3 + i] -= I * dJdy[i0]
        # d^2x
        # ddR[1+i,1+j] = I(ddJdYY[I,J]r[I,i]r[J,j])
        ddR[1 + 0, 1 + 0] = I * (
            ddJdyy[0, 0] * r[0] * r[0]
        )
        ddR[1 + 0, 1 + 1] = I * (
            ddJdyy[0, 1] * r[0] * r[1]
            +ddJdyy[0, 0] * r[0] * r[2]
        )

        ddR[1 + 1, 1 + 1] = I * (
            ddJdyy[1, 1] * r[1] * r[1]
            +ddJdyy[1, 0] * r[1] * r[2]
            +ddJdyy[0, 1] * r[2] * r[1]
            +ddJdyy[0, 0] * r[2] * r[2]
        )


@cuda.jit(device=True, inline=True)
def __dRf3(I, x, r, y, T, rT, n, Y, rY, MY, R, dR, ddR, order):
    DIM = 3
    index = cuda.local.array((6, 2), dtype=i4)
    index[0, 0], index[0, 1] = 0, 0
    index[1, 0], index[1, 1] = 1, 1
    index[2, 0], index[2, 1] = 2, 2
    index[3, 0], index[3, 1] = 0, 1
    index[4, 0], index[4, 1] = 1, 2
    index[5, 0], index[5, 1] = 0, 2

    for i in range(DIM):
        Y[i] = y[i] - x[i]
    if DIM == 2:
        rY[0] = r[0] * Y[0] + r[2] * Y[1]
        rY[1] = r[1] * Y[1]
    else:
        rY[0] = r[0] * Y[0] + r[3] * Y[1] + r[5] * Y[2]
        rY[1] = r[1] * Y[1] + r[4] * Y[2]
        rY[2] = r[2] * Y[2]
        
    # Final vector:
    IP = 0
    for i in range(DIM):
        IP += rT[i] * rY[i]
    m = 0
    for i in range(DIM):
        MY[i] = rY[i] - rT[i] * IP
        m += MY[i] * MY[i]
    
    # Preliminary derivatives:
    dMydT = cuda.local.array((3, 3), f4)
    # Missing a factor of n
    for i in range(DIM):
        for j in range(DIM):
            dMydT[i, j] = rT[i] * (IP * rT[j] - MY[j])
        dMydT[i, i] -= IP
    dg, dJdy, dJdT = cuda.local.array((3,), f4), cuda.local.array(
        (3,), f4), cuda.local.array((3,), f4)
    dg[0] = c_exp(-30 * m)
    dg[1] = -30 * dg[0]
    dg[2] = 900 * dg[0]
    
    # First order derivatives:
    tmp = dg[1] * 2 * n
    for i in range(DIM):
        dJdy[i] = tmp * MY[i]
    tmp = -n * n
    for i in range(DIM):
        dJdT[i] = tmp * (2 * IP * MY[i] * dg[1] + rT[i] * dg[0])

    ddJdyy, ddJdyT, ddJdTT = cuda.local.array((3, 3), f4), cuda.local.array(
        (3, 3), f4), cuda.local.array((3, 3), f4)

    if order > 1:
        tmp = 2 * n
        for i in range(DIM):
            for j in range(i):
                ddJdyy[i, j] = ddJdyy[j, i]
            ddJdyy[i, i] = tmp * (2 * MY[i] * MY[i] * dg[2] + 
                                  (1 - rT[i] * rT[i]) * dg[1])
            for j in range(i + 1, DIM):
                ddJdyy[i, j] = tmp * (2 * MY[i] * MY[j] * 
                                      dg[2] - rT[i] * rT[j] * dg[1])
        tmp = 2 * n * n
        for i in range(DIM):
            for j in range(DIM):
                ddJdyT[i, j] = tmp * ((dMydT[i, j] - MY[i] * rT[j]) * dg[1]
                                      -2 * IP * MY[i] * MY[j] * dg[2])
        tmp = 2 * n * n * n
        for i in range(DIM):
            for j in range(i):
                ddJdTT[i, j] = ddJdTT[j, i]
            ddJdTT[i, i] = tmp * (
                (1.5 * rT[i] * rT[i] - .5) * dg[0]
                +(2 * IP * MY[i] * rT[i] + IP * rT[i] * MY[i] - 
                   IP * dMydT[i, i] - MY[i] * MY[i]) * dg[1]
                +2 * IP * IP * MY[i] * MY[i] * dg[2])
            for j in range(i + 1, DIM):
                ddJdTT[i, j] = tmp * (
                    1.5 * rT[i] * rT[j] * dg[0]
                    +(2 * IP * MY[i] * rT[j] + IP * rT[i] * MY[j] - 
                       IP * dMydT[i, j] - MY[i] * MY[j]) * dg[1]
                    +2 * IP * IP * MY[i] * MY[j] * dg[2])

    # Fill in values:
    R[0] = I * dg[0] * n

    # dI
    dR[0] = dg[0] * n
    ddR[0, 0] = 0
    # dIdx
    # ddR[0, 1 + i] = - r_{j,i}dJdy[j]
    ddR[0, 1 + 0] = -r[0] * dJdy[0]
    ddR[0, 1 + 1] = -(r[3] * dJdy[0] + r[1] * dJdy[1])
    ddR[0, 1 + 2] = -(r[5] * dJdy[0] + r[4] * dJdy[1] + r[2] * dJdy[2])
    # dIdr
    # ddR[0, 4 + (I, i)] = dJdy[I] * Y[i] + dJdT[I] * T[i]
    for i in range(6):
        i0, i1 = index[i]
        ddR[0, 4 + i] = dJdy[i0] * Y[i1] + dJdT[i0] * T[i1]
    # dr, dx
    for i in range(9):
        dR[1 + i] = I * ddR[0, 1 + i]
 
    if order > 1:
        #     d^2r
        #     ddR[1+(I,i), 1+(J,j)] = I*( ddJdyy[I,J]Y[i]Y[j] + ddJdYT[I,J]Y[i]T[j]
        #                               + ddJdYT[J,I]Y[j]T[i] + ddJdTT[I,J]T[i]T[j])
        for i in range(6):
            i0, i1 = index[i]
            for j in range(i, 6):
                j0, j1 = index[j]
                ddR[4 + i, 4 + j] = I * (ddJdyy[i0, j0] * Y[i1] * Y[j1]
                                         +ddJdyT[i0, j0] * Y[i1] * T[j1]
                                         +ddJdyT[j0, i0] * Y[j1] * T[i1]
                                         +ddJdTT[i0, j0] * T[i1] * T[j1]
                                         )
        # dxdr
        # ddR[1+j, 4+(I,i)] = -I(ddJdyy[I,k]Y[i]r[k,j] + ddJdYT[k,I]T[i]r[k,j]
        #                             + dJdY[I](i==j))
        # 0 -> 0,0
        # 1 -> 1,1
        # 2 -> 2,2
        # 3 -> 0,1
        # 4 -> 1,2
        # 5 -> 0,2
        for i in range(6):
            i0, i1 = index[i]
            ddR[1 + 0, 4 + i] = -I * (
                (ddJdyy[i0, 0] * Y[i1] + ddJdyT[0, i0] * T[i1]) * r[0]
            )
            ddR[1 + 1, 4 + i] = -I * (
                (ddJdyy[i0, 1] * Y[i1] + ddJdyT[1, i0] * T[i1]) * r[1]
                +(ddJdyy[i0, 0] * Y[i1] + ddJdyT[0, i0] * T[i1]) * r[3]
            )
            ddR[1 + 2, 4 + i] = -I * (
                (ddJdyy[i0, 2] * Y[i1] + ddJdyT[2, i0] * T[i1]) * r[2]
                +(ddJdyy[i0, 1] * Y[i1] + ddJdyT[1, i0] * T[i1]) * r[4]
                +(ddJdyy[i0, 0] * Y[i1] + ddJdyT[0, i0] * T[i1]) * r[5]
            )
            # j == i1
            ddR[1 + i1, 4 + i] -= I * dJdy[i0]
        # d^2x
        # ddR[1+i,1+j] = I(ddJdYY[I,J]r[I,i]r[J,j])
        ddR[1 + 0, 1 + 0] = I * (
            ddJdyy[0, 0] * r[0] * r[0]
        )
        ddR[1 + 0, 1 + 1] = I * (
            ddJdyy[0, 1] * r[0] * r[1]
            +ddJdyy[0, 0] * r[0] * r[3]
        )
        ddR[1 + 0, 1 + 2] = I * (
            ddJdyy[0, 2] * r[0] * r[2]
            +ddJdyy[0, 1] * r[0] * r[4]
            +ddJdyy[0, 0] * r[0] * r[5]
        )
 
        ddR[1 + 1, 1 + 1] = I * (
            ddJdyy[1, 1] * r[1] * r[1]
            +ddJdyy[1, 0] * r[1] * r[3]
            +ddJdyy[0, 1] * r[3] * r[1]
            +ddJdyy[0, 0] * r[3] * r[3]
        )
        ddR[1 + 1, 1 + 2] = I * (
            ddJdyy[1, 2] * r[1] * r[2]
            +ddJdyy[1, 1] * r[1] * r[4]
            +ddJdyy[1, 0] * r[1] * r[5]
            +ddJdyy[0, 2] * r[3] * r[2]
            +ddJdyy[0, 1] * r[3] * r[4]
            +ddJdyy[0, 0] * r[3] * r[5]
        )
 
        ddR[1 + 2, 1 + 2] = I * (
            ddJdyy[2, 2] * r[2] * r[2]
            +ddJdyy[2, 1] * r[2] * r[4]
            +ddJdyy[2, 0] * r[2] * r[5]
            +ddJdyy[1, 2] * r[4] * r[2]
            +ddJdyy[1, 1] * r[4] * r[4]
            +ddJdyy[1, 0] * r[4] * r[5]
            +ddJdyy[0, 2] * r[5] * r[2]
            +ddJdyy[0, 1] * r[5] * r[4]
            +ddJdyy[0, 0] * r[5] * r[5]
        )


@cuda.jit(device=True, inline=True)
def __drf2(I, x, r, Y, T, W0, s, R, dR, ddR, order):
    DIM = 2
    X = cuda.local.array((2,), f4)
  
    for i in range(DIM):
        X[i] = x[i] - Y[i]
    
    rr, n = __Radius(r), __projnorm(X, T, DIM) 
    if n + rr < s:
        R[0] = __total(I, r)
        for j0 in range(6):
            dR[j0] = 0
            for j1 in range(j0, 6):
                ddR[j0, j1] = 0
                ddR[j1, j0] = 0
        return
    elif n - rr > s:
        R[0] = 0
        for j0 in range(6):
            dR[j0] = 0
            for j1 in range(j0, 6):
                ddR[j0, j1] = 0
                ddR[j1, j0] = 0
        return
       
    rT = cuda.local.array((2,), f4)
    rT[0] = r[0] * T[0] + r[2] * T[1]
    rT[1] = r[1] * T[1]
    
    # Normalise rT:
    n = 0
    for i in range(DIM):
        n += rT[i] * rT[i]
    n = 1 / c_sqrt(n)
    for i in range(DIM):
        rT[i] *= n
   
    # Square of points is: x-s -> x+s
    buf0, buf1, buf2 = cuda.local.array((2,), f4), cuda.local.array((2,), f4), cuda.local.array((2,), f4)
    buf3, buf4, buf5 = cuda.local.array((1,), f4), cuda.local.array((6,), f4), cuda.local.array((6, 6), f4)
    limits = cuda.local.array((2,), f4)
    limits[0] = X[0] * W0[0] + X[1] * W0[1]
    limits[1] = min(limits[0] + rr, s)
    limits[0] = max(limits[0] - rr, -s)
       
    for i in range(DIM):
        Y[i] = limits[0] * W0[i]
    limits[0] = .1 * (limits[1] - limits[0])
    R[0] = 0
    for j0 in range(6):
        dR[j0] = 0
        for j1 in range(j0, 6):
            ddR[j0, j1] = 0
 
    count = 0
    for _ in range(11):
        for i in range(DIM):
            Y[i] += limits[0] * W0[i]
        __dRf2(I, X, r, Y, T, rT, n, buf0, buf1, buf2, buf3, buf4, buf5, order)
    
        R[0] += buf3[0]
        for j0 in range(6):
            dR[j0] += buf4[j0]
            for j1 in range(j0, 6):
                ddR[j0, j1] += buf5[j0, j1]
                
        count += 1
 
    scale = 10 * limits[0] / count
    R[0] *= scale
    for j0 in range(6):
        dR[j0] *= scale
        for j1 in range(j0, 6):
            ddR[j0, j1] *= scale
            ddR[j1, j0] = ddR[j0, j1]
 
 
@cuda.jit(device=True, inline=True)
def __drf3(I, x, r, Y, T, W0, W1, s, R, dR, ddR, order):
    DIM = 3
    X = cuda.local.array((3,), f4)
    for i in range(DIM):
        X[i] = x[i] - Y[i]
    R[0] = 0
    for j0 in range(10):
        dR[j0] = 0
        for j1 in range(10):
            ddR[j0, j1] = 0
  
    rr, n = __Radius(r), __projnorm(X, T, DIM)
    if n + rr < s:
        R[0] = __total(I, r)
        return
    elif n - rr > s:
        return
      
    rT = cuda.local.array((3,), f4)
    rT[0] = r[0] * T[0] + r[3] * T[1] + r[5] * T[2]
    rT[1] = r[1] * T[1] + r[4] * T[2]
    rT[2] = r[2] * T[2]
  
    # Normalise rT:
    n = 0
    for i in range(DIM):
        n += rT[i] * rT[i]
    n = 1 / c_sqrt(n)
    for i in range(DIM):
        rT[i] *= n
  
    # Square of points is: x-s -> x+s
    buf0, buf1, buf2 = cuda.local.array((3,), f4), cuda.local.array((3,), f4), cuda.local.array((3,), f4)
    buf3, buf4, buf5 = cuda.local.array((1,), f4), cuda.local.array((10,), f4), cuda.local.array((10, 10), f4)
    limits = cuda.local.array((2, 2), f4)
    # <x,w_i> \not\in (limits[i,0], limits[i,1]) => Rf(x)=0 or x\not\in [-s,s]
    limits[0, 0] = X[0] * W0[0] + X[1] * W0[1] + X[2] * W0[2]
    limits[0, 1] = min(limits[0, 0] + rr, s)
    limits[0, 0] = max(limits[0, 0] - rr, -s)
    limits[1, 0] = X[0] * W1[0] + X[1] * W1[1] + X[2] * W1[2]
    limits[1, 1] = min(limits[1, 0] + rr, s)
    limits[1, 0] = max(limits[1, 0] - rr, -s)
    
    for i in range(DIM):
        Y[i] = limits[0, 0] * W0[i] + limits[1, 0] * W1[i]
    for i in range(2):
        limits[i, 0] = .1 * (limits[i, 1] - limits[i, 0])
          
    count = 0
    for _ in range(11):
        for i in range(DIM):
            Y[i] += limits[0, 0] * W0[i]
        for __ in range(11):
            for i in range(DIM):
                Y[i] += limits[1, 0] * W1[i]
            __dRf3(I, X, r, Y, T, rT, n, buf0, buf1, buf2, buf3, buf4, buf5, order)
            R[0] += buf3[0]
            for j0 in range(10):
                dR[j0] += buf4[j0]
                for j1 in range(j0, 10):
                    ddR[j0, j1] += buf5[j0, j1]
              
            count += 1
        for i in range(DIM):
            Y[i] -= 11 * limits[1, 0] * W1[i]
          
    scale = 100 * limits[0, 0] * limits[1, 0] / count
    R[0] *= scale
    for j0 in range(10):
        dR[j0] *= scale
        for j1 in range(j0, 10):
            ddR[j0, j1] *= scale
            ddR[j1, j0] = ddR[j0, j1]


def L2derivs_RadProj(atom, Rad, C, order=2):
    S = Rad.range
    t, w, p = S.orientations, S.ortho, S.detector
    if hasattr(C, 'asarray'):
        C = C.asarray()
         
    block = -(-w[0].shape[0] // THREADS)
    f = empty((block,), dtype='f4')
    if len(w) == 1:
        Df, DDf = empty((6, block), dtype='f4'), empty((6, 6, block), dtype='f4')
    else:
        Df, DDf = empty((10, block), dtype='f4'), empty((10, 10, block), dtype='f4')
  
    if len(w) == 1:
        w = (w[0], empty((1, 1), dtype='f4'))
  
    if len(p) == 1:  # dim=2
        __L2derivs_RadProj_2D[block, THREADS](atom.I[0], atom.x[0], r2aniso(atom.r[:1], 2)[0],
                                              t, w[0], p[0], C, f, Df, DDf, order)
    else:
        if len(w) == 1:  # dim=2.5
            w = (w[0], empty((1, 1), dtype='f4'))
        __L2derivs_RadProj_3D[block, THREADS](atom.I[0], atom.x[0], r2aniso(atom.r[:1], 3)[0], t,
                                              w[0], w[1], p[0], p[1], C, f, Df, DDf, order)
  
    return f.sum(axis=-1), Df.sum(axis=-1), DDf.sum(axis=-1)
 
  
@cuda.jit
def __L2derivs_RadProj_2D(I, x, r, t, w0, p0, C, f, Df, DDf, order):
    '''
    Computes the 0, 1 and 2 order derivatives of |R(I,x,r)-C|^2/2 for a single atom
    '''
    buf = cuda.shared.array((THREADS, 28), f4)
  
    block, thread = cuda.blockIdx.x, cuda.threadIdx.x
    jj = cuda.grid(1)
    if jj >= t.shape[0]:
        for i0 in range(buf.shape[1]):
            buf[thread, i0] = 0
    else:
        F = cuda.local.array((1,), f4)
        DF = cuda.local.array((6,), f4)
        DDF = cuda.local.array((6, 6), f4)
        R = cuda.local.array((1,), f4)
        dR = cuda.local.array((6,), f4)
        ddR = cuda.local.array((6, 6), f4)
        Y = cuda.local.array((2,), f4)
      
        # Zero fill
        F[0] = 0
        for i in range(6):
            DF[i] = 0
            for j in range(6):
                DDF[i, j] = 0
     
        T, W0 = t[jj], w0[jj]
             
        s = abs(p0[1] - p0[0]) / 2
        dp = 1 / (p0[1] - p0[0])
        rr = __Radius(r)
     
        IP = x[0] * W0[0] + x[1] * W0[1]
        bin0 = max(0, int((IP - rr - p0[0]) * dp))
        bin1 = min(int((IP + rr - p0[0]) * dp) + 2, p0.size) 
     
        for k0 in range(bin0, bin1):
            for i in range(2):
                Y[i] = p0[k0] * W0[i]
      
            # Derivatives of atom
            __drf2(I, x, r, Y, T, W0, s, R, dR, ddR, order)
            # Derivatives of |R-C|^2/2
            R[0] = R[0] - C[jj, k0]
            F[0] += R[0] * R[0]
            for j0 in range(6):
                DF[j0] += R[0] * dR[j0]
                for j1 in range(j0, 6):
                    DDF[j0, j1] += dR[j0] * dR[j1] + R[0] * ddR[j0, j1]
       
        F[0] *= 0.5
        for i0 in range(DDF.shape[0] - 1):
            for i1 in range(i0 + 1, DDF.shape[0]):
                DDF[i1, i0] = DDF[i0, i1]
       
        # Sum over threads
        buf[thread, 0] = F[0]
        i = 1
        for i0 in range(6):
            buf[thread, i] = DF[i0]
            i += 1
        for i0 in range(6):
            for i1 in range(i0 + 1):
                buf[thread, i] = DDF[i0, i1]
                i += 1
    __GPU_reduce_n(buf, 28)
    if thread == 0:
        f[block] = buf[0, 0]
        i = 1
        for i0 in range(6):
            Df[i0, block] = buf[0, i]
            i += 1
        for i0 in range(6):
            for i1 in range(i0):
                DDf[i0, i1, block] = buf[0, i]
                DDf[i1, i0, block] = buf[0, i]
                i += 1
            DDf[i0, i0, block] = buf[0, i]
            i += 1

 
@cuda.jit
def __L2derivs_RadProj_3D(I, x, r, t, w0, w1, p0, p1, C, f, Df, DDf, order):
    '''
    Computes the 0, 1 and 2 order derivatives of |R(I,x,r)-C|^2/2 for a single atom
    '''
    buf = cuda.shared.array((THREADS, 66), f4)
  
    block, thread = cuda.blockIdx.x, cuda.threadIdx.x
    jj = cuda.grid(1)
    if jj >= t.shape[0]:
        for i0 in range(buf.shape[1]):
            buf[thread, i0] = 0
    else:
        F = cuda.local.array((1,), f4)
        DF = cuda.local.array((10,), f4)
        DDF = cuda.local.array((10, 10), f4)
        R = cuda.local.array((1,), f4)
        dR = cuda.local.array((10,), f4)
        ddR = cuda.local.array((10, 10), f4)
        Y = cuda.local.array((3,), f4)
        T = cuda.local.array((3,), f4)
      
        # Zero fill
        F[0] = 0
        for i in range(10):
            DF[i] = 0
            for j in range(10):
                DDF[i, j] = 0
     
        is3D = (w1.size == 1)
        W0, W1 = cuda.local.array((3,), f4), cuda.local.array((3,), f4)
        if is3D:
            T[0], T[1], T[2] = t[jj, 0], t[jj, 1], 0
            W0[0], W0[1], W0[2] = w0[jj, 0], w0[jj, 1], 0
            W1[0], W1[1], W1[2] = 0, 0, 1
        else:
            for i in range(3):
                T[i] = t[jj, i]
                W0[i] = w0[jj, i]
                W1[i] = w1[jj, i]
             
        s = abs(p0[1] - p0[0]) / 2
        dp = 1 / (p0[1] - p0[0])
        rr = __Radius(r)
     
        IP = x[0] * W0[0] + x[1] * W0[1] + x[2] * W0[2]
        bin00 = max(0, int((IP - rr - p0[0]) * dp))
        bin01 = min(int((IP + rr - p0[0]) * dp) + 2, p0.size) 
     
        IP = x[0] * W1[0] + x[1] * W1[1] + x[2] * W1[2]
        bin10 = max(0, int((IP - rr - p1[0]) * dp))
        bin11 = min(int((IP + rr - p1[0]) * dp) + 2, p1.size) 
         
        for k0 in range(bin00, bin01):
            for k1 in range(bin10, bin11):
                for i in range(3):
                    Y[i] = p0[k0] * W0[i] + p1[k1] * W1[i]
     
                # Derivatives of atom
                __drf3(I, x, r, Y, T, W0, W1, s, R, dR, ddR, order)
                # Derivatives of |R-C|^2/2
                R[0] = R[0] - C[jj, k0, k1]
                F[0] += R[0] * R[0]
                for j0 in range(10):
                    DF[j0] += R[0] * dR[j0]
                    for j1 in range(j0, 10):
                        DDF[j0, j1] += dR[j0] * dR[j1] + R[0] * ddR[j0, j1]
      
        F[0] *= 0.5
        for i0 in range(DDF.shape[0] - 1):
            for i1 in range(i0 + 1, DDF.shape[0]):
                DDF[i1, i0] = DDF[i0, i1]
  
        # Sum over threads
        buf[thread, 0] = F[0]
        i = 1
        for i0 in range(10):
            buf[thread, i] = DF[i0]
            i += 1
        for i0 in range(10):
            for i1 in range(i0 + 1):
                buf[thread, i] = DDF[i0, i1]
                i += 1
  
    __GPU_reduce_n(buf, 66)
    if thread == 0:
        f[block] = buf[0, 0]
        i = 1
        for i0 in range(10):
            Df[i0, block] = buf[0, i]
            i += 1
        for i0 in range(10):
            for i1 in range(i0):
                DDf[i0, i1, block] = buf[0, i]
                DDf[i1, i0, block] = buf[0, i]
                i += 1
            DDf[i0, i0, block] = buf[0, i]
            i += 1


if __name__ == '__main__':
    import numpy as np
    from time import time as tic
    
    class dict2obj:

        def __init__(self, **kwargs):
            for t in kwargs:
                setattr(self, t, kwargs[t])

    n, dim = 1, 3
    I = np.random.rand(n)
    x = 5 * np.random.rand(n, dim)
    r = np.random.rand(n, 3 * (dim - 1))
    r[:, :dim] += 1

    I[:] = 1
    x[...] = 2.5
    r[:, :dim], r[:, dim:] = 50, 0
    
    # Scan:
    t = np.linspace(0, 2 * np.pi, 50000)
    t = np.concatenate([np.cos(t).reshape(-1, 1), np.sin(t).reshape(-1, 1), 0 * t.reshape(-1, 1)], axis=1)
    w = t.copy()
    w[:, 0], w[:, 1] = -t[:, 1], t[:, 0]
    if dim == 3:
        w = [w, 0 * t]
        w[1][:, 2] = 1
    else:
        t, w = t[:, :2], (w[:, :2],)

    p = np.linspace(-5, 10, 100), np.linspace(-5, 10, 100)

    atom = dict2obj(I=I, x=x, r=r)
    Rad = dict2obj(range=dict2obj(orientations=t, ortho=w, detector=p[:dim - 1]))
    y = [np.arange(0, 5, 1 / 20), np.arange(0, 10, 1 / 20), np.arange(0, 5, 1 / 20)]
    
    u = zeros([Y.size for Y in y[:2]], dtype='f4')
    myR = zeros([t.shape[0], p[0].size] + ([] if dim == 2 else [p[1].size]), dtype='f4')
    
#     VolProj(atom, y, u)
    T = tic()
    RadProj(atom, Rad, myR)
#     print(myR[0].sum(), pi / 30 / r[0, :dim].prod(), myR.shape)
    print('Projection: ', tic() - T)
#     print(.5 * (myR ** 2).sum())
    
    T = tic()
    R, dR, ddR = L2derivs_RadProj(atom, Rad, 0 * myR, order=2)
    print('Derivatives: ', tic() - T)
#     print('Energy:', R)
    print('Gradient:', dR)
#     print('Hessian:', (abs(ddR)).round())
#     from numpy.linalg import eigvals
#     print('Eigenvalues:', eigvals(ddR))

    exit()

    from matplotlib import pyplot as plt
    plt.clf()
#     plt.imshow(u)
#     plt.imshow(R.T, aspect='auto',
#                 extent=[0, 360, p[0].min(), p[0].max()])
    plt.imshow(myR[1], aspect='auto',
                extent=[p[0].min(), p[0].max(), p[1].min(), p[1].max()])
    plt.pause(1)
    plt.show()
