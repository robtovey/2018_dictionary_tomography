'''
Created on 24 Apr 2018

min \frac12|res-f|_2^2 + t*KL(f|R(atom))
atom(x) = A exp(-|S(x-M)|^2/2) 
R(atom,T)(y) = \frac{A}{\sqrt{2\pi}|ST|} exp(\frac{<ST,S(y-M)>^2}{2|ST|^2} - \frac{|S(y-M)|^2}{2})
log(R(atom,T)) = log(A)-.5log(2\pi|ST|^2) + \frac{<ST,S(y-M)>^2}{2|ST|^2} - \frac{|S(y-M)|^2}{2}

KL(f|g) = \int f log(f/g) - f + g
KL(f|R(atom,T)) = \int f [log(f)-1 - log(A) + .5log(2\pi|ST|^2) + |S(y-M)|^2/2-
                           \frac{<ST,S(y-M)>^2}{2|ST|^2}] + A(2\pi)^{n/2}/det(S)

\min_f \frac12|res-f|_2^2 + t*KL(f|R(atom,T))
    --> f-res + t*log(f/R(atoms)) = 0 
    <=> f+t*log(f) = res+t*log(R(atoms))
    This can be solved by a couple of Newton iterations:
        x = (c+t)/(1+t), x = x(c+t-t*log(x))/(x+t)

\min_A KL(f|R(atom))
    --> -\int f/A + (2\pi)^{n/2}/det(S) = 0 
    <=> A = (\int f)*det(S)/(2\pi)^{n/2}
    Alternative:
    --> -\int f/A + \int R(atom)/oldA = 0 
    <=> A = (oldA*\int f)/\int R(atom)

\min_M KL(f|R(atom))
    --> \int f[S^TS(M-y) - \frac{<ST,S(M-y)>}{|ST|^2} S^TST] = 0
    <=> \int f[S^TS - \frac{(T^TS^TS)^T(T^TS^TS)}{|ST|^2}](M-y) = 0
    <=> S^T[id - \frac{(ST)(ST)^T}{|ST|^2}]S \int f(M-y) = 0
    <=> S^T[id - \frac{(ST)(ST)^T}{|ST|^2}]S (M-(\int yf(y))/\int f(y)) = 0
    <=> M = (\int fy)/\int f    if more than 1 (indep) value of T used

Case 1: S is a scalar
    KL(f|R(atom)) = \int f [log(f)-1 - log(A) + .5log(2\pi S^2) + S^2|y-M|^2/2-
                               S^2\frac{<T,y-M>^2}{2}] + A(2\pi)^{n/2}/S^n
    \min_S KL(f|R(atom))
    --> \int f[1/S + S(|y-M|^2-<T,y-M>^2)] -nA(2\pi)^{n/2}/S^{n+1} = 0
    <=> S^{n+2} [\int f(|y-M|^2-<T,y-M>^2)] + S^n [\int f] = nA(2\pi)^{n/2}
    <=> S^n(S^2[\int f(|y-M|^2-<T,y-M>^2)] - [(n-1)\int f]) = 0
Case 2: S is upper triangular
    tbc...

@author: Rob Tovey
'''
from numpy import log, zeros, sqrt, exp, nansum, linalg, array,\
    warnings, maximum, concatenate
import numba
from code.bin.manager import context
from numpy.linalg.linalg import norm


def KL_2D(f, g):
    warnings.filterwarnings('ignore')
    return nansum(f * (log(f / g) - 1) + g)


def doKL_ProjGDStep_iso(res, atom, t, R):
    '''
    res = data - R(atom)
    min |data-f|^2 s.t. f = R(atom)
    f_{n+1} = argmin |data-f|^2 + t*KL(\bar f_n|f)
    atom_{n+1} = argmin KL(f_{n+1}|R(atom))
    \bar f_{n+1} = R(atom_{n+1})
    '''
    c = context()
    dim = atom.x.shape[1]
    iso = (atom.r.shape[1] == 1)
    res += R(atom)
    f = maximum(0, c.copy(res.array))
    if iso:
        Z = 0
    else:
        Z = (0, 0)
        c.set(atom.r[0, dim:], 0)
#     params = res.array.shape, res.array.dtype
#     const = 2 * pi * params[0][0] * res.array.size / res.space.volume()

#     from matplotlib import pyplot as plt
#     plt.subplot('131')
#     plt.imshow(res.array.T)

    c.set(atom.I[:], 1)
    c.set(atom.x[:], 0)
    c.set(atom.r[0, :3], 10)
    c.set(atom.r[0, 3:], 0)
    for _ in range(100):
        old = [c.copy(atom.r[Z]), c.copy(atom.x).reshape(-1),
               c.copy(atom.I[0])]
        if not iso:
            old[0] = 1 / old[0]
        If = MOM0(f)
        if If.sum() < 1e-8:
            if atom.r[Z] < 1e-4:
                return atom
            c.set(atom.r, c.mul(atom.r, .5))
            continue
        Ify = MOM1(f, res.space.detector, res.space.ortho)

        op, vec = orthog_op(If, Ify, res.space.orientations)
        try:
            M = linalg.solve(op, vec)
        except Exception:
            M = 0 * vec
        Ifyy = MOM2(f, res.space.detector, res.space.ortho, M)
        c.set(atom.x[:], M)

        # S^(n+2)[Ifyy] + S^n[If] -n A*const = 0
        # A = (If*S^n)/const
        S = sqrt(Ifyy / ((dim - 1) * If.sum()))
        if iso:
            c.set(atom.r[:], S)
        else:
            c.set(atom.r[0, :dim], 1 / S)

        RR = R(atom).array
        A = (If.sum() * atom.I[0]) / c.sum(RR)
        RR *= (A / atom.I[0])
        c.set(atom.I[:], A)

        if (norm(old[0] - S) > 1e-4 * norm(old[0])) or (norm(old[1] - M.reshape(-1)) > 1e-4 * norm(old[1])) or (norm(old[2] - A) > 1e-4 * norm(old[2])):
            f = maximum(0, t * res.array + (1 - t) * RR)
        else:
            break
#         print(M, S, A, KL_2D(f, R(atom).array))

#     plt.subplot('132')
#     plt.imshow(R(atom).array.T)
#     plt.subplot('133')
#     plt.imshow(f.T)
#     plt.show(block=True)
#     print(A, S, M)
    return atom


def doKL_LagrangeStep_iso(res, atom, t, R):
    '''
    res = data - R(atom)
    min |data-f|^2 + t*KL(f|R(atom))
    '''
    c = context()
    dim = atom.x.shape[1]
    iso = (atom.r.shape[1] == 1)
    res += R(atom)
    f = maximum(0, c.copy(res.array))
    if iso:
        Z = 0
    else:
        Z = (0, 0)
        c.set(atom.r[dim:], 0)

    for _ in range(100):
        old = [c.copy(atom.r[Z]), c.copy(atom.x).reshape(-1),
               c.copy(atom.I[0])]
        if not iso:
            old[0] = 1 / old[0]
        If = MOM0(f)
        if If.sum() < 1e-8:
            if atom.r[Z] < 1e-4:
                return atom
            if iso:
                c.set(atom.r, c.mul(atom.r, .5))
            else:
                c.set(atom.r, c.mul(atom.r, 2))
            continue
        Ify = MOM1(f, res.space.detector, res.space.ortho)

        op, vec = orthog_op(If, Ify, res.space.orientations)
        try:
            M = linalg.solve(op, vec)
        except Exception:
            M = 0 * vec
        Ifyy = MOM2(f, res.space.detector, res.space.ortho, M)
        c.set(atom.x[:], M)

        S = sqrt(Ifyy / ((dim - 1) * If.sum()))
        if iso:
            c.set(atom.r[:], S)
        else:
            c.set(atom.r[0, :dim], 1 / S)

        RR = R(atom).array
        A = (If.sum() * atom.I[0]) / c.sum(RR)
        RR *= (A / atom.I[0])
        c.set(atom.I[:], A)

        if (norm(old[0] - S) > 1e-4 * norm(old[0])) or (norm(old[1] - M.reshape(-1)) > 1e-4 * norm(old[1])) or (norm(old[2] - A) > 1e-4 * norm(old[2])):
            Newton_sino_2D(res.array, RR, t, f)
        else:
            break

    return atom


@numba.jit(["void(T[:,:],T[:,:],T,T[:,:])".replace('T', T) for T in ['f4', 'f8']], target='cpu', cache=True, nopython=True)
def Newton_sino_2D(a, b, t, x):
    '''
    Solve x + t*log(x) = C by Newton iteration
    where C = a + t*log(b)
    '''
    if t == 0:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x[i, j] = a[i, j]  # +0*log(b[i,j])
    else:
        # C = b*e^a~ b(1+a)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if b[i, j] <= 1e-15:
                    x[i, j] = 0
                else:
                    c = a[i, j] + t * log(b[i, j])
                    if c > 0:
                        c = c + t
                        X = c / (1 + t)
                        for _ in range(10):
                            X = X * (c - t * log(X)) / (X + t)
                        x[i, j] = X
                    else:
                        # x^te^x = c <=> x/t e^(x/t) = c^(1/t)/t
                        c = exp(c / t) / t
                        X = (sqrt(1 + 4 * c) - 1) / 2
                        for _ in range(10):
                            X = (X * X + c * exp(-X)) / (1 + X)
                        x[i, j] = X * t


def Newton_S_2D(a, b, c):
    '''
    Solve ax^4 + bx^2 = c by solving quadratic
    '''
    return sqrt((sqrt(b * b + 4 * a * c) - b) / (2 * a))


def MOM0(res):
    ''' return \int res '''
    return array([context().sum(res[i]) for i in range(res.shape[0])], dtype=res.dtype)


def MOM1(res, p, w):
    '''
    res = res(T,k) where T is an angle and k an n-1 dim. vector
    We compute \int res(T,k)y(k) dk where y(k) = \sum^n p_i(k)w_i
    This is linear in w_i so we just need \int res(T,k)p_i(k) for each i
    '''
    Sdim, Tlen = len(p), res.shape[0]
    # remember to keep this initiated 0!
    M = zeros((Tlen, Sdim, 1), dtype=res.dtype)

    if Sdim == 1:
        __MOM1_1D(res, p[0], M)
        M = M[:, 0] * w[0]
    elif Sdim == 2:
        __MOM1_2D(res, p[0], p[1], M)
        if len(w) == 1:
            M = concatenate((M[:, 0] * w[0], M[:, 1]), axis=1)
        else:
            M = M[:, 0] * w[0] + M[:, 1] * w[1]
    M.shape = (Tlen, Sdim + 1)
    return M


@numba.jit(["void(T[:,:],T[:],T[:,:,:])".replace('T', T) for T in ['f4', 'f8']], target='cpu', cache=True, nopython=True)
def __MOM1_1D(res, p, M):
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            M[i, 0, 0] += res[i, j] * p[j]


@numba.jit(["void(T[:,:,:],T[:],T[:],T[:,:,:])".replace('T', T) for T in ['f4', 'f8']], target='cpu', cache=True, nopython=True)
def __MOM1_2D(res, p0, p1, M):
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            for k in range(res.shape[1]):
                M[i, 0, 0] += res[i, j, k] * p0[j]
                M[i, 1, 0] += res[i, j, k] * p1[k]


def orthog_op(If, Ify, t):
    # op = sum_i If[i](1-t[i]t[i]^T)
    # vec = sum_i (1-t[i]t[i]^T)Ify[i]
    op = zeros((Ify.shape[1], Ify.shape[1]), dtype=t.dtype)
    vec = zeros((Ify.shape[1], ), dtype=t.dtype)

    if t.shape[1] == 2:
        if Ify.shape[1] == 2:
            __orthog_op_2D(If, Ify, t, op, vec)
        else:
            __orthog_op_2p5D(If, Ify, t, op, vec)
    elif t.shape[1] == 3:
        __orthog_op_3D(If, Ify, t, op, vec)

    return op, vec


@numba.jit(["void(T[:],T[:,:],T[:,:],T[:,:],T[:])".replace('T', T) for T in ['f4', 'f8']], target='cpu', cache=True, nopython=True)
def __orthog_op_2D(If, Ify, t, op, vec):
    for i in range(t.shape[0]):
        s = 0
        for j1 in range(2):
            op[j1, j1] += (1 - t[i, j1] * t[i, j1]) * If[i]
            for j2 in range(j1 + 1, 2):
                op[j1, j2] -= t[i, j1] * t[i, j2] * If[i]
            s += t[i, j1] * Ify[i, j1]
        for j1 in range(2):
            vec[j1] += Ify[i, j1] - s * t[i, j1]

    for j1 in range(2 - 1):
        for j2 in range(j1 + 1, 2):
            op[j2, j1] = op[j1, j2]


@numba.jit(["void(T[:],T[:,:],T[:,:],T[:,:],T[:])".replace('T', T) for T in ['f4', 'f8']], target='cpu', cache=True, nopython=True)
def __orthog_op_2p5D(If, Ify, t, op, vec):
    for i in range(t.shape[0]):
        s = 0
        for j1 in range(2):
            op[j1, j1] += (1 - t[i, j1] * t[i, j1]) * If[i]
            for j2 in range(j1 + 1, 2):
                op[j1, j2] -= t[i, j1] * t[i, j2] * If[i]
            s += t[i, j1] * Ify[i, j1]
        op[2, 2] += If[i]

        for j1 in range(2):
            vec[j1] += Ify[i, j1] - s * t[i, j1]
        vec[2] += Ify[i, 2]

    for j1 in range(3 - 1):
        for j2 in range(j1 + 1, 3):
            op[j2, j1] = op[j1, j2]


@numba.jit(["void(T[:],T[:,:],T[:,:],T[:,:],T[:])".replace('T', T) for T in ['f4', 'f8']], target='cpu', cache=True, nopython=True)
def __orthog_op_3D(If, Ify, t, op, vec):
    for i in range(t.shape[0]):
        s = 0
        for j1 in range(3):
            op[j1, j1] += (1 - t[i, j1] * t[i, j1]) * If[i]
            for j2 in range(j1 + 1, 3):
                op[j1, j2] -= t[i, j1] * t[i, j2] * If[i]
            s += t[i, j1] * Ify[i, j1]
        for j1 in range(3):
            vec[j1] += Ify[i, j1] - s * t[i, j1]

    for j1 in range(3 - 1):
        for j2 in range(j1 + 1, 3):
            op[j2, j1] = op[j1, j2]


def MOM2(res, p, w, M=None):
    '''
    res = res(T,k) where T is an angle and k an n-1 dim. vector
    We compute \int res(T,k)(|y-M|^2-<T,y-M>^2)dk where y(k) = \sum^n p_i(k)w_i
    Note |y-M|^2-<T,y-M>^2 = \sum <w_i,y-M>^2 = \sum (p_i(k)-<w_i,M>)^2
    '''
    Sdim, Tlen = len(p), res.shape[0]
    if M is None:
        M = zeros((Sdim, Tlen), dtype=res.dtype)
    else:
        M.shape = [1, Sdim + 1]
        if w[0].shape[1] == M.shape[1]:
            M = [(w[i] * M).sum(1) for i in range(Sdim)]
        else:
            # dim=2.5, w[1] = [0,0,1]
            M = [(w[0] * M[:, :2]).sum(1), 0 * w[0][:, 0] + M[:, 2]]
        # M[i][j] = <w_i[j],M>

    ans = zeros(1, dtype=res.dtype)
    if Sdim == 1:
        __MOM2_1D(res, p[0], M[0], ans)
    elif Sdim == 2:
        __MOM2_2D(res, p[0], p[1], M[0], M[1], ans)
    return ans


@numba.jit(["void(T[:,:],T[:],T[:],T[:])".replace('T', T) for T in ['f4', 'f8']], target='cpu', cache=True, nopython=True)
def __MOM2_1D(res, p, M, ans):
    s, tmp = 0, 0
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            tmp = p[j] - M[i]
            s += res[i, j] * tmp * tmp
    ans[0] = s


@numba.jit(["void(T[:,:,:],T[:],T[:],T[:],T[:],T[:])".replace('T', T) for T in ['f4', 'f8']], target='cpu', cache=True, nopython=True)
def __MOM2_2D(res, p0, p1, M0, M1, ans):
    s, tmp0, tmp1 = 0, 0, 0
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            tmp0 = p0[j] - M0[i]
            for k in range(res.shape[2]):
                tmp1 = p1[k] - M1[i]
                s += res[i, j, k] * (tmp0 * tmp0 + tmp1 * tmp1)
    ans[0] = s


@numba.jit(["T(T,b1)".replace('T', T) for T in ['f4', 'f8']], target='cpu', cache=True, nopython=True)
def __brentRoots(a, big):
    # Root finding for the polynomial: f(x) = ax^3-2x+1
    # First root is in [0,sqrt(1/a)]
    # Second root is in [sqrt(1/a),sqrt(2/a)]
    if big:
        x2 = sqrt(2 / a)
        x0 = sqrt(1 / a)
        x1 = sqrt(3 / (2 * a))
    else:
        x2 = sqrt(1 / a)
        x0, x1 = 0, x2 / 2

    f0 = 1 + x0 * (a * x0 * x0 - 3)
    f1 = 1 + x1 * (a * x1 * x1 - 3)
    f2 = 1 + x2 * (a * x2 * x2 - 3)
    if f2 > 0:
        return -1
    for _ in range(10):
        R, S, T = f1 / f2, f1 / f0, f0 / f2
        x = x1 + S * (T * (R - T) * (x2 - x1) - (1 - R) * (x1 - x0)) / \
            ((T - 1) * (S - 1) * (R - 1))
        fx = 1 + x * (a * x * x - 3)
        if abs(fx) < 1e-8:
            return x
        elif fx > 0:
            if f1 < 0:
                x0 = x
                f0 = fx
                x2 = x1
                f2 = f1
            elif x1 > x:
                x0 = x1
                f0 = f1
                # x2 = x2
            else:
                x0 = x
                f0 = fx
                # x2 = x2
        else:
            if f1 > 0:
                x0 = x1
                f0 = f1
                x2 = x
                f2 = fx
            elif x1 < x:
                x2 = x1
                f2 = f1
                # x0 = x0
            else:
                x2 = x
                f2 = x
                # x0 = x0

        # Linear interpolation:
        # f(x) = f(x0) + (x-x0)/(x2-x0)f(x2)
        # x1 = x0 -(f(x0)*(x2-x0)/f(x2))
        x1 = x0 - f0 * (x2 - x0) / f2
#         # Newton: x1 = x - f(x)/f'(x)
#         x1 = x - fx / (3 * a * x * x - 3)
#         # Midpoint:
#         x1 = .5 * (x0 + x2)
        f1 = 1 + x1 * (a * x1 * x1 - 3)

    if abs(fx) < abs(f1):
        return x
    else:
        return x1
