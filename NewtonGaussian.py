'''
Created on 27 May 2018

@author: rob

We are calculating with:
g(y,\theta) = G(y,\theta)/|\theta|
where G(y,\theta) = exp(.5*J(y,\theta/|\theta|)^2-.5*|y|^2


'''
from numpy import sqrt, arange, random, maximum, minimum
from GaussDictCode.bin.manager import context
from GD_lib import _startup, norm, _step
from numpy.linalg.linalg import solve, eigvalsh


def linesearch(recon, data, max_iter, fidelity, reg, Radon, view=None,
               guess=None, tol=1e-4, min_iter=1, **kwargs):
    c, E, F, nAtoms, max_iter, plot = _startup(
        recon, data, max_iter, Radon, view, guess, 1, **kwargs)
    eps = 1e-4
    
    # Stepsizes
    h = [1] * nAtoms

    if guess is None:
        R = Radon(recon)
        F[0] = fidelity(R, data)
        E[0] = F[0] + reg(recon)
        n = nAtoms
    else:
        n = 1

    dim, iso = recon.space.dim, recon.space.isotropic

    random.seed(1)
    jj = 1
    while n <= nAtoms + 1:
        if n == nAtoms + 1:
            if guess is None or len(max_iter) == 2:
                break
            else:
                n = nAtoms
                max_iter = (max_iter[2], max_iter[1])
        elif guess is not None:
            recon[n - 1] = guess(data - Radon(recon[:n]),
                                 recon[n - 1])
#             recon.I[n - 1] = 0
            R = Radon(recon[:n])
            F[jj - 1] = fidelity(R, data)
            E[jj - 1] = F[jj - 1] + reg(recon[:n])
        for _ in range(max_iter[0]):
            BREAK = [True, True]
            tmp = random.permutation(arange(n))
            for j in tmp:
                for __ in range(max_iter[1]):
                    BREAK[1] = True
                    ___, df, ddf = Radon.L2_derivs(
                        recon[j], (data + Radon(recon[j]) - R))
#                     f += reg(recon[j])
                    df += reg.grad(recon[j])
                    try:
                        ddf += reg.hess(recon[j])[0]
                    except TypeError:
                        ddf += reg.hess(recon[j])
                        
#                     print(df * 0.021875000558793545 ** 2, '\n')
#                     print(ddf * 0.021875000558793545 ** 2)
#                     print(df, '\n')
#                     print(ddf)
#                     exit()
                    
                    H = 1 / h[j]
#                     H = max(0, H - eigvalsh(ddf).min())
                    for i in range(ddf.shape[0]):
                        ddf[i, i] += H

                    try:
                        dx = solve(ddf, -df)
                    except Exception as e:
                        raise e

                    R = _step(recon, Radon, dx, data,
                              E, F, dim, iso, j, jj, n, fidelity, reg)

                    if E[jj] > E[jj - 1]:
#                         c.set(recon.I[j:j + 1], I)
#                         c.set(recon.x[j], x)
#                         c.set(recon.r[j], r)
#                         R = Radon(recon[:n])
                        E[jj] = E[jj - 1]
                        if h[j] > 2 * eps:
                            BREAK[0] = False
                            BREAK[1] = False
                        h[j] /= 10
                    else:
                        if norm(dx) > tol:
                            BREAK[0] = False
                            BREAK[1] = False
                        h[j] *= 2
                    h[j] = min(max(h[j], eps), 1 / eps)

                    jj += 1
                    if BREAK[1]:
                        break
            if BREAK[0] and _ > min_iter:
                break
#             print('Volume: ', recon.I /
#                   recon.r[:, 0] / recon.r[:, 1] / recon.r[:, 2])
#             print('Pos: ', recon.x)
#             print('Rad: ', recon.r)
#             print('Amplitude: ', recon.I)

            plot(recon, R, E, F, n, _, jj)
        n += 1

    plot(recon, R, E, F, n, _, jj, True)
    return recon, E[:jj], F[:jj]


def linesearch_block(recon, data, max_iter, fidelity, reg, Radon, view=None,
                     guess=None, tol=1e-4, min_iter=1, **kwargs):
    c, E, F, nAtoms, max_iter, plot = _startup(
        recon, data, max_iter, Radon, view, guess, 3, **kwargs)
    eps = 1e-4

    # Stepsizes
    h = [[1] * nAtoms for _ in range(3)]

    iso = recon.space.isotropic
    if guess is None:
        R = Radon(recon)
        F[0] = fidelity(R, data)
        E[0] = F[0] + reg(recon)
        n = nAtoms
    else:
        n = 1

    dim = 'Ixr'
    random.seed(1)
    jj = 1
    while n <= nAtoms + 1:
        if n == nAtoms + 1:
            if guess is None or len(max_iter) == 2:
                break
            else:
                n = nAtoms
                max_iter = (max_iter[2], max_iter[1])
        elif guess is not None:
            recon[n - 1] = guess(data - Radon(recon[:n]),
                                 recon[n - 1])
            R = Radon(recon[:n])
            F[jj - 1] = fidelity(R, data)
            E[jj - 1] = F[jj - 1] + reg(recon[:n])
        for _ in range(max_iter[0]):
            BREAK = [True, True]
            tmp = random.permutation(arange(n))
            for j in tmp:
                for t in range(len(dim)):
                    for __ in range(max_iter[1]):
                        BREAK[1] = True
                        T = c.asarray(getattr(recon, dim[t])[j])
                        old = T.copy()

                        ___, df, ddf = Radon.L2_derivs(
                            recon[j], (data + Radon(recon[j]) - R))
                        df += reg.grad(recon[j])
                        ddf += reg.hess(recon[j])

                        H = 1 / h[t][j]
                        for i in range(ddf.shape[0]):
                            ddf[i, i] += H

                        if t == 0:
                            if ddf[0, 0] > 1e-8:
                                dx = -df[0] / ddf[0, 0]
                            else:
                                break
                            T += dx
                            c.set(recon.I[j:j + 1], max(0, T))
                        elif t == 1:
                            try:
                                dx = solve(
                                    ddf[1:dim + 1, 1:dim + 1], -df[1:dim + 1])
                            except Exception:
                                break
                            T += dx
                            c.set(recon.x[j], maximum(-.99,
                                                      minimum(.99, T)))
                        else:
                            t = 2
                            try:
                                dx = solve(
                                    ddf[dim + 1:, dim + 1:], -df[dim + 1:])
                            except Exception:
                                break
                            if iso:
                                T = abs(T + dx)
                            else:
                                T += dx
                                if T[0] < 0:
                                    T[0], T[3], T[5] = -\
                                        T[0], -T[3], -T[5]
                                if T[1] < 0:
                                    T[1], T[4] = -T[1], -T[4]
                                if T[2] < 0:
                                    T[2] = -T[2]
                            c.set(recon.r[j], T)

                        R = Radon(recon[:n])
                        F[jj] = fidelity(R, data)
                        E[jj] = F[jj] + reg(recon[:n])
                        if E[jj] > E[jj - 1]:
                            c.set(getattr(recon, dim[t])[j:j + 1], old)
                            R = Radon(recon[:n])
                            E[jj] = E[jj - 1]
                            if h[t][j] > 2 * eps:
                                BREAK[0] = False
                                BREAK[1] = False
                            h[t][j] /= 10
                        else:
                            h[t][j] *= 1.3
                            if norm(old - T) > tol * norm(old):
                                BREAK[0] = False
                                BREAK[1] = False
                        h[t][j] = min(max(h[t][j], eps), 1 / eps)

                        jj += 1
                        if BREAK[1]:
                            break
#                     print('Pos: ', recon.x)
#                     print('Rad: ', recon.r)
#                     print('Amplitude: ', recon.I)
            if BREAK[0] and _ > min_iter:
                break
#             print('Volume: ', recon.I /
#                   recon.r[:, 0] / recon.r[:, 1] / recon.r[:, 2])
#             print('Pos: ', recon.x)
#             print('Rad: ', recon.r)
#             print('Amplitude: ', recon.I)

            plot(recon, R, E, F, n, _, jj)
        n += 1

    plot(recon, R, E, F, n, _, jj, True)
    return recon, E[:jj], F[:jj]
