'''
Created on 27 May 2018

@author: rob

We are calculating with:
g(y,\theta) = G(y,\theta)/|\theta|
where G(y,\theta) = exp(.5*J(y,\theta/|\theta|)^2-.5*|y|^2


'''
from numpy import sqrt, arange, random, maximum, minimum
from code.bin.manager import context
from GD_lib import _startup, norm
from numpy.linalg.linalg import solve, eigvalsh


def __derivs(recon):
    dim = recon.space.dim
    iso = recon.space.isotropic
    if dim == 2:
        if iso:
            from code.bin.NewtonGaussian2D import derivs_iso as d
        else:
            from code.bin.NewtonGaussian2D import derivs_aniso as d
    else:
        if iso:
            from code.bin.NewtonGaussian3D import derivs_iso as d
        else:
            from code.bin.NewtonGaussian3D import derivs_aniso as d

    return dim, iso, d


def linesearch(recon, data, max_iter, fidelity, reg, Radon, view,
               guess=None, gt=None, RECORD=None, tol=1e-4, min_iter=1):
    if RECORD is None:
        c, E, F, nAtoms, max_iter, plotter, plt = _startup(
            recon, data, max_iter, view, guess, gt, RECORD, 1)
    else:
        c, E, F, nAtoms, max_iter, plotter, plt, writer = _startup(
            recon, data, max_iter, view, guess, gt, RECORD, 1)
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

    dim, iso, derivs = __derivs(recon)

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
                for __ in range(max_iter[1]):
                    BREAK[1] = True
                    I = c.asarray(recon.I[j:j + 1]).copy()
                    x = c.asarray(recon.x[j]).copy()
                    r = c.asarray(recon.r[j]).copy()

                    ___, df, ddf = derivs(
                        recon[j], Radon, (data + Radon(recon[j]) - R))
#                     f += reg(recon[j])
                    df += reg.grad(recon[j])
                    ddf += reg.hess(recon[j])

                    H = 1 / h[j]
#                         H = max(0, H - eigvalsh(ddf).min())
                    for i in range(ddf.shape[0]):
                        ddf[i, i] += H

                    try:
                        dx = solve(ddf, -df)
                    except Exception:
                        break
#                         c.set(recon.I[j:j + 1], I + dx[0])
                    c.set(recon.x[j], x + dx[1:dim + 1])
#                     c.set(recon.r[j], r + dx[dim + 1:])

                    c.set(recon.I[j:j + 1], max(0, I + dx[0]))
#                         c.set(recon.x[j], maximum(-.99,
#                                                   minimum(.99, x + dx[1:4])))
                    if iso:
                        rr = abs(r + dx[dim + 1])
                    else:
                        rr = r + dx[dim + 1:]
                        if rr[0] < 0:
                            rr[0], rr[3], rr[5] = -rr[0], -rr[3], -rr[5]
                        if rr[1] < 0:
                            rr[1], rr[4] = -rr[1], -rr[4]
                        if rr[2] < 0:
                            rr[2] = -rr[2]
                    c.set(recon.r[j], rr)

                    R = Radon(recon[:n])
                    F[jj] = fidelity(R, data)
                    E[jj] = F[jj] + reg(recon[:n])
                    if E[jj] > E[jj - 1]:
                        c.set(recon.I[j:j + 1], I)
                        c.set(recon.x[j], x)
                        c.set(recon.r[j], r)
                        R = Radon(recon[:n])
                        E[jj] = E[jj - 1]
                        if h[j] > 2 * eps:
                            BREAK[0] = False
                            BREAK[1] = False
                        h[j] /= 10
                    else:
                        if norm(dx) > tol * sqrt(I * I + (x * x).sum() + (r * r).sum()):
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

            plotter(recon[:n], R, E, F, jj)
            try:
                plt.pause(.1)
                if RECORD is not None:
                    writer.grab_frame()
                    print(n / nAtoms, _ / max_iter[0], E[jj - 1])
            except NameError:
                exit()
        n += 1

    print('Reconstruction Finished', jj, F[:jj].min())
    if RECORD is not None:
        writer.finish()
    plt.show(block=True)
    return recon, E[:jj], F[:jj]


def linesearch_block(recon, data, max_iter, fidelity, reg, Radon, view,
                     guess=None, gt=None, RECORD=None, tol=1e-4, min_iter=1):
    if RECORD is None:
        c, E, F, nAtoms, max_iter, plotter, plt = _startup(
            recon, data, max_iter, view, guess, gt, RECORD, 3)
    else:
        c, E, F, nAtoms, max_iter, plotter, plt, writer = _startup(
            recon, data, max_iter, view, guess, gt, RECORD, 3)
    eps = 1e-4

    # Stepsizes
    h = [[1] * nAtoms for _ in range(3)]

    _, iso, derivs = __derivs(recon)
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

                        ___, df, ddf = derivs(
                            recon[j], Radon, (data + Radon(recon[j]) - R))
                        df += reg.grad(recon[j])
                        ddf += reg.hess(recon[j])

                        H = 1 / h[t][j]
                        for i in range(ddf.shape[0]):
                            ddf[i, i] += H

                        if t == 0:
                            if ddf[0, 0] > 1e-8:
                                dx = - df[0] / ddf[0, 0]
                            else:
                                break
                            T += dx
                            c.set(recon.I[j:j + 1], max(0, T))
                        elif t == 1:
                            try:
                                dx = solve(
                                    ddf[1:dim + 1, 1:dim + 1], - df[1:dim + 1])
                            except Exception:
                                break
                            T += dx
                            c.set(recon.x[j], maximum(-.99,
                                                      minimum(.99, T)))
                        else:
                            t = 2
                            try:
                                dx = solve(
                                    ddf[dim + 1:, dim + 1:], - df[dim + 1:])
                            except Exception:
                                break
                            if iso:
                                T = abs(T + dx)
                            else:
                                T += dx
                                if T[0] < 0:
                                    T[0], T[3], T[5] = - \
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

            plotter(recon[:n], R, E, F, jj)
            try:
                plt.pause(.1)
                if RECORD is not None:
                    writer.grab_frame()
                    print(n / nAtoms, _ / max_iter[0], E[jj - 1])
            except NameError:
                exit()
        n += 1

    print('Reconstruction Finished', jj, F[:jj].min())
    if RECORD is not None:
        writer.finish()
    plt.show(block=True)
    return recon, E[:jj], F[:jj]
