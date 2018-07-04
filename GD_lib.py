'''
Created on 21 May 2018

@author: Rob Tovey
'''
from numpy import prod, zeros, sqrt, arange, random, log10, insert
from code.bin.manager import context


def linesearch(recon, data, max_iter, fidelity, reg, Radon, view,
               guess=None, gt=None, RECORD=None, tol=1e-4, min_iter=1):
    if RECORD is None:
        c, E, F, nAtoms, max_iter, plotter, plt = _startup(
            recon, data, max_iter, view, guess, gt, RECORD, 1)
    else:
        c, E, F, nAtoms, max_iter, plotter, plt, writer = _startup(
            recon, data, max_iter, view, guess, gt, RECORD, 1)
    eps = 1e-8

    # Stepsizes
    h = [1] * nAtoms
    dim = 'Ixr'

    if guess is None:
        R = Radon(recon)
        F[0] = fidelity(R, data)
        E[0] = F[0] + reg(recon)
        n = nAtoms
    else:
        n = 1

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
                    grad = [Radon.grad(recon[j], d, fidelity.grad(
                        R, data, axis=0)) + reg.grad(recon[j], axis=d)
                        for d in dim]
                    T = [c.asarray(getattr(recon, d)[j]) for d in dim]
                    old = [t.copy() for t in T]
                    for t in range(3):
                        c.set(getattr(recon, dim[t])[j:j + 1],
                              T[t] - h[j] * grad[t])

                    R = Radon(recon[:n])
                    F[jj] = fidelity(R, data)
                    E[jj] = F[jj] + reg(recon[:n])
                    if E[jj] > E[jj - 1]:
                        for t in range(3):
                            c.set(getattr(recon, dim[t])[j:j + 1], old[t])
                        E[jj] = E[jj - 1]
                        h[j] /= 10
                    else:
                        h[j] *= 2
                        for t in range(3):
                            if norm(old[t] - T[t]) > tol * norm(old[t]):
                                BREAK[0] = False
                                BREAK[1] = False
                    h[j] = max(h[j], eps)

                    jj += 1
                    if BREAK[1]:
                        break
            if BREAK[0] and _ > min_iter:
                break

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
                     dim='xrI', guess=None, gt=None, RECORD=None, tol=1e-4, min_iter=1):
    dim = dim.lower()
    tmp = ''
    if 'x' in dim:
        tmp += 'x'
    if 'r' in dim:
        tmp += 'r'
    if 'i' in dim:
        tmp += 'I'
    dim = tmp

    if RECORD is None:
        c, E, F, nAtoms, max_iter, plotter, plt = _startup(
            recon, data, max_iter, view, guess, gt, RECORD, len(dim))
    else:
        c, E, F, nAtoms, max_iter, plotter, plt, writer = _startup(
            recon, data, max_iter, view, guess, gt, RECORD, len(dim))
    eps = 1e-8

    # Stepsizes
    h = [[1] * nAtoms for _ in dim]

    if guess is None:
        R = Radon(recon)
        F[0] = fidelity(R, data)
        E[0] = F[0] + reg(recon)
        n = nAtoms
    else:
        n = 1

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
#             if n > 1:
#                 for t in range(len(dim)):
#                     h[t][n - 1] = h[t][n - 2]
        for _ in range(max_iter[0]):
            BREAK = [True, True]
            tmp = random.permutation(arange(n))
            for j in tmp:
                for __ in range(max_iter[1]):
                    BREAK[1] = True
                    for t in range(len(dim)):
                        #                     for t in [1]:
                        #                         print('Here: ', dim[t])
                        grad = Radon.grad(recon[j], dim[t], fidelity.grad(
                            R, data, axis=0)) + reg.grad(recon[j], axis=dim[t])
                        T = c.asarray(getattr(recon, dim[t])[j])
                        old = T.copy()
                        tmp = 1 / max(norm(grad), eps)
                        h[t][j] = min(h[t][j], tmp)
                        T -= h[t][j] * grad[0]
                        c.set(getattr(recon, dim[t])[j:j + 1], T)

                        R = Radon(recon[:n])
                        F[jj] = fidelity(R, data)
                        E[jj] = F[jj] + reg(recon[:n])
                        if E[jj] > E[jj - 1]:
                            c.set(getattr(recon, dim[t])[j:j + 1], old)
                            E[jj] = E[jj - 1]
                            h[t][j] /= 2
                        else:
                            h[t][j] *= 1.1
                            if norm(old - T) > tol * norm(old):
                                BREAK[0] = False
                                BREAK[1] = False
                        h[t][j] = max(h[t][j], eps)

                        jj += 1
                    if BREAK[1]:
                        break
            if BREAK[0] and _ > min_iter:
                break

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


def norm(x):
    return sqrt((x * x).sum())


def _startup(recon, data, max_iter, view, guess, gt, RECORD, L):
    '''
    recon is element of atom space
    L is number of energy recordings per iter

    '''
    if RECORD is not None:
        import matplotlib
        matplotlib.use('Agg')
    from matplotlib import pyplot as plt, animation as mv

    c = context()

    if not hasattr(max_iter, '__iter__'):
        max_iter = [max_iter, 1]
    elif len(max_iter) == 1:
        max_iter = max_iter.copy()
        max_iter.append(1)

    nAtoms = len(recon)

    if guess is None:
        tmp = nAtoms
    else:
        tmp = int(round(nAtoms * (nAtoms + 1) / 2)) + 1
    E = zeros(prod(max_iter) * tmp * L + 1)
    F = zeros(prod(max_iter) * tmp * L + 1)

    dim = recon.space.dim
#     iso = recon.space.isotropic
    if dim == 2:
        fig, ax = plt.subplots(2, 3, figsize=(20, 10))
        if gt is None:
            def plotter(recon, R, E, F, jj):
                n = max_iter[1]
                return _plot2D(ax, view(recon), R, data, E[::n], F[::n], jj // (n * L))
        else:
            def plotter(recon, R, E, F, jj):
                n = max_iter[1]
                return _plot2DwithGT(ax, view(recon), gt, R, data, E[::n], F[::n], jj // (n * L))
    else:
        fig, ax = plt.subplots(2, 4, figsize=(20, 10))
        if gt is None:
            def plotter(recon, R, E, F, jj):
                n = max_iter[1]
                return _plot3D(ax, view(recon), R, data, E[::n], F[::n], jj // (n * L), len(recon))
        else:
            def plotter(recon, R, E, F, jj):
                n = max_iter[1]
                return _plot3DwithGT(ax, view(recon), gt, R, data, E[::n], F[::n], jj // (n * L), len(recon))
    if RECORD is not None:
        writer = mv.writers['ffmpeg'](fps=5, metadata={'title': RECORD})
        writer.setup(fig, RECORD + '.mp4', dpi=100)

        return c, E, F, nAtoms, max_iter, plotter, plt, writer
    return c, E, F, nAtoms, max_iter, plotter, plt


def _plot2D(ax, recon, R, data, E, F, jj):
    for a in ax.reshape(-1):
        a.clear()

    recon.plot(ax[0, 0])
    ax[0, 0].set_title('Reconstruction')
    (R - data).plot(ax[0, 1], aspect='auto')
    ax[0, 1].set_title('Sinogram Residual')
    R.plot(ax[1, 0], aspect='auto')
    ax[1, 0].set_title('Reconstructed Sinogram')
    data.plot(ax[1, 1], aspect='auto')
    ax[1, 1].set_title('Data Sinogram')
    ax[0, 2].plot(log10(F[:jj]))
    ax[0, 2].set_title('log10(Fidelity)')
    ax[1, 2].plot(log10(E[:jj]))
    ax[1, 2].set_title('log10(Energy)')


def _plot2DwithGT(ax, recon, gt, R, data, E, F, jj):
    for a in ax.reshape(-1):
        a.clear()

    recon.plot(ax[0, 0])
    ax[0, 0].set_title('Reconstruction')
    gt.plot(ax[0, 1])
    ax[0, 1].set_title('GT')
    R.plot(ax[1, 0], aspect='auto')
    ax[1, 0].set_title('Reconstructed Sinogram')
#     data.plot(ax[1, 1], aspect='auto')
#     ax[1, 1].set_title('Data Sinogram')
    (R - data).plot(ax[1, 1], aspect='auto')
#     from numpy import minimum
#     ax[1, 1].imshow(minimum(0, (R - data).array), aspect='auto')
    ax[1, 1].set_title('Sinogram Error')
    ax[0, 2].plot(log10(F[:jj]))
    ax[0, 2].set_title('log10(Fidelity)')
    ax[1, 2].plot(log10(E[:jj]))
    ax[1, 2].set_title('log10(Energy)')


def _plot3D(ax, recon, R, data, E, F, jj, nAtoms):
    for a in ax.reshape(-1):
        a.clear()

    recon.plot(ax[0, 0], Slice=[slice(None), slice(
        None), round(recon.shape[2] / 2)])
    ax[0, 0].set_title('Recon, slice from top')
    recon.plot(ax[1, 0], Slice=[slice(None), round(
        recon.shape[1] / 2), slice(None)])
    ax[1, 0].set_title('Recon, slice from front')

    Slice = int(R.shape[0] / 2)
    clim = [0, data.array[Slice].max()]
    data.plot(ax[0, 1], Slice=Slice, clim=clim)
    ax[0, 1].set_title('Middle Data Projection')
    R.plot(ax[1, 1], Slice=Slice, clim=clim)
    ax[1, 1].set_title('Middle Recon Projection')

    Slice = (jj // (3 * nAtoms)) % R.shape[0]
    clim = [0, data.array[Slice].max()]
    data.plot(ax[0, 2], Slice=Slice, clim=clim)
    ax[0, 2].set_title('Data Projection')
    R.plot(ax[1, 2], Slice=Slice, clim=clim)
    ax[1, 2].set_title('Recon Projection')

    ax[0, 3].plot(log10(F[:jj]))
    ax[0, 3].set_title('log10(Fidelity)')
    ax[1, 3].plot(log10(E[:jj]))
    ax[1, 3].set_title('log10(Energy)')


def _plot3DwithGT(ax, recon, gt, R, data, E, F, jj, nAtoms):
    for a in ax.reshape(-1):
        a.clear()

    n = (round(gt.shape[2] / 2), round(gt.shape[1] / 2))
    cax = (gt.array[:, :, n[0]].max(), gt.array[:, n[1]].max())
    gt.plot(ax[0, 0], Slice=[slice(None), slice(
        None), n[0]], vmin=0, vmax=cax[0])
    ax[0, 0].set_title('GT, slice from top')
    gt.plot(ax[0, 1], Slice=[slice(None), n[1],
                             slice(None)], vmin=0, vmax=cax[1])
    ax[0, 1].set_title('GT, slice from front')
    recon.plot(ax[1, 0], Slice=[slice(None), slice(
        None), n[0]], vmin=0, vmax=cax[0])
    ax[1, 0].set_title('Recon, slice from top')
    recon.plot(ax[1, 1], Slice=[slice(None), n[1],
                                slice(None)], vmin=0, vmax=cax[1])
    ax[1, 1].set_title('Recon, slice from front')

    Slice = (jj // (3 * nAtoms)) % R.shape[0]
    clim = [0, data.array[Slice].max()]
    data.plot(ax[0, 2], Slice=Slice, clim=clim)
    ax[0, 2].set_title('Data Projection')
    R.plot(ax[1, 2], Slice=Slice, clim=clim)
    ax[1, 2].set_title('Recon Projection')

    ax[0, 3].plot(log10(F[:jj]))
    ax[0, 3].set_title('log10(Fidelity)')
    ax[1, 3].plot(log10(E[:jj]))
    ax[1, 3].set_title('log10(Energy)')
