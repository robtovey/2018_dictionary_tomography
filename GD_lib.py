'''
Created on 21 May 2018

@author: Rob Tovey
'''
from numpy import prod, zeros, sqrt, arange, random, maximum, minimum
from GaussDictCode.bin.manager import context
from skimage import measure


def linesearch(recon, data, max_iter, fidelity, reg, Radon, view=None,
               guess=None, tol=1e-4, min_iter=1, **kwargs):
    c, E, F, nAtoms, max_iter, plot = _startup(
        recon, data, max_iter, Radon, view, guess, 1, **kwargs)
    eps = 1e-8

    # Stepsizes
    h = [1e-2] * nAtoms
    dim, iso = recon.space.dim, recon.space.isotropic

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
                    ___, df = Radon.L2_derivs(recon[j], (data + Radon(recon[j]) - R), order=1)
                    df += reg.grad(recon[j])

                    R = _step(recon, Radon, R, -df, data,
                              E, F, dim, iso, j, jj, n, fidelity, reg)

                    if E[jj] > E[jj - 1]:
                        E[jj] = E[jj - 1]
                        if h[j] > 2 * eps:
                            BREAK[0] = False
                            BREAK[1] = False
                        h[j] /= 10
                    else:
                        if norm(df) > tol:
                            BREAK[0] = False
                            BREAK[1] = False
                        h[j] *= 2
                    h[j] = min(max(h[j], eps), 1 / eps)

                    jj += 1
                    if BREAK[1]:
                        break
            if BREAK[0] and _ > min_iter:
                break

            plot(recon, R, E, F, n, _, jj)
        n += 1

    plot(recon, R, E, F, n, _, jj, True)
    return recon, E[:jj], F[:jj]


def linesearch_block(recon, data, max_iter, fidelity, reg, Radon, view=None,
                     dim='xrI', guess=None, tol=1e-4, min_iter=1, **kwargs):
    dim = dim.lower()
    tmp = ''
    if 'x' in dim:
        tmp += 'x'
    if 'r' in dim:
        tmp += 'r'
    if 'i' in dim:
        tmp += 'I'
    dim = tmp

    c, E, F, nAtoms, max_iter, plot = _startup(
        recon, data, max_iter, Radon, view, guess, len(dim), **kwargs)
    eps = 1e-8

    # Stepsizes
    h = [[1e-2] * nAtoms for _ in dim]
    
    Dim = recon.space.dim

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
                        ___, df = [thing[0] for thing in 
                                   Radon.L2_derivs(recon[j], (data + Radon(recon[j]) - R), order=1)]
                        if t == 0:
                            grad = df[0]
                        elif t == 1:
                            grad = df[1:1 + Dim]
                        else:
                            grad = df[1 + Dim:]

                        # TODO: integrate regulariser and fidelity again
                        grad += reg.grad(recon[j], axis=dim[t])
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

            plot(recon, R, E, F, n, _, jj)
        n += 1

    plot(recon, R, E, F, n, _, jj, True)
    return recon, E[:jj], F[:jj]


def norm(x): return sqrt((x * x).sum())


def _startup(recon, data, max_iter, Radon, view, guess, L, gt=None, RECORD=None, thresh=None, angles=None):
    '''
    recon is element of atom space
    L is number of energy recordings per iter

    '''
    from time import gmtime, strftime
    print('Start time is: ', strftime("%H:%M", gmtime()))
    if RECORD is not None:
        __makeVid(stage=0)
    from matplotlib import pyplot as plt

    c = context()

    if not hasattr(max_iter, '__iter__'):
        max_iter = [max_iter, 1]
    elif len(max_iter) == 1:
        max_iter = max_iter.copy()
        max_iter.append(1)

    nAtoms = len(recon)

    if view is None:
        view = Radon.discretise
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
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        fig, ax = plt.subplots(2, 4, figsize=(20, 10))
        ax[0, 0].remove()
        ax[0, 0] = fig.add_subplot(241, projection='3d')
        ax[0, 0].set_axis_off()
        ax[1, 0].remove()
        ax[1, 0] = fig.add_subplot(245, projection='3d')
        ax[1, 0].set_axis_off()
        if gt is None:
            if thresh is None:
                thresh = data.asarray().max() / 100
            if angles is None:
                angles = ((20, 45), (20, 135))

            def plotter(recon, R, E, F, jj):
                n = max_iter[1]
                return _plot3D(ax, view(recon), R, data, E[::n], F[::n], jj // (n * L), len(recon), thresh, angles)

        else:
            ax[0, 1].remove()
            ax[0, 1] = fig.add_subplot(242, projection='3d')
            ax[0, 1].set_axis_off()
            ax[1, 1].remove()
            ax[1, 1] = fig.add_subplot(246, projection='3d')
            ax[1, 1].set_axis_off()
            if thresh is None:
                thresh = gt.asarray().max() / 10
            if angles is None:
                angles = ((0, 0), (0, 90))

            def plotter(recon, R, E, F, jj):
                n = max_iter[1]
                return _plot3DwithGT(ax, view(recon), gt, R, data, E[::n], F[::n], jj // (n * L), len(recon), thresh, angles)

    if RECORD is not None:
        writer = __makeVid(fig, RECORD, stage=1)

        def do_plot(recon, r, e, f, n, i, jj, end=False):
            if end:
                print('Reconstruction Finished', jj, F[:jj].min())
                __makeVid(writer, plt, stage=3)
            else:
                plotter(recon[:n], r, e, f, jj)
                __makeVid(writer, plt, stage=2)
                print('%.2f, %.2f, % 5.3f' % (n / nAtoms,
                                              i / max_iter[0], E[jj - 1]))

    else:

        def do_plot(recon, r, e, f, n, i, jj, end=False):
            if end:
                print('Reconstruction Finished', jj, F[:jj].min())
                plt.show(block=True)
            else:
                print('%.2f, %.2f, % 5.3f' % (n / nAtoms,
                                              i / max_iter[0], E[jj - 1]))
                plt.draw()
                plt.pause(.01)
                plotter(recon[:n], r, e, f, jj)
                plt.show(block=False)

    return c, E, F, nAtoms, max_iter, do_plot


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

    ax[0, 2].plot(F[:jj])
    ax[0, 2].set_yscale('log')
    ax[0, 2].tick_params(axis='y', which='minor', bottom=True)
    ax[0, 2].set_title('Fidelity')
    ax[1, 2].plot(E[:jj])
    ax[1, 2].set_yscale('linear')
    ax[1, 2].set_ylim(E[jj - 1], 1.5 * E[jj // 4] - E[jj - 1] / 3)
    ax[1, 2].set_title('Energy')


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
#     ax[1, 1].imshow(minimum(0, (R - data).data), aspect='auto')
    ax[1, 1].set_title('Sinogram Error')

    ax[0, 2].plot(F[:jj])
    ax[0, 2].set_yscale('log')
    ax[0, 2].tick_params(axis='y', which='minor', bottom=True)
    ax[0, 2].set_title('Fidelity')
    ax[1, 2].plot(E[:jj])
    ax[1, 2].set_yscale('linear')
    ax[1, 2].set_ylim(E[jj - 1], 1.5 * E[jj // 4] - E[jj - 1] / 3)
    ax[1, 2].set_title('Energy')


def _plot3D(ax, recon, R, data, E, F, jj, nAtoms, thresh, angles):
    for a in ax.reshape(-1):
        a.clear()

    try:
        tmp = recon.asarray()
        m = _get3DvolPlot(ax[0, 0], tmp, angle=angles[0], thresh=thresh)
        m = _get3DvolPlot(ax[1, 0], tmp, angle=angles[1], m=m)
    except ValueError:
        pass
    ax[0, 0].set_title('Recon, view from orientation: ' + str(angles[0]))
    ax[1, 0].set_title('Recon, view from orientation: ' + str(angles[1]))

#     recon.plot(ax[0, 0], Slice=[slice(None), slice(
#         None), round(recon.shape[2] / 2)])
#     ax[0, 0].set_title('Recon, slice from top')
#     recon.plot(ax[1, 0], Slice=[slice(None), round(
#         recon.shape[1] / 2), slice(None)])
#     ax[1, 0].set_title('Recon, slice from front')

    Slice = int(R.shape[0] / 2)
    clim = [0, data.data[Slice].real.max()]
    data.plot(ax[0, 1], Slice=Slice, clim=clim)
    ax[0, 1].set_title('Middle Data Projection')
    R.plot(ax[1, 1], Slice=Slice, clim=clim)
    ax[1, 1].set_title('Middle Recon Projection')

    Slice = (jj // (3 * nAtoms)) % R.shape[0]
    clim = [0, data.data[Slice].real.max()]
    data.plot(ax[0, 2], Slice=Slice, clim=clim)
    ax[0, 2].set_title('Data Projection')
    R.plot(ax[1, 2], Slice=Slice, clim=clim)
    ax[1, 2].set_title('Recon Projection')

    ax[0, 3].plot(F[:jj])
    ax[0, 3].set_yscale('log')
    ax[0, 3].set_title('Fidelity')
    ax[1, 3].plot(E[:jj])
    ax[1, 3].set_yscale('linear')
    ax[1, 3].set_ylim(E[jj - 1], 1.5 * E[jj // 4] - E[jj - 1] / 3)
    ax[1, 3].set_title('Energy')


def _plot3DwithGT(ax, recon, gt, R, data, E, F, jj, nAtoms, thresh, angles):
    for a in ax.reshape(-1):
        a.clear()

    tmp = gt.asarray()
    m = _get3DvolPlot(ax[0, 0], tmp, angle=angles[0], thresh=thresh)
    ax[0, 0].set_title('GT, view from orientation: ' + str(angles[0]))
    _get3DvolPlot(ax[0, 1], tmp, angle=angles[1], m=m)
    ax[0, 1].set_title('GT, view from orientation: ' + str(angles[1]))

    try:
        tmp = recon.asarray()
        m = _get3DvolPlot(ax[1, 0], tmp, angle=angles[0], thresh=thresh)
        _get3DvolPlot(ax[1, 1], tmp, angle=angles[1], m=m)
    except Exception:
        pass
    ax[1, 0].set_title('Recon, view from orientation: ' + str(angles[0]))
    ax[1, 1].set_title('Recon, view from orientation: ' + str(angles[1]))

#     n = (round(gt.shape[2] / 2), round(gt.shape[1] / 2))
#     cax = (gt.data[:, :, n[0]].real.max(), gt.data[:, n[1]].real.max())
#     gt.plot(ax[0, 0], Slice=[slice(None), slice(
#         None), n[0]], vmin=0, vmax=cax[0])
#     ax[0, 0].set_title('GT, slice from top')
#     gt.plot(ax[0, 1], Slice=[slice(None), n[1],
#                              slice(None)], vmin=0, vmax=cax[1])
#     ax[0, 1].set_title('GT, slice from front')
#     recon.plot(ax[1, 0], Slice=[slice(None), slice(
#         None), n[0]], vmin=0, vmax=cax[0])
#     ax[1, 0].set_title('Recon, slice from top')
#     recon.plot(ax[1, 1], Slice=[slice(None), n[1],
#                                 slice(None)], vmin=0, vmax=cax[1])
#     ax[1, 1].set_title('Recon, slice from front')

    Slice = (jj // (3 * nAtoms)) % R.shape[0]
    clim = [0, data.data[Slice].real.max()]
    data.plot(ax[0, 2], Slice=Slice, clim=clim)
    ax[0, 2].set_title('Data Projection')
    R.plot(ax[1, 2], Slice=Slice, clim=clim)
    ax[1, 2].set_title('Recon Projection')

    ax[0, 3].plot(F[:jj])
    ax[0, 3].set_yscale('log')
    ax[0, 3].set_title('Fidelity')
    ax[1, 3].plot(E[:jj])
    ax[1, 3].set_yscale('linear')
    ax[1, 3].set_ylim(E[jj - 1], 1.5 * E[jj // 4] - E[jj - 1] / 3)
    ax[1, 3].set_title('Energy')


def _get3DvolPlot(ax, img, angle=(45, 0), thresh=None, m=None):
    if m is None:
        try:
            v, f, _, _ = measure.marching_cubes_lewiner(img, thresh, allow_degenerate=False)
            m = (v[:, 0], v[:, 1], f, v[:, 2])
        except Exception as e:
            return None

    if ax is None:
        from matplotlib import pyplot
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        ax = pyplot.gcf().add_subplot(111, projection='3d')

    tmp = ax.plot_trisurf(*m, alpha=1)
    tmp.set_edgecolor(tmp._facecolors)
    ax.set_xlim(0, img.shape[0])
    ax.set_ylim(0, img.shape[1])
    ax.set_zlim(0, img.shape[2])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(*angle)

    return m


def __makeVid(*args, stage=0, fps=5):
    '''
    stage 0: args=[], no return value
        sets graphics library
    stage 1: args=[figure, title], returns writer
        initiates file/writer
    stage 2: args=[writer, pyplot]
        records new frame
    stage 3: args=[writer, pyplot]
        saves video
    '''
    if stage == 0:
        import matplotlib
        matplotlib.use('Agg')
    elif stage == 1:
        from matplotlib import animation as mv
        writer = mv.writers['ffmpeg'](fps=5, metadata={'title': args[1]})
        writer.setup(args[0], args[1] + '.mp4', dpi=100)
        return writer
    elif stage == 2:
        args[1].pause(.1)
        args[0].grab_frame()
    elif stage == 3:
        args[0].finish()
        args[1].show(block=True)


def _step(recon, Radon, R, d, data, E, F, dim, iso, j, jj, n, fidelity, reg):
    c = context()

    R_old = Radon(recon[j])
    I = c.asarray(recon.I[j:j + 1]).copy()
    x = c.asarray(recon.x[j]).copy()
    r = c.asarray(recon.r[j]).copy()

#     c.set(recon.I[j:j + 1], I + d[0])
    c.set(recon.x[j], x + d[1:dim + 1])
#     c.set(recon.r[j], r + d[dim + 1:])

    c.set(recon.I[j:j + 1], max(0, I + d[0]))
#     c.set(recon.x[j], maximum(-.99, minimum(.99, x + d[1:1 + dim])))
    if iso:
        rr = abs(r + d[dim + 1])
    else:
        rr = r + d[dim + 1:]
        if dim == 2:
            if rr[0] < 0:
                rr[0], rr[2] = -rr[0], -rr[2]
            if rr[1] < 0:
                rr[1] = -rr[1]
        else:
            if rr[0] < 0:
                rr[0], rr[3], rr[5] = -rr[0], -rr[3], -rr[5]
            if rr[1] < 0:
                rr[1], rr[4] = -rr[1], -rr[4]
            if rr[2] < 0:
                rr[2] = -rr[2]
    c.set(recon.r[j], rr)
    
    R, R_old = R + Radon(recon[j]) - R_old, R
#     R = Radon(recon[:n])
    F[jj] = fidelity(R, data)
    E[jj] = F[jj] + reg(recon[:n])

    if E[jj] > E[jj - 1]:
        c.set(recon.I[j:j + 1], I)
        c.set(recon.x[j], x)
        c.set(recon.r[j], r)
        R = R_old
#         R = Radon(recon[:n])
    
    return R
