import numpy as np
from matplotlib.tri import Triangulation

from sklearn_evaluation.util import default_heatmap


def add(first, second, ax, invert_axis=False, max_=None):
    # Adapted from: https://stackoverflow.com/a/63531813/709975

    # TODO: validate first and second have the same shape
    M, N = first.shape if not invert_axis else first.shape[::-1]
    x = np.arange(M + 1)
    y = np.arange(N + 1)

    xs, ys = np.meshgrid(x, y)

    zs = (xs * ys) % 10
    zs = zs[:-1, :-1].ravel()

    if max_ is None:
        max_ = np.max([first.max(), second.max()])

    triangles1 = [(i + j * (M + 1), i + 1 + j * (M + 1), i + (j + 1) * (M + 1))
                  for j in range(N) for i in range(M)]
    triangles2 = [(i + 1 + j * (M + 1), i + 1 + (j + 1) * (M + 1),
                   i + (j + 1) * (M + 1)) for j in range(N) for i in range(M)]
    triang1 = Triangulation(xs.ravel() - 0.5, ys.ravel() - 0.5, triangles1)
    triang2 = Triangulation(xs.ravel() - 0.5, ys.ravel() - 0.5, triangles2)

    cmap = default_heatmap()

    img1 = ax.tripcolor(triang1, first.ravel(), cmap=cmap, vmax=max_)
    _ = ax.tripcolor(triang2, second.ravel(), cmap=cmap, vmax=max_)
    # ax.figure.colorbar(img1)

    ax.set_xlim(x[0] - 0.5, x[-1] - 0.5)
    ax.set_ylim(y[-1] - 0.5, y[0] - 0.5)

    for pad, arr in ((-1 / 5, first), (1 / 5, second)):
        for (y, x), v in np.ndenumerate(arr):
            try:
                label = '{:.2}'.format(v)
            except Exception:
                label = v

            ax.text(x + pad,
                    y + pad,
                    label,
                    horizontalalignment='center',
                    verticalalignment='center')
