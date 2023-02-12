from contextlib import contextmanager
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from sklearn_evaluation.util import truncate_colormap
import inspect


@contextmanager
def tmp_theme(ax=None, coloring=True, ax_style=True):
    """
    Adds a scheme coloring and styling to matplotlib plots.
    """
    try:
        theme = dict()

        if coloring:
            _set_default_plot_colors()
            cmap = get_and_register_cmap()
            theme['cmap'] = cmap
        yield theme

    finally:
        if ax_style:
            _set_default_ax_style(ax)


def apply_theme(coloring=True, ax_style=True):
    """
    Decorates `def plot(self, ax = None, **theme)` and applies a visual theme to matplotlib plots.

    Parameters
    ----------
    coloring bool, default True
        Apply color scheme

    ax_style bool, default True
        Apply ax style
    """
    def decorator(func):
        def wrapper_func(*args):
            ax = args[1] if len(args) > 1 else None

            with tmp_theme(ax, coloring, ax_style) as theme:
                if 'theme' in inspect.signature(func).parameters.keys():
                    return func(*args, **theme)
                else:
                    return func(*args)

        return wrapper_func
    return decorator


def _set_default_plot_colors():
    _material_ui_colors = [
        "#00B0FF",
        "#F50057",
        "#00C853",
        "#D500F9",
        "#F44336",
        "#FF9100",
        "#00BFA5",
        "#651FFF",
    ]
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=_material_ui_colors)


def _set_default_ax_style(ax=None):
    if ax is None:
        ax = plt.subplot()
    ax.set_xlim([0.0, 1.005])
    ax.set_ylim([0.0, 1.005])
    ax.legend(loc="best", fontsize="10")
    ax.spines[["right", "top"]].set_visible(False)
    ax.tick_params(right=False, top=False)


def get_and_register_cmap():
    _material_ui_colors = [
        "#00B0FF",
        "#f8fdff",
    ]
    cmap = LinearSegmentedColormap.from_list(
        'material_cmap', _material_ui_colors)
    cmap = truncate_colormap(cmap, 0, 1)

    if cmap.name not in plt.colormaps():
        mpl.colormaps.register(cmap)
        plt.rcParams['image.cmap'] = cmap.name
    return cmap
