from contextlib import contextmanager
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from sklearn_evaluation.util import truncate_colormap


@contextmanager
def tmp_theme(ax_style, color_style):
    """
    Adds a scheme coloring and styling to matplotlib plots.

    Parameters
    ----------
    ax_style : str, default 'no_frame'
        Define which ax style to apply.

        Availble styles:
        'no_frame' no top and right boundaries

        'frame' ax with border

    color_style : str, default 'monochromatic'
        Define which coloring style to apply.

        Availble styles:

        'monochromatic' a palette in which a single color tint is used

        'gradient' a palette of two colors gradually shift from one to another
    """
    try:
        _set_default_plot_colors(color_style)
        _set_default_rc_params(ax_style)

        yield

    finally:
        pass


def _set_default_rc_params(ax_style):
    """
    Set default rcParams
    """
    # https://matplotlib.org/stable/tutorials/introductory/customizing.html#the-default-matplotlibrc-file
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['legend.loc'] = 'best'

    if ax_style == "no_frame":
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['ytick.right'] = False
        plt.rcParams['xtick.top'] = False


def get_color_palette(n_colors=None):
    """
    Returns default color palette
    """
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

    if n_colors is None:
        n_colors = len(_material_ui_colors)

    return _material_ui_colors[:n_colors]


def apply_theme(ax_style="no_frame", color_style="monochromatic"):
    """
    Decorates plotting function and applies a visual theme to matplotlib plots.

    Parameters
    ----------
    ax_style : str, default 'no_frame'
        Define which ax style to apply.

        Availble styles:
        'no_frame' wihout top and right boundaries

        'frame' ax with border

    color_style : str, default 'monochromatic'
        Define which coloring style to apply.

        Availble styles:

        'monochromatic' a palette in which a single color tint is used

        'gradient' a palette of two colors gradually shift from one to another
    """
    def decorator(func):
        def wrapper_func(*args, **kwargs):
            with tmp_theme(ax_style, color_style):
                return func(*args, **kwargs)

        return wrapper_func
    return decorator


def _set_default_plot_colors(color_style):
    material_ui_colors = get_color_palette()
    plt.rcParams["axes.prop_cycle"] = mpl.cycler(color=material_ui_colors)

    if color_style == "gradient":
        gradient_cmap()
    else:
        default_cmap()


def default_cmap():
    """
    Returns palette in which a single color tint is used.
    """
    _material_ui_colors = [
        "#c5edff",
        "#00B0FF",
    ]

    cmap = LinearSegmentedColormap.from_list(
        'material_cmap', _material_ui_colors)
    cmap = truncate_colormap(cmap, 0, 1)

    if cmap.name not in plt.colormaps():
        mpl.colormaps.register(cmap)

    plt.rcParams['image.cmap'] = cmap.name
    return cmap


def gradient_cmap():
    """
    Returns a palette of two colors gradually shift from one to another
    """
    _material_ui_colors = ["#00B0FF", "#ffa7c6"]
    cmap = LinearSegmentedColormap.from_list(
        'material_ui_bar', _material_ui_colors)

    if cmap.name not in plt.colormaps():
        mpl.colormaps.register(cmap)

    plt.rcParams['image.cmap'] = cmap.name

    return cmap


def default_heatmap():
    return truncate_colormap(plt.cm.OrRd, 0.1, 0.7)
