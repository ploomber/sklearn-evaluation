from contextlib import contextmanager
import matplotlib.pyplot as plt
import matplotlib as mpl


@contextmanager
def tmp_theme(ax=None):
    """
    Adds a scheme coloring and styling to matplotlib plots.
    """
    try:
        _set_default_plot_colors()
        yield
    finally:
        _set_default_ax_style(ax)


def apply_theme(func):
    """
    Decorates `def plot(self, ax = None)` and applies a visual theme to matplotlib plots.
    """
    def wrapper_func(*args, **kwargs):
        # we assume the plotting function is ``def plot(self, ax = None)``
        ax = args[1] if len(args) > 1 else None

        with tmp_theme(ax):
            return func(*args, **kwargs)

    return wrapper_func


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
