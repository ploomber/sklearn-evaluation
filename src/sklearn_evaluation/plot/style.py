from contextlib import contextmanager
import matplotlib.pyplot as plt
from sklearn_evaluation import __version__
import matplotlib as mpl


@contextmanager
def tmp_theme():
    try:
        _set_default_plot_colors()
        yield
    finally:
        _set_default_ax_style()


def apply_theme(func):
    def wrapper_func(*args, **kwargs):
        with tmp_theme():
            func(*args, **kwargs)
    return wrapper_func


def _set_default_plot_colors():
    _material_ui_colors = ['#00B0FF', '#F50057', '#00C853',
                           '#D500F9', '#F44336', '#FF9100', '#00BFA5', '#651FFF']
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=_material_ui_colors)


def _set_default_ax_style():
    ax = plt.subplot()
    ax.set_xlim([0.0, 1.005])
    ax.set_ylim([0.0, 1.005])
    ax.legend(loc="best", fontsize='10')
    ax.spines[['right', 'top']].set_visible(False)
    ax.tick_params(right=False, top=False)
