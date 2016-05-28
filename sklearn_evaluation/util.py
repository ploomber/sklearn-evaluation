import re

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


def get_model_name(model):
    s = str(type(model))
    model_name = re.search(".*'(.+?)'.*", s).group(1).split(".")[-1]
    return model_name


def gen_ax():
    fig = Figure()
    FigureCanvas(fig)
    ax = fig.add_subplot(111)
    return ax
