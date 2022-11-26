from functools import reduce
import base64
from six import BytesIO
import matplotlib


class EvaluatorHTMLSerializer:
    """
    Wraps ClassifierEvaluator so attributes and methods return an HTML
    serializable version of them
    """

    def __init__(self, evaluator):
        self.evaluator = evaluator

    def __getattr__(self, key):
        attr = getattr(self.evaluator, key)

        if callable(attr):
            return HTMLSerializableCallable(attr)
        else:
            return attr


class HTMLSerializableCallable:
    """Wraps a method so that the results is serialized after it is run
    """

    def __init__(self, attr):
        self.attr = attr

    def __call__(self, *args, **kwargs):
        obj = self.attr(*args, **kwargs)

        if isinstance(obj, matplotlib.axes.Axes):
            return figure2html(obj.get_figure())
        elif isinstance(obj, matplotlib.figure.Figure):
            return figure2html(obj)
        elif hasattr(obj, 'to_html'):
            return obj.to_html()
        else:
            raise TypeError('Unsupported type {}'.format(type(obj)))


def figure2html(fig):
    return base64_2_html(figure2base64(fig))


def base64_2_html(img):
    try:
        html = '<img src="data:image/png;base64,' + img + '"></img>'  # py2
    except Exception:
        img = img.decode("utf-8")
        html = '<img src="data:image/png;base64,' + img + '"></img>'  # py3
    return html


def figure2base64(fig):
    io = BytesIO()
    fig.savefig(io, format='png')
    try:
        fig_base64 = base64.encodebytes(io.getvalue())  # py3
    except Exception:
        fig_base64 = base64.encodestring(io.getvalue())  # py2
    return fig_base64


def prettify_list(elements):
    elements = [
        str(idx + 1) + '. ' + str(el) for idx, el in enumerate(elements)
    ]
    return reduce(lambda x, y: x + '<br>' + y, elements)


def prettify_dict(d):
    return prettify_list([key + ': ' + str(d[key]) for key in d.keys()])


def try_figure2html(obj):
    if hasattr(obj, 'savefig'):
        return figure2html(obj)
    else:
        return obj


def try_serialize_figures(d):
    return {key: try_figure2html(value) for key, value in d.items()}
