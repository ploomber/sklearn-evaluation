from string import Template
import plots
from cStringIO import StringIO
import base64
import os
from datetime import datetime
from utils import get_model_name
import matplotlib
import re

try:
    import mistune
except:
    raise Exception('You need to install mistune to use the report module')


def generate(evaluator, template, path=None, style='default'):
    # get list of tags from template
    tags = parse_tags(template)

    # get attributes from evaluator using the tags
    attrs = getattr_from_list(evaluator, tags)

    # convert axes objects to HTML with base64
    for k, v in attrs.items():
        if isinstance(v, matplotlib.axes.Axes):
            attrs[k] = figure2html(v.get_figure())

    # replace tags with values
    report = template.format(**attrs)

    # apply style
    report = apply_style(report, style)

    if path is not None:
        report_file = open(path, 'w')
        report_file.write(report)
        report_file.close()
    else:
        return report


def apply_style(template, style_name):
    # Read md template and compile to html
    pkg = os.path.dirname(os.path.abspath(__file__))
    markdown = mistune.Markdown()
    html = markdown(template)

    # Add css
    filepath = os.path.join(pkg, 'styles', '{}.css'.format(style_name))
    f = open(filepath, 'r')
    css = f.read()
    html = '<style>'+css+'</style>'+html
    return html


def figure2html(fig):
    return base64_2_html(figure2base64(fig))


def base64_2_html(img):
    return '<img src="data:image/png;base64,'+img+'"></img>'


def figure2base64(fig):
    io = StringIO()
    fig.savefig(io, format='png')
    fig_base64 = base64.encodestring(io.getvalue())
    return fig_base64


def prettify_list(l):
    l = [str(idx+1)+'. '+str(el) for idx, el in enumerate(l)]
    return reduce(lambda x, y: x+'<br>'+y, l)


def prettify_dict(d):
    return prettify_list([key+': '+str(d[key]) for key in d.keys()])


def parse_tags(s):
    """
        Return a list of tags (e.g. {tag_a}, {tag_b}) found in string s
    """
    return re.findall('{(\w+)\}*', s)


def getattr_from_list(obj, attr_names):
    return {attr: getattr(obj, attr) for attr in attr_names}
