from difflib import HtmlDiff
import random
import string
import base64
import copy
from pathlib import Path
from collections.abc import Mapping

import black
import pandas as pd
from IPython.display import HTML, Image
from jinja2 import Template

from .NotebookIntrospector import NotebookIntrospector


class NotebookCollection(Mapping):
    """Compare output from a collection of notebooks

    Parameters
    ----------
    paths : list
        Paths to notebooks to load

    ids : list or 'filenames', default=None
        List of ids (one per notebook), if None, paths are used as identifiers,
        if 'filenames', the file name is extracted from each path and used
        as identifier (ignores extension)

    """
    def __init__(self, paths, ids=None):
        if ids is None:
            ids = paths
        elif ids == 'filenames':
            ids = [_get_filename(path) for path in paths]

        self.nbs = {
            id_: NotebookIntrospector(path, to_df=False)
            for id_, path in zip(ids, paths)
        }

        nb = list(self.nbs.values())[0]

        self._keys = list(nb.tag2output.keys())
        self._raw = RawMapping(self)

    def __getitem__(self, key):
        raw = [nb[key] for nb in self.nbs.values()]
        e, ids_out = add_summary_tab(raw, list(self.nbs.keys()))
        m = {k: v for k, v in zip(ids_out, e)}
        html = make_tabs(ids_out, e)
        return HTMLOutput(m, html)

    # TODO: get rid of this
    @property
    def raw(self):
        return self._raw

    def __iter__(self):
        for k in self._keys:
            yield k

    def _ipython_key_completions_(self):
        return self._keys

    def __len__(self):
        return len(self._keys)


class HTMLOutput(Mapping):
    def __init__(self, mapping, html):
        self._mapping = mapping
        self._html = html

    def __getitem__(self, key):
        return self._mapping[key]

    def _ipython_key_completions_(self):
        return self._mapping.keys()

    def __iter__(self):
        for k in self._mapping:
            yield k

    def __len__(self):
        return len(self._mapping)

    def _repr_html_(self):
        return self._html


class RawMapping(Mapping):
    def __init__(self, collection):
        self.collection = collection

    def __getitem__(self, key):
        return {name: nb[key] for name, nb in self.collection.nbs.items()}

    def _ipython_key_completions_(self):
        return self.collection.keys()

    def __iter__(self):
        for k in self.collection:
            yield k

    def __len__(self):
        return len(self.collection)


def _get_filename(path):
    path = Path(path)
    return path.name.replace(path.suffix, '')


def add_summary_tab(elements, ids):
    out = copy.copy(elements)
    out_ids = copy.copy(ids)

    if isinstance(elements[0], (HTML, pd.DataFrame)):
        summary = make_df_summary(elements, ids)
    elif isinstance(elements[0], Mapping):
        summary = make_mapping_summary(elements)
    else:
        summary = None

    if summary is not None:
        out.append(summary)
        out_ids.append('Summary')

    return out, out_ids


def make_tabs(names, contents):
    # random prefix to prevent multiple tab outputs to clash with each other
    prefix = ''.join(random.choice(string.ascii_lowercase) for i in range(3))

    contents = [process_content(content) for content in contents]
    html = Template("""
<ul class="nav nav-tabs" id="myTab" role="tablist">
  {% for name in names %}
  <li class="nav-item" role="presentation">
    <a class="nav-link" id="{{prefix}}-{{name}}-tab" data-toggle="tab" href="#{{prefix}}-{{name}}" role="tab" aria-controls="{{prefix}}-{{name}}" aria-selected="true">{{name}}</a>
  </li>
  {% endfor %}
</ul>
<div class="tab-content" id="myTabContent">
  {% for name, content in zip(names, contents) %}
  <div class="tab-pane fade" id="{{prefix}}-{{name}}" role="tabpanel" aria-labelledby="{{prefix}}-{{name}}-tab">{{content}}</div>
  {% endfor %}
</div>
""").render(names=names, zip=zip, contents=contents, prefix=prefix)
    return html


def to_df(obj):
    if isinstance(obj, pd.DataFrame):
        return obj

    dfs = pd.read_html(obj.data)

    if len(dfs) > 1:
        raise ValueError('More than one table detected')

    df = dfs[0]
    df.columns = process_columns(df.columns)
    df = df.set_index(df.columns[0])
    return df


def process_columns(columns):
    if isinstance(columns, pd.MultiIndex):
        return [process_multi_index_col(name) for name in columns]
    else:
        return [None, *columns[1:]]


def process_multi_index_col(col):
    names = [name for name in col if 'Unnamed:' not in name]
    return names[0]


def color_negative_red(val):
    color = 'red' if val < 0 else 'black'
    return 'color: %s' % color


def color_neg_and_pos(val):
    color = 'green' if val < 0 else 'red'
    return 'color: %s' % color


def color_max(s):
    is_max = s == s[~s.isna()].max()
    return ['color: red' if v else '' for v in is_max]


def color_min(s):
    is_max = s == s[~s.isna()].min()
    return ['color: green' if v else '' for v in is_max]


_htmldiff = HtmlDiff()


def make_mapping_summary(mappings):
    if len(mappings) != 2:
        return None

    m1, m2 = mappings

    s1 = black.format_str(str(m1), mode=_fm).splitlines()
    s2 = black.format_str(str(m2), mode=_fm).splitlines()

    return _htmldiff.make_file(s1, s2)


def make_df_summary(tables, ids):
    dfs = [to_df(table) for table in tables]

    # Single-row data frames, each metric is a single number
    # TODO: check dims are consistent
    if len(dfs[0]) == 1:
        out = pd.concat(dfs)
        out.index = ids
        out = out.T

        if len(tables) == 2:
            c1, c2 = out.columns
            diff = out[c2] - out[c1]
            # TODO: add ratio and percentage
            out['diff'] = diff
            styled = out.style.applymap(color_neg_and_pos, subset=['diff'])
        else:
            styled = out.style.apply(color_max,
                                     axis='columns').apply(color_min,
                                                           axis='columns')
    # Multiple rows, each metric is a vector
    else:
        # we can only return a summary if dealing with two tables
        if len(tables) == 2:
            out = dfs[1] - dfs[0]
            styled = out.style.applymap(color_neg_and_pos)
        else:
            styled = None

    return styled


def data2html_img(data):
    img = base64.encodebytes(data).decode('utf-8')
    return '<img src="data:image/png;base64, {}"/>'.format(img)


_fm = black.FileMode(string_normalization=False, line_length=40)


def process_content(content):
    """Returns an HTML string representation of the content
    """
    if isinstance(content, Image):
        return data2html_img(content.data)
    elif isinstance(content, HTML):
        return content.data
    elif hasattr(content, '_repr_html_'):
        return content._repr_html_()
    elif isinstance(content, Mapping):
        c = black.format_str(str(content), mode=_fm)
        return f'<pre>{c}</pre>'
    else:
        return str(content)
