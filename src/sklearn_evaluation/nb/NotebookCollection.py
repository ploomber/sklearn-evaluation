import random
import string
import base64
import copy
from pathlib import Path
from collections.abc import Mapping

import pandas as pd
from IPython.display import HTML, Image
from jinja2 import Template

from .NotebookIntrospector import NotebookIntrospector


class NotebookCollection(Mapping):
    def __init__(self, paths, to_df=False, keys=None):
        if keys is None:
            keys = paths
        elif keys == 'filenames':
            keys = [_get_filename(path) for path in paths]

        self.nbs = {
            key: NotebookIntrospector(path, to_df=to_df)
            for key, path in zip(keys, paths)
        }

        # TODO: remove this
        self.to_df = to_df

        nb = list(self.nbs.values())[0]

        self._keys = list(nb.tag2output.keys())
        self._raw = RawOutput(self, to_df)

    def __getitem__(self, key):
        raw = [nb[key] for nb in self.nbs.values()]
        e, ids_out = add_summary_tab(raw, list(self.nbs.keys()))
        return make_tabs(ids_out, e)

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


class AbstractOutput(Mapping):
    def __init__(self, collection):
        self.collection = collection

    def _ipython_key_completions_(self):
        return self.collection.keys()

    def __iter__(self):
        for k in self.collection:
            yield k

    def __len__(self):
        return len(self.collection)


class RawOutput(AbstractOutput):
    def __init__(self, collection, to_df):
        super().__init__(collection)
        self.to_df = to_df

    def __getitem__(self, key):
        item = [nb[key] for nb in self.collection.nbs.values()]
        return item if not self.to_df else _to_df(item,
                                                  index=self.collection.keys())


def _get_filename(path):
    path = Path(path)
    return path.name.replace(path.suffix, '')


def _to_df(values, index):
    df = pd.DataFrame(values)
    df.index = index
    return df


def add_summary_tab(elements, ids):
    out = copy.copy(elements)
    out_ids = copy.copy(ids)

    if isinstance(elements[0], (HTML, pd.DataFrame)):
        summary = HTML(metrics_df(elements, ids)._repr_html_())
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
    return HTML(html)


def to_df(obj):
    if isinstance(obj, pd.DataFrame):
        return obj

    dfs = pd.read_html(obj.data)

    if len(dfs) > 1:
        raise ValueError('More than one table detected')

    df = dfs[0]

    if df.columns[0] == 'Unnamed: 0':
        df.columns = [None, *df.columns[1:]]

    df = df.set_index(df.columns[0])
    return df


def color_negative_red(val):
    color = 'red' if val < 0 else 'black'
    return 'color: %s' % color


def color_neg_and_pos(val):
    color = 'green' if val < 0 else 'red'
    return 'color: %s' % color


def color_max(s):
    is_max = s == s.max()
    return ['color: red' if v else '' for v in is_max]


def color_min(s):
    is_max = s == s.min()
    return ['color: green' if v else '' for v in is_max]


def metrics_df(tables, ids):
    out = pd.concat([to_df(table) for table in tables])
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

    return styled


def data2html_img(data):
    img = base64.encodebytes(data).decode('utf-8')
    return '<img src="data:image/png;base64, {}"/>'.format(img)


def process_content(content):
    """Returns an HTML string representation of the content
    """
    if isinstance(content, Image):
        return data2html_img(content.data)
    elif isinstance(content, HTML):
        return content.data
    elif isinstance(content, pd.DataFrame):
        return content._repr_html_()
    else:
        t = type(content)
        raise NotImplementedError(
            f'Do not know how to handle object of type {t!r}')
