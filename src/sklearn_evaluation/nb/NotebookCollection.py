from functools import partial
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
from jinja2 import Environment, PackageLoader

from .NotebookIntrospector import NotebookIntrospector

_env = Environment(loader=PackageLoader('sklearn_evaluation', 'assets/nb'))


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
    def __init__(self, paths, ids=None, scores=False):
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
        self._scores = scores

    def __getitem__(self, key):
        raw = [nb[key] for nb in self.nbs.values()]
        e, ids_out = add_compare_tab(raw, list(self.nbs.keys()), self._scores)
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


def add_compare_tab(elements, ids, scores_arg):
    out = copy.copy(elements)
    out_ids = copy.copy(ids)

    if isinstance(elements[0], (HTML, pd.DataFrame)):
        summary = make_df_summary(elements, ids, scores_arg)
    elif isinstance(elements[0], (Mapping, list)):
        summary = make_mapping_summary(elements)
    else:
        summary = None

    if summary is not None:
        out.append(summary)
        out_ids.append('Compare')

    return out, out_ids


def make_tabs(names, contents):
    # random prefix to prevent multiple tab outputs to clash with each other
    prefix = ''.join(random.choice(string.ascii_lowercase) for i in range(3))
    contents = [process_content(content) for content in contents]
    template = _env.get_template('template.html')

    html = template.render(names=names,
                           zip=zip,
                           contents=contents,
                           prefix=prefix)
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


def color_neg_green(s):
    return (s < 0).replace({True: 'color: green', False: 'color: red'})


def color_neg_red(s):
    return (s < 0).replace({True: 'color: red', False: 'color: green'})


def color(s, which, color):
    to_color = s == getattr(s[~s.isna()], which)()
    return [f'color: {color}' if v else '' for v in to_color]


_color_map = {
    'error': {
        'max': partial(color, which='max', color='red'),
        'min': partial(color, which='min', color='green'),
    },
    'score': {
        'max': partial(color, which='max', color='green'),
        'min': partial(color, which='min', color='red'),
    }
}


def is_score(scores_arg, index):
    if not scores_arg:
        return False
    elif scores_arg is True:
        return True
    else:
        return index in scores_arg


def split_errors_and_scores(axis, scores_arg, axis_second, transpose=False):
    scores = [i for i in axis if is_score(scores_arg, i)]
    errors = list(set(axis) - set(scores))

    errors_slice = pd.IndexSlice[errors, axis_second]
    scores_slice = pd.IndexSlice[scores, axis_second]

    if transpose:
        errors_slice = errors_slice[::-1]
        scores_slice = scores_slice[::-1]

    return errors_slice, scores_slice


_htmldiff = HtmlDiff()


def make_mapping_summary(mappings):
    if len(mappings) != 2:
        return None

    m1, m2 = mappings

    s1 = black.format_str(str(m1), mode=_fm).splitlines()
    s2 = black.format_str(str(m2), mode=_fm).splitlines()

    return _htmldiff.make_file(s1, s2)


def color_max(s):
    is_max = s == s[~s.isna()].max()
    return ['color: red' if v else '' for v in is_max]


def make_df_summary(tables, ids, scores_arg):
    dfs = [to_df(table) for table in tables]

    # Single-row data frames, each metric is a single number
    # TODO: check dims are consistent
    if len(dfs[0]) == 1:
        out = pd.concat(dfs)
        out.index = ids
        out = out.T
        errors, scores = split_errors_and_scores(out.index,
                                                 scores_arg,
                                                 axis_second=out.columns)

        if len(tables) == 2:
            c1, c2 = out.columns
            out['diff'] = out[c2] - out[c1]
            out['diff_relative'] = (out[c2] - out[c1]) / out[c2]
            out['ratio'] = out[c2] / out[c1]

        styled = out.style.apply(_color_map['error']['max'],
                                 subset=errors,
                                 axis='columns')
        styled = styled.apply(_color_map['error']['min'],
                              subset=errors,
                              axis='columns')
        styled = styled.apply(_color_map['score']['max'],
                              subset=scores,
                              axis='columns')
        styled = styled.apply(_color_map['score']['min'],
                              subset=scores,
                              axis='columns')

        styled = styled.format({'diff_relative': '{:.2%}'})

    # Multiple rows, each metric is a vector
    else:
        # we can only return a summary if dealing with two tables
        if len(tables) == 2:
            # TODO: generate "Compare diff", "Compare diff relative"
            # and "Compare ratio"
            out = dfs[1] - dfs[0]
            errors, scores = split_errors_and_scores(out.columns,
                                                     scores_arg,
                                                     axis_second=out.index,
                                                     transpose=True)

            styled = out.style.apply(color_neg_green,
                                     subset=errors,
                                     axis='rows')
            styled = styled.apply(color_neg_red, subset=scores, axis='rows')
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
