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
from .sets import differences
from ..table import Table
from ..telemetry import SKLearnEvaluationLogger

_env = Environment(loader=PackageLoader("sklearn_evaluation", "assets/nb"))
_fm = black.FileMode(string_normalization=False, line_length=40)
_htmldiff = HtmlDiff()


class NotebookCollection(Mapping):
    """Compare output from a collection of notebooks

    To access output, notebooks must tag the cells (one tag per cell). For
    instructions on tagging cells, `see this <https://papermill.readthedocs.io/en/latest/usage-parameterize.html>`_.

    :doc:`Click here <../user_guide/NotebookCollection>` to see the user guide.

    Parameters
    ----------
    paths : list
        Paths to notebooks to load

    ids : list or 'filenames', default=None
        List of ids (one per notebook), if None, paths are used as identifiers,
        if 'filenames', the file name is extracted from each path and used
        as identifier (ignores extension)
    """

    @SKLearnEvaluationLogger.log(
        feature="NotebookCollection", action="init-NotebookCollection"
    )
    def __init__(self, paths, ids=None, scores=False):
        if ids is None:
            ids = paths
        elif ids == "filenames":
            ids = [_get_filename(path) for path in paths]

        self.nbs = {
            id_: NotebookIntrospector(path, to_df=False)
            for id_, path in zip(ids, paths)
        }

        nb = list(self.nbs.values())[0]

        self._keys = list(nb.tag2output.keys())
        self._scores = scores

    def __getitem__(self, key):
        raw = [nb[key] for nb in self.nbs.values()]
        e, ids_out = add_compare_tab(raw, list(self.nbs.keys()), self._scores)
        mapping = {k: v for k, v in zip(ids_out, e)}
        html = tabs_html_from_content(ids_out, e)
        return HTMLMapping(mapping, html)

    def __iter__(self):
        for k in self._keys:
            yield k

    def _ipython_key_completions_(self):
        return self._keys

    def __len__(self):
        return len(self._keys)


class HTMLMapping(Mapping):
    """A mapping that has an HTML representation

    Parameters
    ----------
    mapping : dict
        The mapping with the data

    HTML : str
        The HTML representation of the mapping
    """

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


def _get_filename(path):
    path = Path(path)
    return path.name.replace(path.suffix, "")


def add_compare_tab(elements, ids, scores_arg):
    """
    Processes tab contents and ids, adding a "Compare" tab if possible

    Parameters
    ----------
    elements
        The elements to compare

    ids : list
        The IDs for each element

    Returns
    -------
    out
        A new set of element to display. It might contain one new element in
        index 0 if we can provide a comparison view

    out_ids
        A new set of IDs. It might contain a new ID in index 0 if we can
        provide a comparison view
    """
    out = copy.copy(elements)
    out_ids = copy.copy(ids)

    if isinstance(elements[0], (HTML, pd.DataFrame)):
        summary = compare_df(elements, ids, scores_arg)
    elif isinstance(elements[0], Mapping):
        summary = compare_diff(elements)
    # lists with dicts fail because they are not hashable
    elif isinstance(elements[0], (list, set)):
        summary = compare_sets(elements, ids=ids)
    else:
        summary = None

    if summary is not None:
        out.insert(0, summary)
        out_ids.insert(0, "Compare")

    return out, out_ids


def tabs_html_from_content(names, contents):
    """
    Generate the tabs and content to display as an HTML string
    """
    # random prefix to prevent multiple tab outputs to clash with each other
    prefix = "".join(random.choice(string.ascii_lowercase) for i in range(3))
    contents_html = [to_html_str(content) for content in contents]
    template = _env.get_template("template.html")
    html = template.render(names=names, zip=zip, contents=contents_html, prefix=prefix)
    return html


def to_df(obj):
    """
    Converts pandas.DataFrame, if the object is already one, returns it.
    Otherwise it tried to convert it from a HTML table. Raises an error
    if more than one table is detected
    """
    if isinstance(obj, pd.DataFrame):
        return obj

    dfs = pd.read_html(obj.data)

    if len(dfs) > 1:
        raise NotImplementedError(
            "More than one table detected, only outputs"
            " with a single table are supported"
        )

    df = dfs[0]
    df.columns = process_columns(df.columns)
    df = df.set_index(df.columns[0])
    return df


def process_columns(columns):
    """
    Helper function to parse column names from pandas.DataFrame objects
    parsed from HTML tables
    """
    if isinstance(columns, pd.MultiIndex):
        return [process_multi_index_col(name) for name in columns]
    else:
        return [None, *columns[1:]]


def process_multi_index_col(col):
    """
    Helper function to parse column names from pandas.DataFrame objects
    with  multi indexes parsed from HTML tables
    """
    names = [name for name in col if "Unnamed:" not in name]
    return names[0]


def color_neg_green(s):
    return (s < 0).replace({True: "color: green", False: "color: red"})


def color_neg_red(s):
    return (s < 0).replace({True: "color: red", False: "color: green"})


def color(s, which, color):
    """
    pandas.DataFrame function to add color to cell's text
    """
    to_color = s == getattr(s[~s.isna()], which)()
    return [f"color: {color}" if v else "" for v in to_color]


_color_map = {
    "error": {
        "max": partial(color, which="max", color="red"),
        "min": partial(color, which="min", color="green"),
    },
    "score": {
        "max": partial(color, which="max", color="green"),
        "min": partial(color, which="min", color="red"),
    },
}


def is_in(elements, value):
    """
    Determines if a value is in a list. It also handles degenerate cases
    when instead of a list, elements is True, False  or None
    """
    if not elements:
        return False
    elif elements is True:
        return True
    else:
        return value in elements


def split_errors_and_scores(axis, scores_arg, axis_second, transpose=False):
    """
    Determines which metrics are scores and which ones are metrics based on
    the "scores_arg". Returns a pd.IndexSlice object that can be used in
    pandas.DataFrame styling functions
    """
    scores = [i for i in axis if is_in(scores_arg, i)]
    errors = list(set(axis) - set(scores))

    errors_slice = pd.IndexSlice[errors, axis_second]
    scores_slice = pd.IndexSlice[scores, axis_second]

    if transpose:
        errors_slice = errors_slice[::-1]
        scores_slice = scores_slice[::-1]

    return errors_slice, scores_slice


def compare_diff(mappings):
    """
    Generates an HTML object with a diff view of two mappings
    """
    if len(mappings) != 2:
        return None

    m1, m2 = mappings

    s1 = black.format_str(str(m1), mode=_fm).splitlines()
    s2 = black.format_str(str(m2), mode=_fm).splitlines()

    return HTML(_htmldiff.make_file(s1, s2))


def compare_sets(sets, ids):
    """
    Generates a Table object with three columns comparing two sets: 1) elements
    in both sets, 2) elements in the first set and 3) elements in the second
    set. Raises an error if sets does not have two elements
    """
    if len(sets) != 2:
        return None

    header = ["Both"] + [f"Only in {id_}" for id_ in ids]

    return Table.from_columns(content=differences(*sets), header=header)


def compare_df(tables, ids, scores_arg):
    """
    Generates a comparison from a list of tables. Taables can be either a
    pandas.DataFrame or a str with an HTML table. The output depends
    on the number of tables and rows. Returns a pandas.DataFrame with style
    added (colors)
    """
    dfs = [to_df(table) for table in tables]

    # Single-row data frames, each metric is a single number
    # TODO: check dims are consistent
    if len(dfs[0]) == 1:
        out = pd.concat(dfs)
        out.index = ids
        out = out.T
        errors, scores = split_errors_and_scores(
            out.index, scores_arg, axis_second=out.columns
        )

        if len(tables) == 2:
            c1, c2 = out.columns
            out["diff"] = out[c2] - out[c1]
            out["diff_relative"] = (out[c2] - out[c1]) / out[c2]
            out["ratio"] = out[c2] / out[c1]

        styled = out.style.apply(
            _color_map["error"]["max"], subset=errors, axis="columns"
        )
        styled = styled.apply(_color_map["error"]["min"], subset=errors, axis="columns")
        styled = styled.apply(_color_map["score"]["max"], subset=scores, axis="columns")
        styled = styled.apply(_color_map["score"]["min"], subset=scores, axis="columns")

        styled = styled.format({"diff_relative": "{:.2%}"})

    # Multiple rows, each metric is a vector
    else:
        # we can only return a summary if dealing with two tables
        if len(tables) == 2:
            # TODO: generate "Compare diff", "Compare diff relative"
            # and "Compare ratio"
            out = dfs[1] - dfs[0]
            errors, scores = split_errors_and_scores(
                out.columns, scores_arg, axis_second=out.index, transpose=True
            )

            styled = out.style.apply(color_neg_green, subset=errors, axis="rows")
            styled = styled.apply(color_neg_red, subset=scores, axis="rows")
        else:
            styled = None

    return styled


def data2html_img(data):
    """Converts a png image (bytes) to HTML str with the image in base64"""
    img = base64.encodebytes(data).decode("utf-8")
    return '<img src="data:image/png;base64, {}"/>'.format(img)


def to_html_str(content):
    """Returns an HTML string representation of the content"""
    if isinstance(content, Image):
        return data2html_img(content.data)
    elif isinstance(content, HTML):
        return content.data
    elif hasattr(content, "_repr_html_"):
        return content._repr_html_()
    elif isinstance(content, Mapping):
        c = black.format_str(str(content), mode=_fm)
        # add <pre></pre> to keep whitespace
        return f"<pre>{c}</pre>"
    else:
        return str(content)
