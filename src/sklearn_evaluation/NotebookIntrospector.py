from pathlib import Path
import base64
from collections.abc import Mapping
import ast

import nbformat
from IPython.display import Image, HTML


def _safe_literal_eval(source, none_if_error=False):
    try:
        return ast.literal_eval(source)
    except SyntaxError:
        return None if none_if_error else source


def _do_nothing(source):
    return source


def _process_cell(cell):
    tags = cell.metadata.get('tags')
    # TODO: show warning if more than one tag or tagged cell with no output
    return None if (not tags or not cell.outputs) else (tags[0], cell.outputs)


def _process_output(output):
    out_type = output['output_type']

    if out_type == 'stream':
        # this happens when doing
        # a = [1, 2, 3]
        # print(a)
        # return dict so they all have the same format
        # {mime_type: content, ...}
        return {'text/plain': output['text']}
    elif out_type in {'execute_result', 'display_data'}:
        # this happens when doing:
        # a = [1, 2, 3]
        # a
        return output['data']
    else:
        raise NotImplementedError('Processing output of type "{}" is not '
                                  'implemented'.format(out_type))


def _filter_and_process_outputs(outputs):
    allowed = {'stream', 'display_data', 'execute_result'}
    outputs = [o for o in outputs if o['output_type'] in allowed]
    # just return the latest output if any
    return None if not outputs else _process_output(outputs[-1])


def _parse_output(output, literal_eval):
    if 'image/png' in output:
        return Image(data=base64.b64decode(output['image/png']))
    elif 'text/html' in output:
        return HTML(output['text/html'])
    elif 'text/plain' in output:
        out = output['text/plain']
        return out if not literal_eval else _safe_literal_eval(out)


class NotebookIntrospector(Mapping):
    """

    Notes
    -----
    Ignores untagged cells, if a cell has more than one tag, it uses the first
    one as identifier. If a cell has more than one output, it uses the last
    one and discards the rest.
    """

    # TODO: how to handle a print(var) case? it removes the '' and causes a
    # NameError in eval
    def __init__(self, path, literal_eval=True):
        self.nb = nbformat.read(path, nbformat.NO_CONVERT)
        self.tag2output_raw = self._tag2output()
        self.tag2output = {
            k: _parse_output(v, literal_eval=literal_eval)
            for k, v in self.tag2output_raw.items()
        }

    def _tag2output(self):
        # TODO: raise warning if notebook does not have any tagged cells

        cells = [
            _process_cell(c) for c in self.nb.cells if c.cell_type == 'code'
        ]

        # ignore untagged cells and cells with no output
        cells = {c[0]: c[1] for c in cells if c is not None}

        # ignore some types of outputs on each cell (i.e. errors) and get
        # the last valid one
        cells = {
            k: _filter_and_process_outputs(outputs)
            for k, outputs in cells.items()
        }

        # TODO: remove entries with None values?

        return cells

    def __getitem__(self, key):
        return self.tag2output[key]

    def __iter__(self):
        for k in self.tag2output:
            yield k

    def __len__(self):
        return len(self.tag2output)

    def __repr__(self):
        return '{} with {}'.format(type(self).__name__, set(self.tag2output))

    def _ipython_key_completions_(self):
        return self.tag2output.keys()


class NotebookCollection(Mapping):
    def __init__(self, paths, to_df=False, keys=None):
        if keys is None:
            keys = paths
        elif keys == 'filenames':
            keys = [_get_filename(path) for path in paths]

        self.nbs = {
            key: NotebookIntrospector(path)
            for key, path in zip(keys, paths)
        }
        self.to_df = to_df

    def __getitem__(self, key):
        item = [nb[key] for nb in self.nbs.values()]
        return item if not self.to_df else _to_df(item, index=self.nbs.keys())

    def __iter__(self):
        for k in self.nbs.keys():
            yield k

    def __len__(self):
        return len(self.nbs)


def _get_filename(path):
    path = Path(path)
    return path.name.replace(path.suffix, '')


def _to_df(values, index):
    import pandas as pd
    df = pd.DataFrame(values)
    df.index = index
    return df
