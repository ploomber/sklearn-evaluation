import base64
from copy import deepcopy
from collections.abc import Mapping
from functools import partial
import ast

import nbformat
from IPython.display import Image, HTML


def _safe_eval(source, none_if_error=False):
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
        # return dict so they all ahve the same format
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


def _select_mime_type(output):
    if 'image/png' in output:
        return Image(data=base64.b64decode(output['image/png']))
    elif 'text/html' in output:
        return HTML(output['text/html'])
    elif 'text/plain' in output:
        return output['text/plain']


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
    def __init__(self, path):
        self.nb = nbformat.read(path, nbformat.NO_CONVERT)
        self.tag2output = self._tag2output()

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
        return _select_mime_type(self.tag2output[key])

    def __iter__(self):
        for k in self.tag2output:
            yield k

    def __len__(self):
        return len(self.tag2output)

    def eval(self, key):
        # TODO: only eval outputs of type "stream" - we have to keep a
        # secondary dictionary with output types
        outputs = self.tag2output.get(key)

        if not len(outputs):
            return None
        if len(outputs) == 1:
            return ast.literal_eval(outputs['text/plain'])
        else:
            return outputs

    def display(self, key):
        """Display an output
        """
        pass

    def to_dict(self, eval_=True, skip_eval_fail=True):
        # TODO: show warning if failing to eval at least one key
        fn = partial(_safe_eval,
                     none_if_error=skip_eval_fail) if eval_ else _do_nothing
        out = deepcopy(self.tag2output)
        d = {k: fn(v['text/plain']) for k, v in out.items()}
        return {k: v for k, v in d.items() if v is not None}

    def __repr__(self):
        return '{} with {}'.format(type(self).__name__, set(self.tag2output))

    def _ipython_key_completions_(self):
        return self.tag2output.keys()
