import base64
from collections.abc import Mapping
import ast

import parso
import pandas as pd
import nbformat
from IPython.display import Image, HTML


def _safe_literal_eval(source, to_df=False, none_if_error=False):
    """
    Evaluates a literal, if the code cannot be parsed, it returns the original
    source as a string unless non_if_error is True, in such case it returns
    None
    """
    try:
        result = ast.literal_eval(source)

        if isinstance(result, Mapping) and to_df:
            result = pd.DataFrame(result, index=[0])

        return result

    except (SyntaxError, ValueError):
        return None if none_if_error else source.strip()


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


def _allowed_output(output):
    allowed = {'stream', 'display_data', 'execute_result'}

    if output['output_type'] in allowed:
        return output.get('name') != 'stderr'
    else:
        return False


def _filter_and_process_outputs(outputs):
    outputs = [o for o in outputs if _allowed_output(o)]
    # just return the latest output if any
    return None if not outputs else _process_output(outputs[-1])


def _parse_output(output, literal_eval, to_df, text_only):
    if not text_only and 'image/png' in output:
        return Image(data=base64.b64decode(output['image/png']))
    elif not text_only and 'text/html' in output:
        return HTML(output['text/html'])
    elif 'text/plain' in output:
        out = output['text/plain']
        return out if not literal_eval else _safe_literal_eval(out,
                                                               to_df=to_df)


def find_cell_with_tag(cells, tag):
    for cell in cells:
        if ('metadata' in cell and 'tags' in cell['metadata']
                and tag in cell['metadata']['tags']):
            return cell


def parse_injected_parameters_cell(cells):
    # this is a very simple implementation, for a more robust solution
    # re-implement with ast or parso
    cell = find_cell_with_tag(cells, tag='injected-parameters')

    if not cell:
        return dict()

    children = parso.parse(cell['source']).children

    statements = [
        _process_stmt(c) for c in children
        if c.type in {'simple_stmt', 'expr_stmt'}
    ]

    return {
        stmt.children[0].value:
        ast.literal_eval(stmt.children[2].get_code().strip())
        for stmt in statements if stmt is not None
    }


def _process_stmt(stmt):
    if stmt.type == 'expr_stmt':
        return stmt
    else:
        for c in stmt.children:
            if c.type == 'expr_stmt':
                return c


class NotebookIntrospector(Mapping):
    """Retrieve output from a notebook file with tagged cells.

    For instructions on tagging cells,
    `see this <https://papermill.readthedocs.io/en/latest/usage-parameterize.html>`_.

    Notes
    -----
    Ignores untagged cells, if a cell has more than one tag, it uses the first
    one as identifier. If a cell has more than one output, it uses the last
    one and discards the rest.
    """

    # TODO: how to handle a print(var) case? it removes the '' and causes a
    # NameError in eval
    def __init__(self, path, literal_eval=True, to_df=False):
        self.nb = nbformat.read(path, nbformat.NO_CONVERT)
        self.tag2output_raw = self._tag2output()
        self.literal_eval = literal_eval
        self.tag2output = {
            k: _parse_output(v,
                             literal_eval=literal_eval,
                             to_df=to_df,
                             text_only=False)
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

    def to_json_serializable(self):
        return {
            k: _parse_output(v,
                             literal_eval=self.literal_eval,
                             to_df=False,
                             text_only=True)
            for k, v in self.tag2output_raw.items()
        }

    def get_injected_parameters(self):
        return parse_injected_parameters_cell(self.nb.cells)
