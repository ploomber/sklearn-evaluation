from copy import deepcopy
from collections.abc import Mapping

import nbformat


def _process_cell(cell):
    tags = cell.metadata.get('tags')
    return None if not tags else (tags[0], cell.outputs)


def _process_output(output):
    out_type = output['output_type']

    if out_type == 'display_data':
        data = output['data']
        # matplotlib charts sometimes produce execute_result and display_data
        # e.g. plt.plot([1, 2, 3], [1, 2, 3]) produces:
        # [<matplotlib.lines.Line2D at ...>]
        # and the plot. We ignore any execute_result outputs that only contain
        # plain text - we still have to find out if this is problematic for
        # other types of outputs
        return None if set(data) == {'text/plain'} else data
    elif out_type == 'stream':
        # so they all ahve the same format {mime_type: content, ...}
        return {'text/plain': output['text']}
    elif out_type == 'execute_result':
        return output['data']
    else:
        raise NotImplementedError('Processing output of type "{}" is not '
                                  'implemented'.format(out_type))


def _filter_and_process_outputs(outputs):
    allowed = {'stream', 'display_data', 'execute_result'}
    outputs = [
        _process_output(o) for o in outputs if o['output_type'] in allowed
    ]
    return [o for o in outputs if o is not None]


class NotebookIntrospector(Mapping):
    def __init__(self, path):
        self.nb = nbformat.read(path, nbformat.NO_CONVERT)
        self.tag2outputs = self._tag2outputs()

    def _tag2outputs(self):

        cells = [
            _process_cell(c) for c in self.nb.cells if c.cell_type == 'code'
        ]

        # ignore untagged cells
        cells = {c[0]: c[1] for c in cells if c is not None}

        # ignore some types of outputs on each cell
        cells = {
            k: _filter_and_process_outputs(outputs)
            for k, outputs in cells.items()
        }

        return cells

    def __getitem__(self, key):
        return self.tag2outputs[key]

    def __iter__(self):
        for k in self.tag2outputs:
            yield k

    def __len__(self):
        return len(self.tag2outputs)

    def eval(self, key):
        # TODO: only eval outputs of type "stream" - we have to keep a
        # secondary dictionary with output types
        outputs = self.tag2outputs.get(key)

        if not len(outputs):
            return None
        if len(outputs) == 1:
            return eval(outputs[0]['text/plain'])
        else:
            return outputs

    def display(self, key):
        """Display an output
        """
        pass

    def to_dict(self):
        out = deepcopy(self.tag2outputs)
        return {k: v[0]['text/plain'] for k, v in out.items()}
