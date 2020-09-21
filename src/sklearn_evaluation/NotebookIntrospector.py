import nbformat


def _process_cell(cell):
    tags = cell.metadata.get('tags')
    return None if not tags else (tags[0], cell.outputs)


class NotebookIntrospector:
    def __init__(self, path):
        self.nb = nbformat.read(path, nbformat.NO_CONVERT)
        self.tag2outputs = self._tag2outputs()

    def _tag2outputs(self):

        cells = [
            _process_cell(c) for c in self.nb.cells if c.cell_type == 'code'
        ]

        return {c[0]: c[1] for c in cells if c is not None}

    def __getitem__(self, key):
        return self.tag2outputs[key]

    def eval(self, key):
        outputs = self.tag2outputs.get(key)

        if not len(outputs):
            return None
        if len(outputs) == 1:
            return eval(outputs[0]['text'])
        else:
            return outputs

    def get_tags(self):
        return set(self.tag2outputs)
