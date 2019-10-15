from tempfile import NamedTemporaryFile

import graphviz

from goldberg.core.causal_graph import CausalGraphTraverser, EventState


class CausalGraphViewer:
    def __init__(self, root):
        self.root = root
        self._file = NamedTemporaryFile()
        self.colors = {
            EventState.asleep: 'white',
            EventState.awake: 'yellow',
            EventState.success: 'green',
            EventState.failure: 'orange',
        }

    def render(self, filename=None, compact=True):
        filename = filename if filename else self._file.name
        g = graphviz.Digraph('G', filename=filename)
        g.attr('node', shape='circle')
        g.attr('node', fontname='Linux Biolinum O')
        to_process = {self.root}
        processed = set()
        if compact:
            self._labels = {}
            label = 'A'
            for event in CausalGraphTraverser(self.root).get_events():
                self._labels[event.name] = label
                label = increment_str(label)
        while to_process:
            event = to_process.pop()
            if event in processed:
                continue
            node_label = self._labels[event.name] if compact else event.name
            g.node(node_label, style='filled', color='black',
                   fillcolor=self.colors[event.state])
            if event.outcome:
                for trans in event.outcome.transitions:
                    edge_label = None if compact else str(int(trans.active))
                    dest_label = (self._labels[trans.dest.name] if compact
                                  else trans.dest.name)
                    g.edge(node_label, dest_label, label=edge_label)
                    to_process.add(trans.dest)
            processed.add(event)
        g.view()


def increment_char(c):
    """Increment an uppercase character, returning 'A' if 'Z' is given."""
    return chr(ord(c) + 1) if c != 'Z' else 'A'


def increment_str(s):
    lpart = s.rstrip('Z')
    num_replacements = len(s) - len(lpart)
    new_s = lpart[:-1] + increment_char(lpart[-1]) if lpart else 'A'
    new_s += 'A' * num_replacements
    return new_s
