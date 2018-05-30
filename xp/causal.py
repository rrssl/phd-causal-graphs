from enum import Enum
from tempfile import NamedTemporaryFile

from graphviz import Digraph

import xp.config as cfg


class EventState(Enum):
    asleep = 1
    awake = 2
    success = 3
    failure = 4


class Event:
    def __init__(self, name, condition, precondition, outcome, verbose=False):
        self.name = name
        self.condition = condition
        self.precondition = precondition
        self.outcome = outcome
        self.verbose = verbose
        self.state = EventState.asleep
        self.wake_time = 0

    def __repr__(self):
        return self.name

    def reset(self):
        self.state = EventState.asleep
        self.wake_time = 0

    def update(self, time):
        if self.state is EventState.asleep:
            if self.precondition is None or self.precondition():
                self.state = EventState.awake
                self.wake_time = time
                if self.verbose:
                    print("{} is waiting to happen.".format(self))
        if self.state is EventState.awake:
            if self.condition():
                self.state = EventState.success
                if self.verbose:
                    print("{} has happened.".format(self))
                if self.outcome:
                    self.outcome(self.condition)
            elif time - self.wake_time > cfg.MAX_WAIT_TIME:
                self.state = EventState.failure
                if self.verbose:
                    print("{} has not happened.".format(self))


class Transition:
    def __init__(self, source=None, dest=None):
        self.source = source
        self.dest = dest
        self.active = False

    def __repr__(self):
        r = "transition from {} to {}".format(self.source, self.dest)
        return r


class AnyBefore:
    """Precondition"""
    def __init__(self, transitions=None):
        self.transitions = transitions if transitions is not None else []

    def __call__(self):
        return any(trans.active for trans in self.transitions)


class AllBefore:
    """Precondition"""
    def __init__(self, transitions=None):
        self.transitions = transitions if transitions is not None else []

    def __call__(self):
        return all(trans.active for trans in self.transitions)


class AllAfter:
    """Outcome"""
    def __init__(self, transitions=None, verbose=False):
        self.transitions = transitions if transitions is not None else []
        self.verbose = verbose

    def __call__(self, condition):
        for trans in self.transitions:
            trans.active = True
            if self.verbose:
                print("Activating {}".format(trans))


class CategoricalAfter:
    """Outcome"""
    def __init__(self, transitions=None, categories=None, verbose=False):
        self.transitions = transitions if transitions is not None else []
        self.categories = categories if categories is not None else []
        self.verbose = verbose

    def __call__(self, condition):
        category = condition()
        for trans, c in zip(self.transitions, self.categories):
            if c == category:
                trans.active = True
                if self.verbose:
                    print("Activating {}.".format(trans))


class CausalGraphState:
    success = 1
    failure = 2


class CausalGraphTraverser:
    def __init__(self, root, verbose=False):
        self.root = root
        self.verbose = verbose
        self.state = None
        # This is a legacy attribute to work with old termination conditions.
        self.last_wake_time = 0

    def get_events(self):
        to_traverse = {self.root}
        traversed = set()
        while to_traverse:
            event = to_traverse.pop()
            if event.outcome:
                for trans in event.outcome.transitions:
                    if trans.dest not in traversed:
                        to_traverse.add(trans.dest)
            traversed.add(event)
        return traversed

    def reset(self):
        self.state = None
        self.last_wake_time = 0
        to_reset = {self.root}
        reset = set()
        while to_reset:
            event = to_reset.pop()
            event.reset()
            if event.outcome:
                for trans in event.outcome.transitions:
                    trans.active = False
                    if trans.dest not in reset:
                        to_reset.add(trans.dest)
            reset.add(event)

    def update(self, time):
        if self.state in (CausalGraphState.success, CausalGraphState.failure):
            return
        failed = False
        awake = False
        to_process = {self.root}
        while to_process:
            event = to_process.pop()
            event.update(time)
            if event.state is EventState.success and event.outcome:
                for trans in event.outcome.transitions:
                    if trans.active:
                        to_process.add(trans.dest)
            elif event.state is EventState.failure:
                failed = True
            elif event.state is EventState.awake:
                awake = True
                # Legacy
                self.last_wake_time = max(event.wake_time, self.last_wake_time)
        if not awake:
            if failed:
                self.state = CausalGraphState.failure
                if self.verbose:
                    print("Failure of {} with {}".format(self, event))
            else:
                self.state = CausalGraphState.success
                if self.verbose:
                    print("Success of {} with {}".format(self, event))


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

    def render(self, filename=None):
        filename = filename if filename else self._file.name
        g = Digraph('G', filename=filename)
        to_process = {self.root}
        processed = set()
        while to_process:
            event = to_process.pop()
            if event in processed:
                continue
            g.node(event.name, style='filled', color='black',
                   fillcolor=self.colors[event.state])
            if event.outcome:
                for trans in event.outcome.transitions:
                    g.edge(event.name, trans.dest.name,
                           label=str(int(trans.active)))
                    to_process.add(trans.dest)
            processed.add(event)
        g.view()


def connect(source, dest):
    """Connect two events.

    source.outcome and dest.precondition must exist.

    """
    trans = Transition(source, dest)
    source.outcome.transitions.append(trans)
    dest.precondition.transitions.append(trans)
