from enum import Enum

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
                if self.outcome:
                    self.outcome(self.condition)
                if self.verbose:
                    print("{} has happened.".format(self))
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
        self.last_wake_time = 0

    def reset(self):
        self.state = None
        self.last_wake_time = 0
        to_reset = [self.root]
        while to_reset:
            event = to_reset.pop(0)
            event.reset()
            if event.outcome:
                for trans in event.outcome.transitions:
                    to_reset.append(trans.dest)

    def update(self, time):
        if self.state in (CausalGraphState.success, CausalGraphState.failure):
            return
        to_process = [self.root]
        while to_process:
            event = to_process.pop(0)
            event.update(time)
            if event.state is EventState.success:
                if event.outcome:
                    for trans in event.outcome.transitions:
                        if trans.active:
                            to_process.append(trans.dest)
                else:
                    self.state = CausalGraphState.success
                    if self.verbose:
                        print("Success of {} with {}".format(self, event))
                    break
            elif event.state is EventState.failure:
                self.state = CausalGraphState.failure
                if self.verbose:
                    print("Failure of {} with {}".format(self, event))
                break
            else:
                self.last_wake_time = max(event.wake_time, self.last_wake_time)


def connect(source, dest):
    """Connect two events.

    source.outcome and dest.precondition must exist.

    """
    trans = Transition(source, dest)
    source.outcome.transitions.append(trans)
    dest.precondition.transitions.append(trans)