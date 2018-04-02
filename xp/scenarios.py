"""
Various scenarios used to learn interactions between objects.

"""
from math import cos, radians, sin
from random import uniform

import chaospy as cp
import numpy as np
from panda3d.core import NodePath, Point3, Vec3
from shapely.affinity import rotate, translate
from shapely.geometry import box

import core.primitives as prim
import xp.config as cfg
import xp.dominoes.geom as dom


def init_scene():
    """Initialize the Panda3D scene."""
    scene = NodePath("scene")
    world = prim.World()
    world.set_gravity(cfg.GRAVITY)
    floor = prim.Plane(
            "floor", geom=False,
            friction=cfg.FLOOR_MATERIAL_FRICTION,
            restitution=cfg.FLOOR_MATERIAL_RESTITUTION)
    floor.create()
    floor.attach_to(scene, world)
    return scene, world


def add_domino_run(
        name, coords, scene, world, tilt_first_dom=False, make_geom=False):
    domino_factory = prim.DominoRun(
            name,
            (cfg.t, cfg.w, cfg.h), coords, geom=make_geom, mass=cfg.MASS,
            friction=cfg.DOMINO_MATERIAL_FRICTION,
            restitution=cfg.DOMINO_MATERIAL_RESTITUTION,
            angular_damping=cfg.DOMINO_ANGULAR_DAMPING
            )
    dom_np = domino_factory.create()
    domino_factory.attach_to(scene, world)
    if tilt_first_dom:
        # Set initial conditions for first domino
        first_domino = dom_np.get_child(0)
        dom.tilt_box_forward(first_domino, cfg.TOPPLING_ANGLE+.5)
        first_domino.node().set_transform_dirty()
        # Alternative with initial velocity:
        #  angvel_init = Vec3(0., 15., 0.)
        #  angvel_init = Mat3.rotate_mat(headings[0]).xform(angvel_init)
        #  first_domino.node().set_angular_velocity(angvel_init)
    return dom_np


def add_ball(name, pos, scene, world, make_geom=False):
    ball_maker = prim.Ball(
            name, cfg.BALL_RADIUS, geom=make_geom,
            mass=cfg.BALL_MASS,
            friction=cfg.BALL_MATERIAL_FRICTION,
            restitution=cfg.BALL_MATERIAL_RESTITUTION,
            linear_damping=cfg.BALL_LINEAR_DAMPING,
            angular_damping=cfg.BALL_ANGULAR_DAMPING
            )
    ball_np = ball_maker.create()
    ball_np.set_pos(pos)
    ball_maker.attach_to(scene, world)
    return ball_np


def add_plank(name, pos, hpr, scene, world, make_geom=False):
    plank_maker = prim.Box(
            name, (cfg.PLANK_LENGTH, cfg.PLANK_WIDTH, cfg.PLANK_THICKNESS),
            geom=make_geom,
            friction=cfg.PLANK_MATERIAL_FRICTION,
            restitution=cfg.PLANK_MATERIAL_RESTITUTION,
            )
    plank_np = plank_maker.create()
    plank_np.set_pos_hpr(pos, hpr)
    plank_maker.attach_to(scene, world)
    return plank_np


def perturbate_dominoes(coords, randfactor, maxtrials=10):
    """Randomly perturbate dominoes, but keep the distribution valid.

    Parameters
    ----------
    coords : (n,3) numpy array
      Global X, Y and heading of each domino.
    randfactor : float
      Randomization factor: fraction of the 'reasonable' range for each
      coordinate, which will be used as the parameter of the uniform
      distribution used for the randomization.

    Returns
    -------
    new_coords : (n,3) numpy array
      New coordinates.

    """
    base = box(-cfg.t/2, -cfg.w/2, cfg.t/2,  cfg.w/2)
    dominoes = [translate(rotate(base, ai), xi, yi) for xi, yi, ai in coords]
    new_coords = np.empty_like(coords)
    rng_x = cfg.X_MAX * randfactor
    rng_y = cfg.Y_MAX * randfactor
    rng_a = cfg.A_MAX * randfactor
    for i in range(len(dominoes)):
        ntrials = 0
        while ntrials < maxtrials:
            new_coords[i] = coords[i] + [uniform(-rng_x, rng_x),
                                         uniform(-rng_y, rng_y),
                                         uniform(-rng_a, rng_a)]
            dominoes[i] = translate(
                    rotate(base, new_coords[i, 2]), *new_coords[i, :2])

            # Find first domino to intersect the current domino
            try:
                next(di for di in dominoes
                     if di is not dominoes[i] and di.intersects(dominoes[i]))
            except StopIteration:
                # No domino intersects the current one
                break
            ntrials += 1
        else:
            # Valid perturbated coordinates could not be found in time.
            new_coords[i] = coords[i]
            dominoes[i] = translate(
                    rotate(base, new_coords[i, 2]), *new_coords[i, :2])
    return new_coords


def has_toppled(domino):
    return domino.get_r() >= cfg.TOPPLING_ANGLE


class DominoRunTerminationCondition:
    def __init__(self, domrun_np, verbose=False):
        self._domrun_np = domrun_np
        self.verbose = verbose
        self.reset()

    def __repr__(self):
        r = "{} on {}".format(type(self).__name__, self._domrun_np.get_name())
        return r

    def reset(self):
        self.status = None
        self.terminated = False
        self.last_event_time = 0
        # Internals
        self._domlist = list(self._domrun_np.get_children())

    def has_started(self):
        return has_toppled(self._domrun_np.get_child(0))

    def update_and_check(self, time):
        # Avoid checking if already terminated.
        if self.terminated:
            return True
        # Check start condition.
        if self.status is None:
            if self.has_started():
                self.status = 'started'
                self.last_event_time = time
                if self.verbose:
                    print(self._domrun_np.get_name(), "has started to topple")
            else:
                return False
        # Update internal state.
        # Use 'while' here because close dominoes can topple at the same time.
        while (self._domlist and has_toppled(self._domlist[0])):
            self._domlist.pop(0)
            self.last_event_time = time
        # Update status.
        if not self._domlist:
            # All dominoes toppled in order.
            self.status = 'success'
            self.terminated = True
            if self.verbose:
                print(self._domrun_np.get_name(), "has completely toppled")
        elif time - self.last_event_time > cfg.MAX_WAIT_TIME:
            # The chain broke.
            self.status = 'failure'
            self.terminated = True
            if self.verbose:
                print(self._domrun_np.get_name(), "has timed out")
        return self.terminated


class MoveCollideTerminationCondition:
    def __init__(self, world, a, b, verbose):
        self._world = world
        self._a = a
        self._b = b
        self.verbose = verbose
        self.reset()

    def __repr__(self):
        r = "{} between {} and {}".format(
            type(self).__name__,
            self._a.get_name(),
            self._b.get_name()
        )
        return r

    def reset(self):
        self.status = None
        self.terminated = False
        self.last_event_time = 0

    def has_started(self):
        body = self._a.node()
        lin = body.get_linear_velocity().length_squared()
        rot = body.get_angular_velocity().length_squared()
        if lin + rot > 1e-2:
            return True

    def update_and_check(self, time):
        # Avoid checking if already terminated.
        if self.terminated:
            return True
        # Check start condition.
        if self.status is None:
            if self.has_started():
                self.status = 'started'
                self.last_event_time = time
                if self.verbose:
                    print(self._a.get_name(), "has started to move")
            else:
                return False
        # Update internal state.
        result = self._world.contact_test_pair(self._a.node(), self._b.node())
        # Update status.
        if result.get_num_contacts():
            self.status = 'success'
            self.terminated = True
            self.last_event_time = time
            if self.verbose:
                print(self._a.get_name(), "has hit", self._b.get_name())
        if time - self.last_event_time > cfg.MAX_WAIT_TIME:
            # The chain broke.
            self.status = 'failure'
            self.terminated = True
            if self.verbose:
                print("Collision between {} and {} has timed out".format(
                    self._a.get_name(), self._b.get_name())
                )
        return self.terminated


class AndTerminationCondition:
    """This condition terminates when all sub conditions have terminated."""
    def __init__(self, conditions, verbose=False):
        self.conditions = conditions
        self.verbose = verbose
        self.reset()

    def __repr__(self):
        r = "{} between ".format(type(self).__name__)
        for c in self.conditions[:-1]:
            r += "{}, ".format(c)
        r += "and {}".format(self.conditions[-1])
        return r

    def reset(self):
        self.status = None
        self.terminated = False
        self.last_event_time = 0
        for c in self.conditions:
            c.reset()

    def has_started(self):
        return any(c.has_started() for c in self.conditions)

    def update_and_check(self, time):
        # Avoid checking if already terminated.
        if self.terminated:
            return True
        # Check start condition.
        if self.status is None:
            if self.has_started():
                self.status = 'started'
                self.last_event_time = time
                if self.verbose:
                    start = next(c for c in self.conditions if c.has_started())
                    print("One condition has started:", start)
            else:
                return False
        # Update internal state.
        terminated = True
        for c in self.conditions:
            terminated &= c.update_and_check(time)
            if c.last_event_time > self.last_event_time:
                self.last_event_time = c.last_event_time
        # Update status.
        if terminated:
            success = not any(c.status == 'failure' for c in self.conditions)
            self.status = 'success' if success else 'failure'
            self.terminated = True
            if self.verbose:
                if success:
                    print("All conditions have successfully terminated")
                else:
                    fail = [c for c in self.conditions
                            if c.status == 'failure']
                    print("The following conditions were not met:", fail)
        elif time - self.last_event_time > cfg.MAX_WAIT_TIME:
            # The chain broke.
            self.status = 'failure'
            self.terminated = True
            if self.verbose:
                noterm = [c for c in self.conditions if not c.terminated]
                print("The following conditions have not terminated:", noterm)
        return self.terminated


class DummyTerminationCondition:
    def __init__(self):
        self.reset()

    def reset(self):
        self.status = None
        self.terminated = False
        self.last_event_time = 0

    def has_started(self):
        return True

    def update_and_check(self, time):
        return False


class CausalGraphTerminationCondition:
    def __init__(self, causal_graph):
        self.causal_graph = causal_graph
        self.reset()

    def reset(self):
        self.status = None
        self.terminated = False
        self.last_event_time = 0
        self.causal_graph.reset()

    def update_and_check(self, time):
        # Avoid checking if already terminated.
        if self.terminated:
            return True
        self.causal_graph.update(time)
        self.last_event_time = self.causal_graph.last_wake_time
        if self.causal_graph.state == 1:
            self.status = 'success'
            self.terminated = True
        elif self.causal_graph.state == 2:
            self.status = 'failure'
            self.terminated = True
        return self.terminated


class DominoRunTopplingTimeObserver:
    """Observes the toppling time of each domino in a run.

    Attributes
    ----------
    times : (n,) ndarray
      The toppling time of each domino. A domino that doesn't topple has
      a time equal to np.inf.

    """
    def __init__(self, domrun_np):
        self.dominoes = list(domrun_np.get_children())
        self.times = np.full(len(self.dominoes), np.inf)
        self.last_toppled_id = -1

    def __call__(self, time):
        n = len(self.dominoes)
        # Use while here because close dominoes can topple at the same time
        while (self.last_toppled_id < n-1
                and has_toppled(self.dominoes[self.last_toppled_id+1])):
            self.last_toppled_id += 1
            self.times[self.last_toppled_id] = time


class Samplable:
    """Samplable scenario"""
    @staticmethod
    def get_distribution():
        raise NotImplementedError

    @classmethod
    def sample(cls, n, rule='R'):
        if rule == 'R':
            cp.seed(n)
        return cls.get_distribution().sample(n, rule=rule).T

    @classmethod
    def sample_valid(cls, n, max_trials=None, rule='R'):
        if max_trials is None:
            max_trials = 2 * n
        cand_samples = cls.sample(max_trials, rule)
        samples = np.empty((n, cand_samples.shape[1]))
        n_valid = 0
        for sample in cand_samples:
            if cls.check_valid(*cls.init_scenario(sample)):
                samples[n_valid] = sample
                n_valid += 1
                if n_valid == n:
                    break
        else:
            print("Ran out of trials")
        return samples


class Scenario:
    """Base class for all scenarios."""
    def check_physically_valid(self):
        return self.check_valid(self.scene, self.world)

    def succeeded(self):
        return self.terminate.status == 'success'


class TwoDominoesLastRadial(Samplable, Scenario):
    """The center of D2 lies on the line orthogonal to D1's largest face and
    going through D1's center. Here for demos, not interesting otherwise.

    Samplable with 2 parameters (distance between centers and
    relative heading angle).

    """
    def __init__(self, sample, make_geom=False, **kwargs):
        self.scene, self.world = self.init_scenario(sample, make_geom)
        domrun = self.scene.find("domino_run")
        self.terminate = DominoRunTerminationCondition(domrun)

    @staticmethod
    def check_valid(scene, world):
        domrun = scene.find("domino_run")
        d1 = domrun.get_child(0).node()
        d2 = domrun.get_child(1).node()
        return world.contact_test_pair(d1, d2).get_num_contacts() == 0

    @staticmethod
    def get_distribution():
        dist_x = cp.Uniform(cfg.X_MIN, cfg.X_MAX)
        dist_a = cp.Uniform((0 if cfg.REDUCE_SYM else cfg.A_MIN), cfg.A_MAX)
        return cp.J(dist_x, dist_a)

    @staticmethod
    def get_parameters(scene):
        d1, d2 = scene.find("domino_run").get_children()
        # Untilt 1st dom before computing the distance.
        tilt = d1.get_r()
        dom.tilt_box_forward(d1, -tilt)
        d = (d2.get_pos() - d1.get_pos()).length()
        # Retilt 1st dom.
        dom.tilt_box_forward(d1, tilt)
        a = d2.get_h() - d1.get_h()
        return d, a

    @staticmethod
    def init_scenario(sample, make_geom=False):
        scene, world = init_scene()
        coords = np.zeros((2,  3))
        angle = radians(sample[1])
        coords[1] = sample[0]*cos(angle), sample[0]*sin(angle), sample[1]
        add_domino_run("domino_run", coords, scene, world, True, make_geom)
        return scene, world


class TwoDominoesLastFree(Samplable, Scenario):
    """D2 can be wherever wrt D1 (within config bounds).

    Samplable with 3 parameters (relative position in the XY plane and
    relative heading angle).

    """
    def __init__(self, sample, make_geom=False, **kwargs):
        self.scene, self.world = self.init_scenario(sample, make_geom)
        domrun = self.scene.find("domino_run")
        self.terminate = DominoRunTerminationCondition(domrun)

    check_valid = TwoDominoesLastRadial.check_valid

    @staticmethod
    def get_distribution():
        #  dist_x = cp.Uniform(X_MIN, X_MAX)
        #  dist_y = cp.Uniform((0 if cfg.REDUCE_SYM else Y_MIN), Y_MAX)
        #  dist_a = cp.Uniform(A_MIN, A_MAX)
        dist_x = cp.Truncnorm(cfg.X_MIN,
                              cfg.X_MAX,
                              (cfg.X_MIN+cfg.X_MAX)/2,
                              (cfg.X_MAX-cfg.X_MIN)/4)
        if cfg.REDUCE_SYM:
            dist_y = cp.Truncnorm((cfg.Y_MIN+cfg.Y_MAX)/2,
                                  cfg.Y_MAX,
                                  (cfg.Y_MIN+cfg.Y_MAX)/2,
                                  (cfg.Y_MAX-cfg.Y_MIN)/4)
        else:
            dist_y = cp.Truncnorm(cfg.Y_MIN,
                                  cfg.Y_MAX,
                                  (cfg.Y_MIN+cfg.Y_MAX)/2,
                                  (cfg.Y_MAX-cfg.Y_MIN)/4)
        dist_a = cp.Truncnorm(cfg.A_MIN,
                              cfg.A_MAX,
                              (cfg.A_MIN+cfg.A_MAX)/2,
                              (cfg.A_MAX-cfg.A_MIN)/4)
        return cp.J(dist_x, dist_y, dist_a)

    @staticmethod
    def get_parameters(scene):
        d1, d2 = scene.find("domino_run").get_children()
        # Untilt 1st dom before computing the distance.
        tilt = d1.get_r()
        dom.tilt_box_forward(d1, -tilt)
        x, y, _ = d2.get_pos() - d1.get_pos()
        # Retilt 1st dom.
        dom.tilt_box_forward(d1, tilt)
        a = d2.get_h() - d1.get_h()
        return x, y, a

    @staticmethod
    def init_scenario(sample, make_geom=False):
        scene, world = init_scene()
        coords = np.zeros((2,  3))
        coords[1] = sample
        add_domino_run("domino_run", coords, scene, world, True, make_geom)
        return scene, world


class DominoesStraightLastFree(Samplable, Scenario):
    """Any number of dominoes on a straight line, the last being free.

    Samplable with 4 DoFs (relative position in the XY plane,
    relative heading angle and spacing between previous doms).

    """
    def __init__(self, sample, make_geom=False, **kwargs):
        nprev = kwargs.pop('nprev', 0)
        self.scene, self.world = self.init_scenario(sample, nprev, make_geom)
        domrun = self.scene.find("domino_run")
        self.terminate = DominoRunTerminationCondition(domrun)

    @staticmethod
    def check_valid(scene, world):
        dominoes = [d.node() for d in scene.find("domino_run").get_children()]
        for d1, d2 in zip(dominoes[:-1], dominoes[1:]):
            if world.contact_test_pair(d1, d2).get_num_contacts():
                return False
        return True

    @staticmethod
    def get_distribution():
        #  dist_x = cp.Uniform(X_MIN, X_MAX)
        #  dist_y = cp.Uniform((0 if cfg.REDUCE_SYM else Y_MIN), Y_MAX)
        #  dist_a = cp.Uniform(A_MIN, A_MAX)
        dist_x = cp.Truncnorm(cfg.X_MIN,
                              cfg.X_MAX,
                              (cfg.X_MIN+cfg.X_MAX)/2,
                              (cfg.X_MAX-cfg.X_MIN)/4)
        if cfg.REDUCE_SYM:
            dist_y = cp.Truncnorm((cfg.Y_MIN+cfg.Y_MAX)/2,
                                  cfg.Y_MAX,
                                  (cfg.Y_MIN+cfg.Y_MAX)/2,
                                  (cfg.Y_MAX-cfg.Y_MIN)/4)
        else:
            dist_y = cp.Truncnorm(cfg.Y_MIN,
                                  cfg.Y_MAX,
                                  (cfg.Y_MIN+cfg.Y_MAX)/2,
                                  (cfg.Y_MAX-cfg.Y_MIN)/4)
        dist_a = cp.Truncnorm(cfg.A_MIN,
                              cfg.A_MAX,
                              (cfg.A_MIN+cfg.A_MAX)/2,
                              (cfg.A_MAX-cfg.A_MIN)/4)
        dist_s = cp.Truncnorm(cfg.MIN_SPACING,
                              cfg.MAX_SPACING,
                              (cfg.MIN_SPACING+cfg.MAX_SPACING)/2,
                              (cfg.MAX_SPACING-cfg.MIN_SPACING)/4)
        return cp.J(dist_x, dist_y, dist_a, dist_s)

    @staticmethod
    def get_parameters(scene):
        doms = list(scene.find("domino_run").get_children())
        # Untilt 1st dom before computing the distance.
        tilt = doms[0].get_r()
        dom.tilt_box_forward(doms[0], -tilt)
        x, y, _ = doms[-1].get_pos() - doms[-2].get_pos()
        a = doms[-1].get_h() - doms[-2].get_h()
        prev = doms[:-2]
        if prev:
            s = sum(
                (d2.get_pos() - d1.get_pos()).length()
                for d1, d2 in zip(prev[:-1], prev[1:])
            ) / (len(prev) - 1)
        else:
            s = 0.
        # Retilt 1st dom.
        dom.tilt_box_forward(doms[0], tilt)
        return x, y, a, s

    @staticmethod
    def init_scenario(sample, nprev=0, make_geom=False):
        scene, world = init_scene()
        length = sample[3] * nprev
        coords = np.zeros((nprev+2, 3))
        coords[:-2, 0] = length * np.linspace(-1, 0, nprev+1)[:-1]
        coords[-1] = sample[:3]
        add_domino_run("domino_run", coords, scene, world, True, make_geom)
        return scene, world


class DominoesStraightTwoLastFree(Samplable, Scenario):
    """Any number of dominoes in a straight line, the last two being free.

    Samplable with 6 DoFs (relative transforms of domino 2 vs 1 and 3 vs 2.)

    """
    def __init__(self, sample, make_geom=False, **kwargs):
        nprev = kwargs.pop('nprev', 0)
        self.scene, self.world = self.init_scenario(sample, nprev, make_geom)
        domrun = self.scene.find("domino_run")
        self.terminate = DominoRunTerminationCondition(domrun)

    check_valid = DominoesStraightLastFree.check_valid

    @staticmethod
    def get_distribution():
        dist_x1 = cp.Truncnorm(cfg.X_MIN,
                               cfg.X_MAX,
                               (cfg.X_MIN+cfg.X_MAX)/2,
                               (cfg.X_MAX-cfg.X_MIN)/4)
        if cfg.REDUCE_SYM:
            dist_y1 = cp.Truncnorm((cfg.Y_MIN+cfg.Y_MAX)/2,
                                   cfg.Y_MAX,
                                   (cfg.Y_MIN+cfg.Y_MAX)/2,
                                   (cfg.Y_MAX-cfg.Y_MIN)/4)
        else:
            dist_y1 = cp.Truncnorm(cfg.Y_MIN,
                                   cfg.Y_MAX,
                                   (cfg.Y_MIN+cfg.Y_MAX)/2,
                                   (cfg.Y_MAX-cfg.Y_MIN)/4)
        dist_a1 = cp.Truncnorm(cfg.A_MIN,
                               cfg.A_MAX,
                               (cfg.A_MIN+cfg.A_MAX)/2,
                               (cfg.A_MAX-cfg.A_MIN)/4)
        dist_x2 = cp.Truncnorm(cfg.X_MIN,
                               cfg.X_MAX,
                               (cfg.X_MIN+cfg.X_MAX)/2,
                               (cfg.X_MAX-cfg.X_MIN)/4)
        dist_y2 = cp.Truncnorm(cfg.Y_MIN,
                               cfg.Y_MAX,
                               (cfg.Y_MIN+cfg.Y_MAX)/2,
                               (cfg.Y_MAX-cfg.Y_MIN)/4)
        dist_a2 = cp.Truncnorm(cfg.A_MIN,
                               cfg.A_MAX,
                               (cfg.A_MIN+cfg.A_MAX)/2,
                               (cfg.A_MAX-cfg.A_MIN)/4)
        return cp.J(dist_x1, dist_y1, dist_a1, dist_x2, dist_y2, dist_a2)
        # TODO. See if we this is functionally equivalent to the simpler:
        #  dist_xya1 = TwoDominoesLastFree.get_distribution()
        #  redsym = cfg.REDUCE_SYM
        #  cfg.REDUCE_SYM = False
        #  dist_xya2 = TwoDominoesLastFree.get_distribution()
        #  cfg.REDUCE_SYM = redsym
        #  return cp.J(dist_xya1, dist_xya2)

    @staticmethod
    def get_parameters(scene):
        doms = list(scene.find("domino_run").get_children())
        *_, d1, d2, d3 = doms[-3:]
        # Untilt 1st dom before computing the distance.
        tilt = doms[0].get_r()
        dom.tilt_box_forward(doms[0], -tilt)
        x1, y1, _ = d2.get_pos() - d1.get_pos()
        a1 = d2.get_h() - d1.get_h()
        x2, y2, _ = d3.get_pos(d2)
        a2 = d3.get_h() - d2.get_h()
        # Retilt 1st dom.
        dom.tilt_box_forward(doms[0], tilt)
        return x1, y1, a1, x2, y2, a2

    @staticmethod
    def init_scenario(sample, nprev=0, make_geom=False):
        scene, world = init_scene()
        length = nprev * cfg.h / 3
        coords = np.zeros((nprev+3, 3))
        coords[:-3, 0] = length * np.linspace(-1, 0, nprev+1)[:-1]
        coords[-2] = sample[:3]
        # Put third domino in the referential of the first domino
        angle = radians(sample[2])
        cos_ = cos(angle)
        sin_ = sin(angle)
        rot = np.array([[cos_, sin_], [-sin_, cos_]])
        coords[-1, :2] = sample[:2] + sample[3:5].dot(rot)
        coords[-1, 2] = sample[2] + sample[5]
        add_domino_run("domino_run", coords, scene, world, True, make_geom)
        return scene, world


class CustomDominoRun(Scenario):
    """A custom domino run.

    Not samplable.

    Parameters
    ----------
    sample : (n, 3) ndarray
      Coordinates of the n dominoes in the run.
    make_geom : bool, optional
      Whether to add a geometry or not.

    """
    def __init__(self, sample, make_geom=False, **kwargs):
        self.scene, self.world = self.init_scenario(sample, make_geom)
        domrun = self.scene.find("domino_run")
        self.terminate = DominoRunTerminationCondition(domrun)

    check_valid = DominoesStraightLastFree.check_valid

    @staticmethod
    def init_scenario(sample, make_geom=False):
        scene, world = init_scene()
        add_domino_run("domino_run", sample, scene, world, True, make_geom)
        return scene, world


class BallPlankDominoes(Samplable, Scenario):
    """A ball rolls on a plank and hits a straight row of dominoes.

    Samplable with 3 DoFs (x, y (of lower right corner wrt the 1st domino)
    and angle of the plank).

    """
    def __init__(self, sample, make_geom=False, **kwargs):
        ndoms = kwargs.pop('ndoms', 1)
        self.scene, self.world = self.init_scenario(sample, ndoms, make_geom)
        domrun = self.scene.find("domino_run")
        self.terminate = DominoRunTerminationCondition(domrun)

    @staticmethod
    def check_valid(scene, world):
        floor = scene.find("floor_solid").node()
        plank = scene.find("plank_solid").node()
        contact = world.contact_test_pair(floor, plank).get_num_contacts()
        if contact:
            return False
        for domino in scene.find("domino_run").get_children():
            contact = world.contact_test_pair(
                    plank, domino.node()).get_num_contacts()
            if contact:
                return False
        return True

    @staticmethod
    def get_distribution():
        dist_x = cp.Truncnorm(cfg.PLANK_X_MIN,
                              cfg.PLANK_X_MAX,
                              (cfg.PLANK_X_MIN+cfg.PLANK_X_MAX)/2,
                              (cfg.PLANK_X_MAX-cfg.PLANK_X_MIN)/4)
        dist_y = cp.Truncnorm(cfg.PLANK_Y_MIN,
                              cfg.PLANK_Y_MAX,
                              (cfg.PLANK_Y_MIN+cfg.PLANK_Y_MAX)/2,
                              (cfg.PLANK_Y_MAX-cfg.PLANK_Y_MIN)/4)
        dist_a = cp.Truncnorm(cfg.PLANK_A_MIN,
                              cfg.PLANK_A_MAX,
                              (cfg.PLANK_A_MIN+cfg.PLANK_A_MAX)/2,
                              (cfg.PLANK_A_MAX-cfg.PLANK_A_MIN)/4)
        return cp.J(dist_x, dist_y, dist_a)

    @staticmethod
    def get_parameters(scene):
        dom = scene.find("domino_run").get_child(0)
        plank = scene.find("plank*")
        corner = NodePath("corner")
        corner.set_pos(plank,
                       Vec3(cfg.PLANK_LENGTH/2, 0, -cfg.PLANK_THICKNESS/2))
        x, _, y = corner.get_pos() - Vec3(dom.get_x(), dom.get_y(), 0)
        a = - plank.get_r()
        return x, y, a

    @staticmethod
    def init_scenario(sample, ndoms=1, make_geom=False):
        scene, world = init_scene()
        # Plank
        angle = radians(sample[2])
        cos_ = cos(angle)
        sin_ = sin(angle)
        rot_t = np.array([[cos_, sin_], [-sin_, cos_]])
        corner_abs = np.asarray(sample[:2])
        corner_rel = np.array([cfg.PLANK_LENGTH/2, -cfg.PLANK_THICKNESS/2])
        center = corner_abs - corner_rel.dot(rot_t)
        pos = Point3(center[0], 0, center[1])
        hpr = Vec3(0, 0, -sample[2])
        add_plank("plank", pos, hpr, scene, world, make_geom)
        # Ball
        center += -corner_rel.dot(rot_t) + [1e-2, cfg.BALL_RADIUS+1e-3]
        pos = Point3(center[0], 0, center[1])
        add_ball("ball", pos, scene, world, make_geom)
        # Dominoes
        coords = np.zeros((ndoms, 3))
        length = ndoms * cfg.h / 3
        coords[:, 0] = np.linspace(0, length, ndoms)
        add_domino_run("domino_run", coords, scene, world, False, make_geom)
        return scene, world


SCENARIOS = (
        TwoDominoesLastRadial,
        TwoDominoesLastFree,
        DominoesStraightLastFree,
        DominoesStraightTwoLastFree,
        BallPlankDominoes,
        )
