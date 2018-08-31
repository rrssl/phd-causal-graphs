import pickle
import subprocess
from itertools import count

from panda3d.core import NodePath, TransformState

import core.config as cfg
from core import primitives
from core.causal_graph import CausalGraphViewer



class LegacyScenario:
    """Base class for new scenarios.

    Parameters
    ----------
    sample : (-,) sequence
      []

    """
    def __init__(self, sample, make_geom=False, graph_view=False, **kwargs):
        self.scene = self.init_scene(sample, make_geom)
        self.causal_graph = self.init_causal_graph(self.scene,
                                                   verbose=make_geom)
        if graph_view:
            self.graph_view = CausalGraphViewer(self.causal_graph.root)

    def check_physically_valid(self):
        return self._check_physically_valid_scene(self._scene)

    @classmethod
    def check_physically_valid_sample(cls, sample):
        scene = cls.init_scene(sample, make_geom=False)
        return cls._check_physically_valid_scene(scene)

    @staticmethod
    def _check_physically_valid_scene(scene):
        raise NotImplementedError

    def export_scene_to_egg(self, filename):
        if filename[-4:] == ".egg":
            filename = filename[:-4]
        self.scene.graph.write_bam_file(filename + ".bam")
        subprocess.run(["bam2egg", "-o", filename + ".egg", filename + ".bam"])

    @classmethod
    def export_scene_to_pdf(cls, filename, sample, sheetsize):
        raise NotImplementedError

    @classmethod
    def get_physical_validity_constraint(cls, sample):
        raise NotImplementedError

    @staticmethod
    def init_causal_graph(scene, verbose=False):
        raise NotImplementedError

    @classmethod
    def init_scene(cls, sample, make_geom=False):
        raise NotImplementedError


class Scene:
    def __init__(self, graph: NodePath, world: primitives.World):
        self.graph = graph
        self.world = world

    def check_physically_valid(self):
        return True

    def export_scene_to_egg(self, filename):
        if filename[-4:] == ".egg":
            filename = filename[:-4]
        self.graph.write_bam_file(filename + ".bam")
        subprocess.run(["bam2egg", "-o", filename + ".egg", filename + ".bam"])

    def export_layout_to_pdf(self, filename, sheetsize):
        pass

    def get_physical_validity_constraint(self):
        return 0


class Scenario:
    def __init__(self, scene: Scene, causal_graph, domain):
        self.scene = scene
        self.causal_graph = causal_graph
        self.domain = domain


class StateObserver:
    """Keeps track of the full state of each non-static object in the scene."""
    def __init__(self, scene: Scene):
        self.graph_root = scene.graph
        self.paths = []
        self.states = dict()
        self._prev_states = dict()
        self.key_gen = count()
        # Find and add nodes that are already tagged.
        for path in self.graph_root.find_all_matches("**/=anim_id"):
            self.states[path.get_name()] = []
            self.paths.append(path)
        # Find and tag any non-static BulletRigidBodyNode.
        for body in scene.world.get_rigid_bodies():
            if not body.is_static():
                path = self.graph_root.any_path(body)
                self.states[path.get_name()] = []
                self.paths.append(path)

    def __call__(self, time):
        for path in self.paths:
            x, y, z = path.get_pos()
            w, i, j, k = path.get_quat()
            if path.has_tag('save_scale'):
                sx, sy, sz = path.get_scale()
                state = [time, x, y, z, w, i, j, k, sx, sy, sz]
            else:
                state = [time, x, y, z, w, i, j, k]
            key = path.get_name()
            prev = self._prev_states
            try:
                state_ = prev[key]
            except KeyError:
                self.states[key].append(state)
                prev[key] = state[1:]
                continue
            if any(abs(s - s_) >= 1e-3 for s, s_ in zip(state[1:], state_)):
                self.states[key].append(state)
                prev[key] = state[1:]

    def export(self, filename, **metadata):
        if filename[-4:] != ".pkl":
            filename += ".pkl"
        data = {'metadata': metadata, 'states': self.states}
        with open(filename, 'wb') as f:
            pickle.dump(data, f)


def init_scene():
    """Initialize the Panda3D scene."""
    graph = NodePath("scene")
    world = primitives.World()
    world.set_gravity(cfg.GRAVITY)
    return Scene(graph, world)


def load_scene(scene_data, geom='LD', phys=True):
    scene = init_scene()
    name2path = {}
    path2parent = {}
    # First pass: create scene graph nodes
    for obj_data in scene_data:
        name = obj_data['name']
        # Instantiate factory
        PrimitiveType = getattr(primitives, obj_data['type'])
        geom_args = {}
        bullet_args = {}
        for k, v in obj_data['args'].items():
            if k.startswith("b_"):
                bullet_args[k[2:]] = v
            else:
                geom_args[k] = v
        obj = PrimitiveType(
            name, **geom_args, geom=geom, phys=phys, **bullet_args
        )
        # Create object
        obj_path = obj.create()
        name2path[name] = obj_path
        # Set object transform
        try:
            xform = obj_data['xform']['value']
        except KeyError:
            xform = [0, 0, 0, 0, 0, 0]
        try:
            rel = obj_data['xform']['relative_xyz_units']
        except KeyError:
            rel = False
        if rel:
            bounds = obj_path.get_bounds()
            unit = (bounds.get_max() - bounds.get_min()) / 2
            xform[0] *= unit[0]
            xform[1] *= unit[1]
            xform[2] *= unit[2]
        obj_path.set_pos_hpr(*xform)
        # Attach object to the root
        obj.attach_to(scene.graph, scene.world)
        # Keep track of potential parent
        try:
            parent = obj_data['parent']
        except KeyError:
            parent = None
        if parent is not None:
            path2parent[obj_path] = parent
    # Second pass for scene graph hierarchy
    for obj_path, parent in path2parent.items():
        obj_path.reparent_to(name2path[parent])
        obj_path.node().set_transform_dirty()
    return scene


def load_domain(scene_data):
    domain = {}
    for obj_data in scene_data:
        name = obj_data['name']
        try:
            range_ = obj_data['xform']['range']
        except KeyError:
            range_ = None
        domain[name] = range_
    return domain


def load_causal_graph(graph_data, scene):
    return None


def load_scenario(scenario_data, geom='LD', phys=True):
    scene = load_scene(scenario_data['scene'], geom, phys)
    domain = load_domain(scenario_data['scene'])
    causal_graph = load_causal_graph(scenario_data['causal_graph'], scene)
    return Scenario(scene, causal_graph, domain)


def simulate_scene(scene: Scene, duration, timestep, callbacks=None):
    """Run the simulator for a given Scene.

    Parameters
    ----------
    scene : Scene
      The Scene to simulate.
    duration : float
      The maximum duration of the simulation (in seconds).
    timestep : float
      The simulator timestep.
    callbacks : callable sequence, optional
      A list of functions of time called _before_ each simulation step.
      After calling each of them, if at least one has returned False, the
      simulation exits.

    Return
    ------
    time : float
      The total time of the simulation.

    """
    world = scene.world
    if callbacks is None:
        callbacks = []
    time = 0.
    do_break = False
    while time <= duration:
        for c in callbacks:
            res = c(time)
            if res is not None and not res:
                do_break = True
        if do_break:
            break
        world.do_physics(timestep, 2, timestep)
        time += timestep
    # Transforms are globally cached by default. Out of the regular
    # Panda3D task process, we need to empty this cache by hand when
    # running a large number of simulations, to avoid memory overflow.
    TransformState.garbage_collect()
    return time
