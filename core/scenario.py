import hashlib
import importlib.util
import json
import pickle
import subprocess
from itertools import combinations, count

import networkx as nx
from panda3d.core import NodePath, TransformState

from . import config as cfg
from . import causal_graph as causal
from . import primitives
from .design_space import load_design_space


class Scene:
    def __init__(self, geom='LD', phys=True):
        self.geom = geom
        self.phys = phys
        self.graph = NodePath("scene")
        if phys:
            self.world = primitives.World()
            self.world.set_gravity(cfg.GRAVITY)
        else:
            self.world = None

    def check_physically_valid(self):
        return self.get_physical_validity_constraint() > -1e-3

    def export_scene_to_egg(self, filename):
        if filename[-4:] == ".egg":
            filename = filename[:-4]
        self.graph.write_bam_file(filename + ".bam")
        subprocess.run(["bam2egg", "-o", filename + ".egg", filename + ".bam"])

    def get_physical_validity_constraint(self):
        """Compute the sum of all physical constraint violations (<= 0)."""
        world = self.world
        constraint = 0
        # Get pulleys' constraint
        for pulley_cb in world._callbacks:
            constraint += min(0, pulley_cb.loose_rope)
        # Check unwanted collisions.
        bodies = world.get_rigid_bodies()
        # Enable collisions for static objects
        static = []
        for body in bodies:
            if body.is_static():
                static.append(body)
                body.set_static(False)
                body.set_active(True)
        # Check penetrations
        for a, b in combinations(bodies, 2):
            # contact_test_pair() ignores all collision flags, so we need
            # to check that these bodies are meant to collide.
            if a.check_collision_with(b):
                result = world.contact_test_pair(a, b)
                if result.get_num_contacts():
                    # Sometimes Bullet detects a collision even though the
                    # distance is positive. Ignore such positive distances.
                    penetration = sum(
                        min(0, c.get_manifold_point().get_distance())
                        for c in result.get_contacts()
                    )
                    # if penetration < 0:
                    #     print(a, b, penetration)
                    constraint += penetration
        # Re-disable collisions for static objects
        for body in static:
            body.set_static(True)
            body.set_active(False)
        # Same issue as described in simulate_scene
        TransformState.garbage_collect()
        return constraint

    def populate(self, prim_graph, xforms, velos):
        """Populate the scene with primitive instances.

        Parameters
        ----------
        prim_graph : networkx.DiGraph
          Tree of primitives reflecting the scene graph hierarchy (as given to
          Scenario).
        xforms : dict
          Dictionary of o_name: o_xform pairs, where o_xform is a 6-tuple
          corresponding to panda3d's (x,y,z,h,p,r) values.
        velos : dict
          Dictionary of o_name: o_velo pairs, where o_velo is a 6-tuple
          corresponding to bullet's linear and angular velocities.
          Only applies to simple primitives (i.e., no 'components' field).

        """
        graph = self.graph
        world = self.world
        name2nopa = {}
        # First pass: create and add simple objects.
        for name, prim in prim_graph.nodes(data='prim'):
            if 'components' not in prim_graph.node[name]:
                nopa = prim.create(self.geom, self.phys, graph, world,
                                   velos[name])
                xform = xforms[name]
                if xform is not None:
                    if len(xform) == 6:
                        nopa.set_pos_hpr(*xform)
                    elif len(xform) == 7:
                        nopa.set_pos_quat(tuple(xform[:3]), tuple(xform[3:]))
                if 'tags' in prim_graph.node[name]:
                    for tag, val in prim_graph.node[name]['tags'].items():
                        nopa.set_tag(tag, str(val))
                name2nopa[name] = nopa
        # Second pass: scene graph hierarchy.
        for parent, child in prim_graph.edges:
            name2nopa[child].reparent_to(name2nopa[parent])
        # Third pass: instantiate complex constructs. We assume only
        # one level of nesting (i.e., components are themselves simple).
        for name, prim in prim_graph.nodes(data='prim'):
            if 'components' in prim_graph.node[name]:
                comps = [name2nopa[c]
                         for c in prim_graph.node[name]['components']]
                nopa = prim.create(self.geom, self.phys, graph, world, comps)
                if nopa is not None:
                    xform = xforms[name]
                    if xform is not None:
                        if len(xform) == 6:
                            nopa.set_pos_hpr(*xform)
                        elif len(xform) == 7:
                            nopa.set_pos_quat(tuple(xform[:3]),
                                              tuple(xform[3:]))
                    if 'tags' in prim_graph.node[name]:
                        for tag, val in prim_graph.node[name]['tags'].items():
                            nopa.set_tag(tag, str(val))
                name2nopa[name] = nopa
        # Last pass: propagate new global transforms to bullet nodes.
        if self.phys:
            for body in world.get_rigid_bodies():
                body.set_transform_dirty()


class Scenario:
    """Abstract representation of a scenario.

    This is just a convenient, "stateless" high-level representation,
    not unlike a dict-like specification but with a number of preprocessing
    operations already done.
    Specifying a position in the design space yields a ScenarioInstance that
    can be used for simulation and visualization.

    Parameters
    ----------
    prim_graph : network.DiGraph
      Tree of primitives reflecting the scene graph hierarchy.
    causal_graph : network.DiGraph
      Directed acyclic graph of events.
    design_space : design_space.DesignSpace
      Design space of the scenario.

    """
    def __init__(self, prim_graph, causal_graph, design_space, velos):
        self.prim_graph = prim_graph
        self.causal_graph = causal_graph
        self.design_space = design_space
        self.velos = velos

    def __eq__(self, other):
        return (
            self.prim_graph.nodes == other.prim_graph.nodes and
            self.causal_graph.edges == other.causal_graph.edges and
            (self.design_space.xform_array == other.design_space.xform_array
             ).all()
        )

    def __hash__(self):
        h1 = str(sorted(self.prim_graph.nodes))
        h2 = str(sorted(sorted(e) for e in self.causal_graph.edges))
        h3 = self.design_space.xform_array.tobytes()
        h = (h1 + h2).encode('utf-8') + h3
        return int(hashlib.md5(h).hexdigest(), 16)

    def check_physically_valid_vector(self, vector):
        scene = Scene(geom=None, phys=True)
        xforms = self.design_space.vector2xforms(vector)
        scene.populate(self.prim_graph, xforms)
        return scene.check_physically_valid()

    def instantiate_from_vector(self, p, geom='LD', phys=True,
                                verbose_causal_graph=True):
        scene = Scene(geom, phys)
        xforms = self.design_space.vector2xforms(p)
        scene.populate(self.prim_graph, xforms, self.velos)
        emb_causal_graph = causal.embed_causal_graph(self.causal_graph, scene,
                                                     verbose_causal_graph)
        return ScenarioInstance(scene, emb_causal_graph)


class ScenarioInstance:
    def __init__(self, scene: Scene,
                 embedded_causal_graph: causal.CausalGraphTraverser):
        self.scene = scene
        self.embedded_causal_graph = embedded_causal_graph

    def simulate(self, duration, timestep, callbacks=None):
        if self.embedded_causal_graph is not None:
            callbacks = [] if callbacks is None else list(callbacks)
            callbacks.insert(0, self.embedded_causal_graph.update)
        simulate_scene(self.scene, duration, timestep, callbacks)
        if self.embedded_causal_graph is not None:
            return self.success

    @property
    def success(self):
        return self.embedded_causal_graph.success


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
            if any(abs(s - s_) >= 1e-5 for s, s_ in zip(state[1:], state_)):
                self.states[key].append(state)
                prev[key] = state[1:]

    def export(self, filename, **metadata):
        if filename[-4:] != ".pkl":
            filename += ".pkl"
        data = {'metadata': metadata, 'states': self.states}
        with open(filename, 'wb') as f:
            pickle.dump(data, f)


def import_scenario_data(path):
    if path.endswith("json"):
        with open(path, 'r') as f:
            scenario_data = json.load(f)
    elif path.endswith("py"):
        script = load_module("loaded_script", path)
        scenario_data = script.DATA
    else:
        print("Unrecognized extension")
        scenario_data = None
    return scenario_data


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_primitives(scene_data):
    """Load a primitive graph from a scene data dict."""
    prim_graph = nx.DiGraph()
    # First pass for primitives.
    for obj_data in scene_data:
        name = obj_data['name']
        PrimitiveType = getattr(primitives, obj_data['type'])
        # Separate geometry and physics arguments.
        geom_args = {}
        bullet_args = {}
        for k, v in obj_data['args'].items():
            if k.startswith("b_"):
                bullet_args[k[2:]] = v
            else:
                geom_args[k] = v
        # Create primitive.
        prim = PrimitiveType(
            name, **geom_args, **bullet_args
        )
        prim_graph.add_node(name, prim=prim)
        # For complex primitives, check if components are defined.
        try:
            components = obj_data['components']
        except KeyError:
            components = None
        if components:
            prim_graph.node[name]['components'] = components
        # Add optional tags.
        try:
            tags = obj_data['tags']
        except KeyError:
            tags = None
        if tags:
            prim_graph.node[name]['tags'] = tags
    # Second pass for scene hierarchy.
    for obj_data in scene_data:
        try:
            parent = obj_data['parent']
        except KeyError:
            parent = None
        if parent is not None:
            prim_graph.add_edge(parent, obj_data['name'])
    return prim_graph


def load_scenario(scenario_data):
    """Load an abstract scenario from a data dict."""
    prim_graph = load_primitives(scenario_data['scene'])
    design_space = load_design_space(scenario_data['scene'])
    try:
        graph_data = scenario_data['causal_graph']
    except KeyError:
        graph_data = None
    causal_graph = causal.load_causal_graph(graph_data)
    velos = load_velocities(scenario_data['scene'])
    scenario = Scenario(prim_graph, causal_graph, design_space, velos)
    return scenario


def load_scenario_instance(scenario_data, geom='LD', phys=True):
    """Load a scenario directly instantiated from a data dict."""
    scenario = load_scenario(scenario_data)
    xforms = load_xforms(scenario_data['scene'])
    return scenario.instantiante_from_xforms(xforms, geom, phys)


def load_scene(scene_data, geom='LD', phys=True):
    """Load a scene from a scene data dict."""
    prim_graph = load_primitives(scene_data)
    xforms = load_xforms(scene_data)
    velos = load_velocities(scene_data)
    scene = Scene(geom, phys)
    scene.populate(prim_graph, xforms, velos)
    return scene


def load_velocities(scene_data):
    """Load a velocities dict from a scene data dict."""
    velos = {}
    for obj_data in scene_data:
        name = obj_data['name']
        try:
            velo = obj_data['velo']['value']
        except KeyError:
            velo = None
        velos[name] = velo
    return velos


def load_xforms(scene_data):
    """Load a transforms dict from a scene data dict."""
    xforms = {}
    for obj_data in scene_data:
        name = obj_data['name']
        try:
            xform = obj_data['xform']['value']
        except KeyError:
            xform = None
        xforms[name] = xform
    return xforms


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
    maxiter = int(duration / timestep)
    do_break = False
    for i in range(maxiter + 1):
        time = i * timestep
        for c in callbacks:
            res = c(time)
            if res is not None and not res:
                do_break = True
        if do_break:
            break
        world.do_physics(timestep, 0)
    # Transforms are globally cached by default. Out of the regular
    # Panda3D task process, we need to empty this cache by hand when
    # running a large number of simulations, to avoid memory overflow.
    TransformState.garbage_collect()
    return time
