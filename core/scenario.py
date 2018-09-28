import pickle
import subprocess
from itertools import count
from math import ceil

import networkx as nx
from panda3d.core import GeomVertexReader, NodePath, TransformState
from shapely.geometry import LineString

import core.config as cfg
from core import causal_graph as causal
from core import primitives
from core.export import VectorFile


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
            self.graph_view = causal.CausalGraphViewer(self.causal_graph.root)

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

    def export_layout_to_pdf(self, filename, sheetsize, plane='xy',
                             exclude=None):
        if exclude is None:
            exclude = []
        geom_nodes = self.graph.find_all_matches("**/+GeomNode")
        objects = []
        min_u = min_v = float('inf')
        max_u = max_v = -min_u
        # First pass: retrieve the vertices.
        for node_path in geom_nodes:
            if node_path.name in exclude:
                continue
            geom_node = node_path.node()
            mat = node_path.get_net_transform().get_mat()
            objects.append([])
            for geom in geom_node.get_geoms():
                vertex = GeomVertexReader(geom.get_vertex_data(), 'vertex')
                while not vertex.is_at_end():
                    point = mat.xform_point(vertex.get_data3f())
                    u = getattr(point, plane[0]) * -100
                    v = getattr(point, plane[1]) * -100
                    objects[-1].append([u, v])
                    # Update the min and max.
                    if u < min_u:
                        min_u = u
                    if v < min_v:
                        min_v = v
                    if u > max_u:
                        max_u = u
                    if v > max_v:
                        max_v = v
        # Second pass: get the convex hulls.
        objects = [LineString(o).convex_hull.exterior.coords for o in objects]
        # Determine the size of the whole sheet: i.e., choose landscape or
        # portrait, and determine the number of sheets.
        su, sv = sheetsize
        nu_por = ceil((max_u - min_u)/su)
        nv_por = ceil((max_v - min_v)/sv)
        nu_lan = ceil((max_u - min_u)/sv)
        nv_lan = ceil((max_v - min_v)/su)
        if (nu_por + nv_por) <= (nu_lan + nv_lan):
            nu = nu_por
            nv = nv_por
        else:
            nu = nu_lan
            nv = nu_lan
            su, sv = sv, su
        sheetsize = (nu*su, nv*sv)
        # Create the vector file.
        vec = VectorFile(filename, sheetsize)
        # Add the cut lines.
        guides_color = "#DDDDDD"
        for i in range(1, nu):
            vec.add_polyline([[su*i, 0], [su*i, sheetsize[1]]],
                             linecolor=guides_color)
        for i in range(1, nv):
            vec.add_polyline([[0, sv*i], [sheetsize[0], sv*i]],
                             linecolor=guides_color)
        # Add the stitch guides.
        density = nu + nv
        points = []
        # (This could be rewritten with modulos, but would be harder to read.)
        for i in range(0, density):
            points.append([0, i*sheetsize[1]/density])
            points.append([i*sheetsize[0]/density, 0])
        for i in range(0, density):
            points.append([sheetsize[0], i*sheetsize[1]/density])
            points.append([i*sheetsize[0]/density, sheetsize[1]])
        for i in range(0, density):
            points.append([0, (density-i)*sheetsize[1]/density])
            points.append([i*sheetsize[0]/density, sheetsize[1]])
        for i in range(0, density):
            points.append([sheetsize[0], (density-i)*sheetsize[1]/density])
            points.append([i*sheetsize[0]/density, 0])
        vec.add_polyline(points, linecolor=guides_color)
        # Add the polylines.
        du = sheetsize[0]/2 - (min_u + max_u)/2
        dv = sheetsize[1]/2 - (min_v + max_v)/2
        for o in objects:
            o = [[u + du, v + dv] for u, v in o]
            vec.add_polyline(o)
        # Write the file.
        vec.save()

    def get_physical_validity_constraint(self):
        """Compute the sum of all physical constraint violations (<= 0)."""
        world = self.world
        constraint = 0
        # Get pulleys' constraint
        for pulley_cb in world._callbacks:
            constraint += min(0, pulley_cb.__self__._get_loose_rope_length())
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
        for a, b in zip(bodies[:-1], bodies[1:]):
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
                    constraint += penetration
        # Re-disable collisions for static objects
        for body in static:
            body.set_static(True)
            body.set_active(False)
        # Same issue as described in simulate_scene
        TransformState.garbage_collect()
        return constraint

    def populate(self, prim_graph, xforms):
        """Populate the scene with primitive instances.

        Parameters
        ----------
        prim_graph : networkx.DiGraph
          Tree of primitives reflecting the scene graph hierarchy (as given to
          Scenario).
        xforms : dict
          Dictionary of o_name: o_xform pairs, where o_xform is a 6-tuple
          corresponding to panda3d's (x,y,z,h,p,r) values.

        """
        graph = self.graph
        world = self.world
        name2nopa = {}
        # First pass: create and attach objects to the scene graph and world
        for name, prim in prim_graph.nodes(data='prim'):
            prim.geom = self.geom
            prim.phys = self.phys
            nopa = prim.create()
            nopa.set_pos_hpr(*xforms[name])
            name2nopa[name] = nopa
            prim.attach_to(graph, world)
        # Second pass: scene graph hierarchy.
        for parent, child in prim_graph.edges:
            name2nopa[child].reparent_to(name2nopa[parent])
            if self.phys:
                try:
                    name2nopa[child].node().set_transform_dirty()
                except AttributeError:
                    pass


class Scenario:
    """Abstract representation of a scenario.

    This is just a convenient, "stateless" high-level representation,
    not unlike a dict-like specification but with a number of preprocessing
    operations already done.
    Specifying a position in the design space (or equivalently, a transform for
    each object) via instantiante_from_xforms() yields a ScenarioInstance that
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
    def __init__(self, prim_graph, causal_graph, design_space):
        self.prim_graph = prim_graph
        self.causal_graph = causal_graph
        self.design_space = design_space

    def check_physically_valid_sample(self, sample):
        scene = Scene(geom=None, phys=True)
        xforms = self.design_space.sample2xforms(sample)
        scene.populate(self.prim_graph, xforms)
        return self.check_physically_valid()

    def instantiate_from_sample(self, sample, geom='LD', phys=True):
        xforms = self.design_space.sample2xforms(sample)
        return self.instantiante_from_xforms(xforms, geom, phys)

    def instantiante_from_xforms(self, xforms, geom='LD', phys=True):
        scene = Scene(geom, phys)
        scene.populate(self.prim_graph, xforms)
        emb_causal_graph = causal.embed_causal_graph(self.causal_graph, scene)
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
    scenario = Scenario(prim_graph, causal_graph, design_space)
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
    scene = Scene(geom, phys)
    scene.populate(prim_graph, xforms)
    return scene


def load_xforms(scene_data):
    """Load a transforms dict from a scene data dict."""
    xforms = {}
    for obj_data in scene_data:
        name = obj_data['name']
        try:
            xform = obj_data['xform']['value']
        except KeyError:
            xform = [0, 0, 0, 0, 0, 0]
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
