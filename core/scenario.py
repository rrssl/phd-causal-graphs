import pickle
import subprocess
from itertools import count
from math import ceil

from panda3d.core import GeomVertexReader, NodePath, TransformState
from shapely.geometry import LineString

import core.config as cfg
from core import causal_graph as causal
from core import events
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
    def __init__(self, graph: NodePath, world: primitives.World):
        self.graph = graph
        self.world = world

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
            if any(abs(s - s_) >= 1e-4 for s, s_ in zip(state[1:], state_)):
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
        if phys:
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


def load_causal_graph(graph_data, scene, verbose=False):
    graph = scene.graph
    world = scene.world
    name2path = {}
    name2event = {}
    event2children = {}
    root = None
    # First pass: create the events
    for event_data in graph_data:
        name = event_data['name']
        EventType = getattr(events, event_data['type'])
        args = event_data['args'].copy()
        for key, value in args.items():
            if type(value) is str:
                try:
                    path = name2path[value]
                except KeyError:
                    path = graph.find("**/" + value + "_solid")
                    if path.is_empty():
                        path = None
                    name2path[value] = path
                if path is not None:
                    args[key] = path
        if EventType in (events.Contact, events.NoContact, events.RollingOn):
            args['world'] = world
        event = causal.Event(name, EventType(**args))
        name2event[name] = event
        try:
            children = event_data['children']
        except KeyError:
            children = []
        event2children[event] = children
    # Second pass: connect the events
    has_parent = set()
    for event, children in event2children.items():
        for child in children:
            causal.connect(event, name2event[child])
            has_parent.add(name2event[child])
    root = (event2children.keys() - has_parent).pop()
    graph = causal.CausalGraphTraverser(
        root=root, verbose=verbose
    )
    return graph


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
