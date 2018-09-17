"""
Custom classes to improve on the basic Panda3D viewer.

"""
import math
import pickle

from direct.showbase.ShowBase import ShowBase
from panda3d.bullet import BulletDebugNode
from panda3d.core import (AmbientLight, DirectionalLight, LineSegs, NodePath,
                          Point2, Point3, Quat, ShadeModelAttrib, Vec3, Vec4)

import gui.config as cfg
from gui.coord_grid import ThreeAxisGrid
from gui.uiwidgets import PlayerControls
from core.primitives import World


class TurntableViewer(ShowBase):
    """Provides a Blender-like 'turntable' viewer, more convenient than
    Panda3D's default trackball viewer.

    Parameters
    ----------
    view_h : float, optional
      Initial heading angle from which the scene is viewed. Defaults to 0.
    view_p : float, optional
      Initial pitch angle from which the scene is viewed. Defaults to 0.

    Features
    --------
    - Rotate around pivot (head and pan)
    - Move pivot
    - Zoom
    - Center view on node

    Notes
    -----
    To change the initial camera view:

    >>> self.cam_distance = 10
    >>> self.pivot.set_h(self.pivot, -15)
    >>> self.pivot.set_p(self.pivot, 15)

    Source:
        'camera/free.py' in https://launchpad.net/panda3dcodecollection

    """

    def __init__(self, view_h=180, view_p=0):
        super().__init__()

        self.disable_mouse()
        self.disable_all_audio()
        self.task_mgr.remove("audioLoop")

        # Camera movement
        self.mouse_pos = None
        self.reset_default_mouse_controls()
        self.accept("1", self.view_front)
        self.accept("3", self.view_side)
        self.accept("7", self.view_top)

        # Zoom
        self.accept("wheel_up", self.zoom, [True])
        self.accept("wheel_down", self.zoom, [False])
        self.accept_once("+", self.zoom, [True, True])
        self.accept_once("-", self.zoom, [False, True])
        self.accept_once("home", self.center_view_on, [self.render])

        # Control parameters
        self.cam_distance = cfg.INIT_CAM_DISTANCE
        self.max_cam_distance = cfg.MAX_CAM_DISTANCE
        self.min_cam_distance = cfg.MIN_CAM_DISTANCE  # Must be > 0
        self.zoom_factor = cfg.ZOOM_FACTOR
        self.mouse_speed = cfg.MOUSE_SPEED

        # Pivot node
        self.pivot = self.render.attach_new_node("pivot_point")
        self.pivot.set_pos(0, 0, 0)
        self.pivot.set_h(self.pivot, view_h)
        self.pivot.set_p(self.pivot, view_p)
        self.camera.reparent_to(self.pivot)
        self.camera.set_y(self.cam_distance)
        self.update_lens_near_plane()

        # Framerate
        self.video_frame_rate = cfg.VIDEO_FRAME_RATE
        clock = self.task_mgr.globalClock
        clock.set_mode(clock.M_limited)
        clock.set_frame_rate(self.video_frame_rate)
        self.set_frame_rate_meter(True)  # show framerate

    def center_view_on(self, nodepath):
        bounds = nodepath.get_bounds()
        if bounds.is_empty():
            return
        center = bounds.get_center()
        radius = bounds.get_radius()
        fov = math.radians(
            min(self.camLens.get_fov()))  # min between X and Z axes
        distance = radius / math.tan(fov / 2)
        self.pivot.set_pos(center)
        self.cam_distance = distance
        self.update_lens_near_plane()
        self.accept_once("home", self.center_view_on, [nodepath])

    def reset_default_mouse_controls(self):
        self.accept("mouse3", self.set_move_camera, [True, 'rotate'])
        self.accept(
            "shift-mouse3", self.set_move_camera, [True, 'translate']
        )
        self.accept("mouse3-up", self.set_move_camera, [False])
        #  self.accept("mouse2", self.set_move_camera, [True])
        #  self.accept("mouse2-up", self.set_move_camera, [False])

    def rotate_view_smooth(self, hpr):
        quat = Quat()
        quat.set_hpr(hpr)
        length = (self.pivot.get_quat() - quat).length()
        self.pivot.quatInterval(
            duration=cfg.VIEW_CHANGE_SPEED*length,
            quat=quat
        ).start()

    def set_move_camera(self, move, mode=None):
        if move:
            self.mouse_pos = self.mouseWatcherNode.get_mouse()
            self.task_mgr.add(self.update_cam, "update_cam", extraArgs=[mode],
                              appendTask=True)
        else:
            self.task_mgr.remove("update_cam")

    def shutdown(self):
        self.task_mgr.remove("update_cam")
        super().shutdown()

    def update_cam(self, mode, task):
        mwn = self.mouseWatcherNode
        if mwn.has_mouse():
            x = mwn.get_mouse_x()
            y = mwn.get_mouse_y()
            dt = self.task_mgr.globalClock.get_dt()

            move_x = (self.mouse_pos.get_x() - x) * (self.mouse_speed + dt)
            move_y = (self.mouse_pos.get_y() - y) * (self.mouse_speed + dt)
            self.mouse_pos = Point2(x, y)  # deep copy is needed

            if mode == 'translate':
                # Move the pivot point
                ratio = self.cam_distance / self.max_cam_distance
                self.pivot.set_x(self.pivot, -move_x * ratio)
                self.pivot.set_z(self.pivot,  move_y * ratio)
            elif mode == 'rotate':
                # Rotate the pivot point
                pre_p = self.pivot.get_p()
                self.pivot.set_p(0)
                self.pivot.set_h(self.pivot, move_x)
                self.pivot.set_p(pre_p)
                self.pivot.set_p(self.pivot, move_y)
        # Always look at the pivot point
        self.camera.look_at(self.pivot)

        return task.cont

    def update_lens_near_plane(self):
        self.camLens.set_near(self.cam_distance * cfg.CAM_LENS_NEAR_FACTOR)

    def view_front(self):
        self.rotate_view_smooth(Vec3(180, 0, 0))

    def view_side(self):
        self.rotate_view_smooth(Vec3(-90, 0, 0))

    def view_top(self):
        self.rotate_view_smooth(Vec3(180, 90, 0))

    def zoom(self, zoom_in, from_key=False):
        if zoom_in:
            if self.cam_distance > self.min_cam_distance:
                self.cam_distance *= 1 - self.zoom_factor
                if from_key:
                    self.accept_once("+", self.zoom, [True, True])
        else:
            if self.cam_distance < self.max_cam_distance:
                self.cam_distance *= 1 + self.zoom_factor
                if from_key:
                    self.accept_once("-", self.zoom, [False, True])
        self.camera.set_y(self.cam_distance)
        self.update_lens_near_plane()


def create_axes(name):
    """Create the XYZ-axes indicator."""
    axes = LineSegs(name)
    axes.set_thickness(2)
    axes_size = .1

    axes.set_color((1, 0, 0, 1))
    axes.move_to(axes_size, 0, 0)
    axes.draw_to(0, 0, 0)

    axes.set_color((0, 1, 0, 1))
    axes.move_to(0, axes_size, 0)
    axes.draw_to(0, 0, 0)

    axes.set_color((0, 0, 1, 1))
    axes.move_to(0, 0, axes_size)
    axes.draw_to(0, 0, 0)

    return axes.create()


class Modeler(TurntableViewer):
    """Provides the look and feel of a basic 3D modeler.

    Parameters
    ----------
    grid : str or None, optional
        If not None, grid axes are specified by 'x', 'y' and 'z' (no matter
        the order). Defaults to 'xy'.

    Features
    --------
    - Flat shading
    - Slightly visible wireframe
    - Directional light towards the object
    - Axes and 'ground' indicator

    """

    def __init__(self, grid='xy', **viewer_kwargs):
        super().__init__(**viewer_kwargs)

        self.models = self.render.attach_new_node("models")
        self.visual = self.render.attach_new_node("visual")
        # Shading
        self.models.set_attrib(ShadeModelAttrib.make(ShadeModelAttrib.M_flat))
        self.models.set_render_mode_filled_wireframe(
                cfg.MODELS_WIREFRAME_COLOR)
        # Lights
        dlight = DirectionalLight("models_dlight")
        dlnp = self.camera.attach_new_node(dlight)
        dlnp.look_at(-self.cam.get_pos())
        self.models.set_light(dlnp)
        alight = AmbientLight("models_alight")
        alight.set_color(cfg.MODELS_AMBIENT_LIGHT_COLOR)
        self.models.set_light(self.render.attach_new_node(alight))
        alight = AmbientLight("visual_alight")
        alight.set_color(cfg.VISUAL_AMBIENT_LIGHT_COLOR)
        self.visual.set_light(self.render.attach_new_node(alight))
        # Background
        self.set_background_color(cfg.BACKGROUND_COLOR)
        # Axes indicator (source: panda3dcodecollection, with modifications.)
        # Load the axes that should be displayed
        axes = self.aspect2d.attach_new_node(create_axes("axes_indicator"))
        axes.set_pos(self.a2dLeft+.15, 0, self.a2dBottom+.12)
        axes.set_depth_test(True)   # make sure axes are drawn
        axes.set_depth_write(True)  # in the right order
        self.task_mgr.add(self.update_axes, "update_axes", extraArgs=[axes],
                          appendTask=True)
        # Ground plane
        if grid is not None:
            grid_maker = ThreeAxisGrid(
                xsize=('x' in grid), ysize=('y' in grid), zsize=('z' in grid),
                gridstep=1
            )
            grid_maker.gridColor = grid_maker.subdivColor = cfg.GRID_COLOR
            grid_maker.create().reparent_to(self.visual)
        # Save scene
        self.accept('s', self.models.write_bam_file, ["scene.bam"])
        # Center view on the entire scene
        self.accept_once("home", self.center_view_on, [self.models])
        # Show models
        self.accept('l', self.models.ls)

    def update_axes(self, axes, task):
        # Point of reference for each rotation is super important here.
        # We want the axes have the same orientation wrt the screen (render2d),
        # as the orientation of the scene (render) wrt the camera.
        axes.set_hpr(self.render2d, self.render.get_hpr(self.camera))
        return task.cont

    def shutdown(self):
        self.task_mgr.remove("update_axes")
        super().shutdown()


class PhysicsViewer(Modeler):
    """Provides control and visualization for the physical simulation.

    Features
    --------
    - Play/pause/reset physics
    - Bullet debug mode

    TODO:
        - Add visual timeline

    """

    def __init__(self, frame_rate=cfg.PHYSICS_FRAME_RATE, world=None,
                 **viewer_kwargs):
        super().__init__(**viewer_kwargs)
        self.physics_frame_rate = frame_rate

        if world is None:
            self.world = World()
            self.world.set_gravity(cfg.GRAVITY)
        else:
            self.world = world
        self.world_time = 0.

        self.task_mgr.add(self.update_physics, "update_physics")
        self.accept('d', self.toggle_bullet_debug)
        self.accept('r', self.reset_physics)
        self.accept('space', self.toggle_physics)
        self.accept('n', self.do_physics, [1/60])
        self.play_physics = False
        # Initialize cache after __init__ is done.
        self._physics_cache = {}
        self.task_mgr.do_method_later(
                0, self._create_cache, "init_physics_cache", [], sort=0)

    def _add_to_cache(self, path):
        """Cache the state of an object.

        State is defined as a triplet:
            - transform,
            - linear velocity,
            - angular velocity.

        Parameters
        ----------
        path : NodePath
            Path to a BulletBodyNode.

        """
        self._physics_cache[path] = (path.get_transform(),
                                     path.node().get_linear_velocity(),
                                     path.node().get_angular_velocity())

    def _create_cache(self):
        """Cache the state of each dynamic object in the scene."""
        for path in self.get_dynamic():
            self._add_to_cache(path)

    def get_dynamic(self):
        """Return a list of paths to the dynamic objects in the world."""
        return [NodePath.any_path(body)
                for body in self.world.get_rigid_bodies()
                if not (body.is_static() or body.is_kinematic())]

    def reset_physics(self):
        """Reset the position/velocities/forces of each dynamic object."""
        for man in self.world.get_manifolds():
            man.clear_manifold()
        for path in self._physics_cache.keys():
            state = self._physics_cache[path]
            path.set_transform(state[0])
            body = path.node()
            body.clear_forces()
            body.set_linear_velocity(state[1])
            body.set_angular_velocity(state[2])
            body.set_active(True)
            self.world_time = 0.

    def shutdown(self):
        self.task_mgr.remove("init_physics_cache")
        self.task_mgr.remove("update_physics")
        super().shutdown()

    def toggle_bullet_debug(self):
        try:
            if self._debug_np.is_hidden():
                self._debug_np.show()
                self.models.hide()
                self.set_background_color(cfg.DEBUG_BACKGROUND_COLOR)
            else:
                self._debug_np.hide()
                self.models.show()
                self.set_background_color(cfg.BACKGROUND_COLOR)
        except AttributeError:
            dn = BulletDebugNode("debug")
            dn.show_wireframe(True)
            dn.show_constraints(True)
            dn.show_bounding_boxes(True)
            #  dn.show_normals(True)
            self._debug_np = self.render.attach_new_node(dn)
            self._debug_np.show()
            self.models.hide()
            self.set_background_color(cfg.DEBUG_BACKGROUND_COLOR)
            self.world.set_debug_node(dn)
            self.do_physics(0)  # To force the update of Bullet

    def toggle_physics(self):
        self.play_physics = not self.play_physics

    def do_physics(self, dt):
        # Results for small objects are much more stable with a smaller
        # physics timestep. Typically, for a 1cm-cube you want 300Hz.
        # Rule: timeStep < maxSubSteps * fixedTimeStep
        # If you run interactively at 60Hz, with a simulator frequency of
        # 240Hz, you want maxSubSteps = 240/60+1.
        fv = self.video_frame_rate
        fp = self.physics_frame_rate
        self.world.do_physics(dt, int(fp/fv)+1, 1/fp)
        self.world_time += dt

    def update_physics(self, task):
        if self.play_physics:
            dt = self.task_mgr.globalClock.get_dt()
            self.do_physics(dt)
        return task.cont


class FutureViewer(PhysicsViewer):
    """Provides a glimpse into the future motion of each dynamic object.

    Features
    --------
    - Show and update the motion path of each dynamic object

    """

    def __init__(self, frame_rate=cfg.PHYSICS_FRAME_RATE, world=None):
        super().__init__(frame_rate=frame_rate, world=world)

        self.future_vision_horizon = 20.  # seconds
        self.future_vision_resol = 1 / 10.  # hertz
        self._future_cache = {}
        self.future_vision = self.visual.attach_new_node("future")
        self.future_vision.hide()
        self.task_mgr.do_method_later(
                0, self.update_future, "init_future_cache", [], sort=1)

        self.accept('f', self.toggle_future)

    def redraw_future(self):
        path = self.future_vision.find("trajectories")
        if not path.is_empty():
            path.remove_node()
        # Less subtle method:
        # self.future_vision.node().removeAllChildren()

        polyline = LineSegs("trajectories")
        polyline.set_thickness(2)
        polyline.set_color((1, 0, 1, 1))
        for trajectory in self._future_cache.values():
            polyline.move_to(trajectory[0])
            for pos in trajectory[1:]:
                polyline.draw_to(pos)
        self.future_vision.attach_new_node(polyline.create())
        # print(self.future_vision.get_children())

    def toggle_future(self):
        if self.future_vision.is_hidden():
            self.future_vision.show()
        else:
            self.future_vision.hide()

    def update_future(self):
        for path in self._physics_cache.keys():
            self._future_cache[path] = []

        self.reset_physics()
        time = 0.
        nb_bullet_substeps = max(int(self.future_vision_resol * 60) + 1, 1)
        while time <= self.future_vision_horizon:
            for path, trajectory in self._future_cache.items():
                trajectory.append(path.get_pos())
            self.world.do_physics(self.future_vision_resol, nb_bullet_substeps)
            time += self.future_vision_resol

        self.reset_physics()
        self.redraw_future()


class ScenarioViewer(PhysicsViewer):
    """Physics viewer with additional scenario semantics.

    Colors the objects according to the scenario validity and status.

    Parameters
    ----------
    scenario : Scenario
      Instance of a class from scenario.py.

    """
    def __init__(self, scenario, **viewer_kwargs):
        super().__init__(world=scenario.world, **viewer_kwargs)
        self.scenario = scenario
        scenario.scene.reparent_to(self.models)
        if not scenario.check_physically_valid():
            scenario.scene.set_render_mode_filled_wireframe(
                Vec4(*cfg.SCENARIO_INVALID_COLOR))
        self.status = None
        self.task_mgr.add(self.update_status, "update_status")
        self.accept('r', self.reset_scenario)
        if hasattr(scenario, 'graph_view'):
            self.accept('g', scenario.graph_view.render)

    def update_status(self, task):
        scenario = self.scenario
        scenario.terminate.update_and_check(self.world_time)
        if scenario.terminate.status != self.status:
            self.status = scenario.terminate.status
            if self.status == 'success':
                scenario.scene.set_color(Vec4(*cfg.SCENARIO_SUCCESS_COLOR))
            elif self.status == 'failure':
                scenario.scene.set_color(Vec4(*cfg.SCENARIO_TIMEOUT_COLOR))
            else:
                scenario.scene.clear_color()
        return task.cont

    def reset_scenario(self):
        self.scenario.terminate.reset()
        self.reset_physics()

    def shutdown(self):
        self.task_mgr.remove("update_status")
        super().shutdown()


class Replayer(Modeler):
    """Replay a sequence captured with a StateObserver.

    The original (simulator) frame rate is automatically remapped to the
    current video frame rate.

    Parameters
    ----------
    scene : string
      Filename of the .bam or .egg scene (.bam keeps more data).
    simu_data : string
      Filename of the .pkl file of simulation data.

    """
    def __init__(self, scene, simu_data, **viewer_kwargs):
        super().__init__(**viewer_kwargs)
        # Load the scene.
        scene = self.loader.load_model(scene)
        scene.reparent_to(self.models)
        # Load the frames.
        with open(simu_data, 'rb') as f:
            simu_data = pickle.load(f)
        simu_frame_rate = simu_data['metadata']['fps']
        length = max(states[-1][0] for states in simu_data['states'].values())
        n_frames = int(length * simu_frame_rate) + 1
        frames = [[] for _ in range(n_frames)]
        name2path = {}
        for name, states in simu_data['states'].items():
            try:
                nopa = name2path[name]
            except KeyError:
                nopa = scene.find("**/{}".format(name))
                name2path[name] = nopa
            for state in states:
                t = state[0]
                fi = int(t * simu_frame_rate)
                frames[fi].append((nopa, state[1:]))
        self.frames = frames
        self.remapping_factor = simu_frame_rate / self.video_frame_rate
        self.frame_start = 0
        self.frame_end = int((n_frames - 1) / self.remapping_factor)

        self.play = False
        self.current_frame = 0
        self.task_mgr.add(self.update_frame, "update_frame")
        self.accept('r', self.update_control, ["startButton"])
        self.accept('space', self.update_control, ["ppButton"])
        self.accept('n', self.update_control, ["nextButton"])
        self.accept('p', self.update_control, ["prevButton"])

        self.controls = PlayerControls(
            parent=self.a2dBottomCenter,
            frameSize=(-.5, .5, -.1, .1),
            pos=Point3(0, 0, .25*9/16),
            command=self.update_control,
            currentFrame=self.current_frame+1,
            numFrames=self.frame_end+1,
        )

    def go_to_frame(self, fi):
        # Clip fi
        fs = self.frame_start
        fe = self.frame_end
        fi = fs if fi < fs else fe if fi > fe else fi
        self.current_frame = fi
        # Update transforms
        fi_original = self.remap_frame(fi)
        for nopa, state in self.frames[fi_original]:
            if nopa.has_tag('save_scale'):
                nopa.set_scale(Vec3(*state[-3:]))
            nopa.set_pos(Point3(*state[:3]))
            nopa.set_quat(Quat(*state[3:7]))

    def go_to_next_frame(self):
        self.go_to_frame(self.current_frame + 1)

    def go_to_previous_frame(self):
        self.go_to_frame(self.current_frame - 1)

    def import_frames(self, filename):
        with open(filename, 'rb') as f:
            frames = pickle.load(f)
        nodepaths = self.scene.find_all_matches("**/=anim_id")
        nodepaths_and_frames = []
        for nopa in nodepaths:
            nodepaths_and_frames.append(
                (nopa, frames[nopa.get_tag('anim_id')])
            )
        return nodepaths_and_frames

    def remap_frame(self, fi):
        return int(fi * self.remapping_factor)

    def reset_frame(self):
        self.go_to_frame(0)

    def toggle_play(self):
        self.play = not self.play

    def update_frame(self, task):
        if self.play:
            fi = (self.current_frame + 1) % (self.frame_end + 1)
            self.go_to_frame(fi)
            self.controls.updateCurrentFrame(self.current_frame + 1)
        return task.cont

    def update_control(self, *args):
        name = args[-1]
        control = self.controls.component(name)
        if name == "timelineSlider":
            self.go_to_frame(round(control['value']) - 1)
        if name == "startButton":
            self.reset_frame()
        if name == "prevButton":
            self.go_to_previous_frame()
        if name == "ppButton":
            self.toggle_play()
            self.controls.togglePlayPause(self.play)
        if name == "nextButton":
            self.go_to_next_frame()
        if name == "endButton":
            self.go_to_frame(self.frame_end)
        self.controls.updateCurrentFrame(self.current_frame + 1)

    def shutdown(self):
        self.task_mgr.remove("update_frame")
        super().shutdown()
