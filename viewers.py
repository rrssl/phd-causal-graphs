# -*- coding: utf-8 -*-
"""
Custom classes to improve on the basic Panda3D viewer.

@author: Robin Roussel
"""
from direct.showbase.ShowBase import ShowBase
from panda3d.core import AmbientLight, DirectionalLight
from panda3d.core import ShadeModelAttrib
from panda3d.core import Point2
from panda3d.core import LineSegs
from panda3d.core import NodePath
from panda3d.bullet import BulletWorld, BulletDebugNode

from coord_grid import ThreeAxisGrid

class TurntableViewer(ShowBase):
    """Provides a Blender-like 'turntable' viewer, more convenient than
    Panda3D's default trackball viewer.

    Features
    --------
    - Rotate around pivot (head and pan)
    - Move pivot
    - Zoom

    TODO:
        - add view reset

    Notes
    -----
    To change the initial camera view:

    >>> self.cam_distance = 10
    >>> self.pivot.set_h(self.pivot, -15)
    >>> self.pivot.set_p(self.pivot, 15)

    Source:
        'camera/free.py' in https://launchpad.net/panda3dcodecollection
    """

    def __init__(self):
        super().__init__()

        self.disable_mouse()

        # Camera movement
        self.mouse_pos = None
        self.start_camera_movement = False
        self.move_pivot = False
#        self.accept("mouse1", self.set_move_camera, [True])
#        self.accept("mouse1-up", self.set_move_camera, [False])
        self.accept("mouse2", self.set_move_camera, [True])
        self.accept("mouse2-up", self.set_move_camera, [False])
        self.accept("mouse3", self.set_move_pivot_and_camera, [True])
        self.accept("mouse3-up", self.set_move_pivot_and_camera, [False])
#        self.accept("shift", self.set_move_pivot, [True])
#        self.accept("shift-up", self.set_move_pivot, [False])
#        self.accept("shift-mouse2", self.set_move_camera, [True])
#        self.accept("shift-mouse2-up", self.set_move_camera, [False])
        # Zoom
        self.accept("wheel_up", self.zoom, [True])
        self.accept("wheel_down", self.zoom, [False])
        self.acceptOnce("+", self.zoom, [True])
        self.acceptOnce("-", self.zoom, [False])

        # Control parameters
        self.cam_distance = 30.
        self.max_cam_distance = 100.
        self.min_cam_distance = 2.  # Must be > 0
        self.zoom_speed = 3.
        self.mouse_speed = 100.

        # Pivot node
        self.pivot = self.render.attach_new_node("Pivot point")
        self.pivot.set_pos(0, 0, 0)
        self.camera.reparent_to(self.pivot)

        # TODO check why the priority value is -4
        self.task_mgr.add(self.update_cam, "update_cam", priority=-4)

    def set_move_camera(self, move_camera):
        if self.mouseWatcherNode.has_mouse():
            self.mouse_pos = self.mouseWatcherNode.get_mouse()
        self.start_camera_movement = move_camera

    def set_move_pivot(self, move_pivot):
        self.move_pivot = move_pivot

    def set_move_pivot_and_camera(self, move):
        self.set_move_pivot(move)
        self.set_move_camera(move)

    def zoom(self, zoom_in):
        if zoom_in:
            if self.cam_distance > self.min_cam_distance:
                # Prevent the distance from becoming negative
                self.cam_distance -= min(
                        self.zoom_speed,
                        self.cam_distance - self.min_cam_distance)
                # Reaccept the zoom in key
                self.acceptOnce("+", self.zoom, [True])
        else:
            if self.cam_distance < self.max_cam_distance:
                self.cam_distance += self.zoom_speed
                # Reaccept the zoom out key
                self.acceptOnce("-", self.zoom, [False])

    def update_cam(self, task):
        if self.mouseWatcherNode.has_mouse():
            x = self.mouseWatcherNode.get_mouse_x()
            y = self.mouseWatcherNode.get_mouse_y()

            # Move the camera if a mouse key is pressed and the mouse moved
            if self.mouse_pos is not None and self.start_camera_movement:
                move_move_x = (self.mouse_pos.get_x() - x) * (
                        self.mouse_speed + self.task_mgr.globalClock.get_dt())
                move_move_y = (self.mouse_pos.get_y() - y) * (
                        self.mouse_speed + self.task_mgr.globalClock.get_dt())
                self.mouse_pos = Point2(x, y)

                if not self.move_pivot:
                    # Rotate the pivot point
                    pre_p = self.pivot.get_p()
                    self.pivot.set_p(0)
                    self.pivot.set_h(self.pivot, move_move_x)
                    self.pivot.set_p(pre_p)
                    self.pivot.set_p(self.pivot, move_move_y)
                else:
                    # Move the pivot point
                    ratio = self.cam_distance / self.max_cam_distance
                    self.pivot.set_x(self.pivot, -move_move_x * ratio)
                    self.pivot.set_z(self.pivot,  move_move_y * ratio)

        # Set the camera zoom
        self.camera.set_y(self.cam_distance)
        # Always look at the pivot point
        self.camera.look_at(self.pivot)

        return task.cont


def create_axes():
    """Create the XYZ-axes indicator."""
    axes = LineSegs()
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

    return NodePath(axes.create())


class Modeler(TurntableViewer):
    """Provides the look and feel of a basic 3D modeler.

    Features
    --------
    - Flat shading
    - Slightly visible wireframe
    - Directional light towards the object
    - Axes and 'ground' indicator

    TODO:
        - add option to place the camera s.t. all objects are visible
        - (optional) add a shaded background
    """

    def __init__(self):
        super().__init__()

        self.models = self.render.attach_new_node("models")
        self.visual = self.render.attach_new_node("visual")
        # Shading
        self.models.set_attrib(ShadeModelAttrib.make(ShadeModelAttrib.M_flat))
        self.models.set_render_mode_filled_wireframe(.3)
        # Lights
        dlight = DirectionalLight("models_dlight")
        dlnp = self.camera.attach_new_node(dlight)
        dlnp.look_at(-self.cam.get_pos())
        self.models.set_light(dlnp)
        alight = AmbientLight("models_alight")
        alight.set_color(.1)
        self.models.set_light(self.render.attach_new_node(alight))
        alight = AmbientLight("visual_alight")
        alight.set_color(.8)
        self.visual.set_light(self.render.attach_new_node(alight))
        # Background
        self.set_background_color(.9, .9, .9)
        # Axes indicator (source: panda3dcodecollection, with modifications.)
        # Load the axes that should be displayed
        axes = create_axes()
        corner = self.aspect2d.attach_new_node("Axes indicator")
        corner.set_pos(self.a2dLeft+.15, 0, self.a2dBottom+.12)
        axes.reparent_to(corner)
        # Make sure it will be drawn above all other elements
        axes.set_depth_test(False)
        axes.set_bin("fixed", 0)
        # Now make sure it will stay in the correct rotation to render.
        self.axes = axes
        self.task_mgr.add(self.update_axes, "update_axes", priority=-4)
        # Ground plane
        grid_maker = ThreeAxisGrid(xsize=10, ysize=10, zsize=0)
        grid_maker.gridColor = grid_maker.subdivColor = .35
        grid_maker.create().reparent_to(self.visual)

    def update_axes(self, task):
        # Point of reference for each rotation is super important here.
        # We want the axes have the same orientation wrt the screen (render2d),
        # as the orientation of the scene (render) wrt the camera.
        self.axes.set_hpr(self.render2d, self.render.get_hpr(self.camera))
        return task.cont


class PhysicsViewer(Modeler):
    """Provides control and visualization for the physical simulation.

    Features
    --------
    - Play/pause/reset physics
    - Bullet debug mode

    TODO:
        - Add visual timeline

    """

    def __init__(self):
        super().__init__()

        self.world = BulletWorld()
        self.world.set_gravity((0, 0, -9.81))
#        self.world_time = 0.

        self.task_mgr.add(self.update_physics, "update_physics")
        self.accept('d', self.toggle_bullet_debug)
        self.accept('r', self.reset_physics)
        self.accept('space', self.toggle_physics)
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
        for path in self._physics_cache.keys():
            state = self._physics_cache[path]
            path.set_transform(state[0])
            body = path.node()
            body.clear_forces()
            body.set_linear_velocity(state[1])
            body.set_angular_velocity(state[2])
            body.set_active(True)
#        self.world_time = 0.

    def toggle_bullet_debug(self):
        try:
            if self._debug_np.is_hidden():
                self._debug_np.show()
            else:
                self._debug_np.hide()
        except AttributeError:
            dn = BulletDebugNode("debug")
            dn.show_wireframe(True)
#            dn.show_constraints(True)
            dn.show_bounding_boxes(True)
#            dn.show_normals(True)
            self._debug_np = self.render.attach_new_node(dn)
            self._debug_np.show()
            self.world.set_debug_node(dn)

    def toggle_physics(self):
        self.play_physics = not self.play_physics

    def update_physics(self, task):
        if self.play_physics:
            dt = self.task_mgr.globalClock.get_dt()
            self.world.do_physics(dt)
#            self.world_time += dt
        return task.cont


class FutureViewer(PhysicsViewer):
    """Provides a glimpse into the future motion of each dynamic object.

    Features
    --------
    - Show and update the motion path of each dynamic object

    """

    def __init__(self):
        super().__init__()

        self.future_vision_horizon = 20. #seconds
        self.future_vision_resol = 1 / 10. # hertz
        self._future_cache = {}
        self.future_vision = self.visual.attach_new_node("future")
        self.future_vision.hide()
        self.task_mgr.do_method_later(
                0, self.update_future, "init_future_cache", [], sort=1)

        self.accept('f', self.toggle_future)

    def redraw_future(self):
        path = self.future_vision.find("trajectories")
        if not path.is_empty(): path.remove_node()
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
#        print(self.future_vision.get_children())

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
