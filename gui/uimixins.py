# -*- coding: utf-8 -*-
"""
Various UI functionalities.

@author: Robin Roussel
"""
import datetime
import math

import numpy as np
from direct.interval.IntervalGlobal import Func, LerpFunc, Parallel, Sequence
from panda3d.core import (CardMaker, CollisionHandlerQueue, CollisionNode,
                          CollisionRay, CollisionTraverser, GeomNode, LineSegs,
                          Plane, Point2, Point3, Vec3, Vec4)


class Focusable:
    """Mixin to add smooth focus functionality to a Modeler.

    """
    def __init__(self):
        self._focus_anim = None
        self._focused = False

    def focus_view(self, nodepath):
        if (self._focused
            or (self._focus_anim is not None
                and self._focus_anim.is_playing())):
            return
        # Get new desired state
        bounds = nodepath.get_bounds()
        center = bounds.get_center()
        radius = bounds.get_radius()
        lens = self.camLens
        fov = min(lens.get_fov()) * math.pi / 180  # min between X and Z axes
        distance = radius / math.tan(fov * .5)
        #  idealFarPlane = distance + radius * 1.5
        #  lens.setFar(max(lens.getDefaultFar(), idealFarPlane))
        #  idealNearPlane = distance - radius
        #  lens.setNear(min(lens.getDefaultNear(), idealNearPlane))

        # Save original state
        self._unfocus_state = {
                'pos': self.pivot.get_pos(),
                'hpr': self.pivot.get_hpr(),
                'zoom': self.cam_distance
                }
        # Launch animation
        time = 1.
        # Note: using the Quat version ensures that the rotation takes the
        # shortest path. We can still give it an HPR argument, which is
        # (I think) easier to visualize than
        # "Quat(0, 0, 1/sqrt(2), 1/sqrt(2))".
        change_rigid = self.pivot.posQuatInterval(
                duration=time,
                pos=center + Vec3(0, 0, distance),
                hpr=Vec3(180, 90, 0),
                blendType='easeOut')
        change_zoom = LerpFunc(
                lambda x: setattr(self, "cam_distance", x),
                duration=time,
                fromData=self.cam_distance,
                toData=distance,
                blendType='easeOut')
        self._focus_anim = Sequence(
                Func(lambda: setattr(self, "move_highlight", False)),
                Parallel(change_rigid, change_zoom),
                Func(lambda: setattr(self, "_focused", True)))
        self._focus_anim.start()

    def unfocus_view(self):
        if (not self._focused
            or (self._focus_anim is not None
                and self._focus_anim.is_playing())):
            return
        # Launch animation
        time = 1.
        change_rigid = self.pivot.posQuatInterval(
                duration=time,
                pos=self._unfocus_state['pos'],
                hpr=self._unfocus_state['hpr'],
                blendType='easeOut')
        change_zoom = LerpFunc(
                lambda x: setattr(self, "cam_distance", x),
                duration=time,
                fromData=self.cam_distance,
                toData=self._unfocus_state['zoom'],
                blendType='easeOut')
        self._focus_anim = Sequence(
                Parallel(change_rigid, change_zoom),
                Func(lambda: setattr(self, "move_highlight", True)),
                Func(lambda: setattr(self, "_focused", False)))
        self._focus_anim.start()


class Tileable:
    """Mixin adding a visual tile selector to a Modeler.

    """
    def __init__(self, tile_size=1):
        self.plane = Plane(Vec3(0, 0, 1), Point3(0, 0, 0))
        self.tile_size = tile_size
        self.tlims = 9

        cm = CardMaker("tile")
        cm.set_frame(Vec4(-1, 1, -1, 1) * tile_size)
        cm.set_color(Vec4(1, 1, 1, .5))
        self.tile = self.visual.attach_new_node(cm.generate())
        self.tile.look_at(Point3(0, 0, -1))
        self.tile.set_two_sided(True)
        self.tile.set_transparency(True)
        #  filters = CommonFilters(self.win, self.cam)
        #  filters.setBloom()
        #  self.tile.setShaderAuto()
        self.tile.hide()

        self.move_highlight = False

    def set_show_tile(self, show):
        if show:
            self.tile.show()
            self.task_mgr.add(self.highlight_tile, "highlight_tile")
        else:
            self.tile.hide()
            self.task_mgr.remove("highlight_tile")
        self.move_highlight = show

    def mouse_to_ground(self, mouse_pos):
        """Get the 3D point where a mouse ray hits the ground plane. If it does
        hit the ground, a Point3; None otherwise.

        Parameters
        ----------
        mouse_pos : (2,) float sequence
          Cartesian coordinates of the mouse in screen space.

        """
        near_point = Point3()
        far_point = Point3()
        self.camLens.extrude(Point2(*mouse_pos), near_point, far_point)
        target_point = Point3()
        do_intersect = self.plane.intersects_line(
                target_point,
                self.render.get_relative_point(self.camera, near_point),
                self.render.get_relative_point(self.camera, far_point)
                )
        if do_intersect:
            return target_point
        else:
            return None

    def highlight_tile(self, task):
        if self.move_highlight and self.mouseWatcherNode.has_mouse():
            mpos = self.mouseWatcherNode.get_mouse()
            pos3d = self.mouse_to_ground(mpos)
            if pos3d is not None:
                pos3d = (np.asarray(pos3d) / self.tile_size).clip(
                        -self.tlims, self.tlims).round() * self.tile_size
                self.tile.set_pos(self.render, *pos3d)
        return task.cont


class Drawable:
    """Mixin giving to ShowBase the ability to sketch on the screen.

    """
    def __init__(self, color=(0, 0, 1, 1), thickness=2):
        self.strokes = []
        self.pencil = LineSegs("pencil")
        self.pencil.set_color(color)
        self.pencil.set_thickness(thickness)
        self.sketch_np = self.render2d.attach_new_node("sketches")

    def set_draw(self, draw):
        if draw:
            if self.mouseWatcherNode.has_mouse():
                pos = self.mouseWatcherNode.get_mouse()
                # /!\ get_mouse returns a shallow copy
                self.strokes.append([list(pos)])
            self.task_mgr.add(self._update_drawing, "update_drawing")
        else:
            self.task_mgr.remove("update_drawing")

    def _update_drawing(self, task):
        if self.mouseWatcherNode.has_mouse():
            # /!\ get_mouse returns a shallow copy
            pos = list(self.mouseWatcherNode.get_mouse())
            stroke = self.strokes[-1]
            # Filter duplicates
            if not (len(stroke) and np.allclose(pos, stroke[-1])):
                stroke.append(pos)
            # Update the drawing
            node = self._draw_stroke(stroke)
            if self.sketch_np.get_num_children() == len(self.strokes):
                node.replace_node(self.sketch_np.get_children()[-1].node())
            else:
                self.sketch_np.attach_new_node(node)
        return task.cont

    def _draw_stroke(self, stroke):
        """Generate the GeomNode for this stroke.

        Returns
        -------
        out : GeomNode
          The node generated by LineSegs.create().

        """
        pencil = self.pencil
        pencil.move_to(stroke[0][0], 0, stroke[0][1])
        for x, y in stroke[1:]:
            pencil.draw_to(x, 0, y)
        return pencil.create()

    def clear_drawing(self):
        self.sketch_np.node().remove_all_children()
        #  self.pencil.reset()
        self.strokes = []

    def save_drawing(self, path=""):
        a = np.array(self.strokes)
        filename = path + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        np.save(filename, a)


class Pickerable:
    """Mixin giving Modeler the ability to pick objects.

    You need to set the python tag 'pickable' to True if you want a model
    to be pickable.

    Attributes
    ----------
    pick_level : int
      Which ancestor to return when calling get_hit_object().
      0 is the object itself (default), 1 is the parent, etc.

    """
    def __init__(self):
        self.pick_traverser = CollisionTraverser()
        self.pick_queue = CollisionHandlerQueue()
        self.picker_ray = CollisionRay()
        picker_node = CollisionNode('mouse_ray')
        picker_node.set_from_collide_mask(GeomNode.get_default_collide_mask())
        picker_node.add_solid(self.picker_ray)
        picker_np = self.camera.attach_new_node(picker_node)
        self.pick_traverser.add_collider(picker_np, self.pick_queue)

        self.pick_level = 0

    def get_hit_object(self):
        if self.mouseWatcherNode.has_mouse():
            pos = self.mouseWatcherNode.get_mouse()
            self.picker_ray.set_from_lens(
                    self.camNode, pos.get_x(), pos.get_y())
            self.pick_traverser.traverse(self.models)
            pick_queue = self.pick_queue
            if pick_queue.get_num_entries() > 0:
                pick_queue.sort_entries()
                picked_obj = pick_queue.get_entry(0).get_into_node_path()
                picked_obj = picked_obj.find_net_python_tag('pickable')
                # False and None are equivqlent here
                if picked_obj.get_python_tag('pickable'):
                        return picked_obj.get_ancestor(self.pick_level)
        return None
