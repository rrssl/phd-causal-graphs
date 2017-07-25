# -*- coding: utf-8 -*-
"""
Various UI functionalities.

@author: Robin Roussel
"""
import datetime
import math
import numpy as np

from direct.interval.IntervalGlobal import Func, LerpFunc, Parallel, Sequence
from panda3d.core import CardMaker
from panda3d.core import LineSegs
from panda3d.core import Plane, Point3, Vec3


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
#        idealFarPlane = distance + radius * 1.5
#        lens.setFar(max(lens.getDefaultFar(), idealFarPlane))
#        idealNearPlane = distance - radius
#        lens.setNear(min(lens.getDefaultNear(), idealNearPlane))
#
        # Save original state
        self._unfocus_state = {
                'pos': self.pivot.get_pos(),
                'hpr': self.pivot.get_hpr(),
                'zoom': self.cam_distance
                }
        # Launch animation
        time = 1.
        # Note: using the Quat version ensures that the rotation takes the
        # shortest path. We cam still give it an HPR argument, which is
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
                toData=self.min_cam_distance,
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
    def __init__(self):
        self.plane = Plane(Vec3(0, 0, 1), Point3(0, 0, 0))
        self.tlims = 9

        cm = CardMaker("tile")
        cm.set_frame(-1, 1, -1, 1)
        cm.set_color(0, 1, 0, .4)
        self.tile = self.visual.attach_new_node(cm.generate())
        self.tile.look_at(0, 0, -1)
        self.tile.set_two_sided(True)
        self.tile.set_transparency(True)
#        filters = CommonFilters(self.win, self.cam)
#        filters.setBloom()
#        self.tile.setShaderAuto()
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

    def mouse_to_ground(self, mouse_pos, target_point):
        """Get the 3D point where a mouse ray hits the ground plane. If it does
        hit the ground, return True; False otherwise.

        Source: http://www.panda3d.org/forums/viewtopic.php?t=5409
        """
        near_point = Point3()
        far_point = Point3()
        self.camLens.extrude(mouse_pos, near_point, far_point)
        return self.plane.intersects_line(
                target_point,
                self.render.get_relative_point(self.camera, near_point),
                self.render.get_relative_point(self.camera, far_point)
                )

    def highlight_tile(self, task):
        if self.move_highlight and self.mouseWatcherNode.has_mouse():
            mpos = self.mouseWatcherNode.get_mouse()
            pos3d = Point3()
            if self.mouse_to_ground(mpos, pos3d):
                pos3d = np.asarray(pos3d).clip(-self.tlims, self.tlims).round()
                self.tile.set_pos(self.render, *pos3d)
        return task.cont


class Drawable:
    """Mixin giving to ShowBase the ability to sketch on the screen.

    """
    def __init__(self):
        # Not the most efficient structure but avoids a lot of low-level code.
        self.strokes = LineSegs("polyline")
        self.strokes.set_thickness(2)
        self.strokes.set_color((0, 0, 1, 1))
        self.sketch_np = None

    def set_draw(self, draw):
        if draw:
            if self.mouseWatcherNode.has_mouse():
                pos = self.mouseWatcherNode.get_mouse()
                self.strokes.move_to(pos[0], 0, pos[1])
            self.task_mgr.add(self.update_drawing, "update_drawing")
        else:
            self.task_mgr.remove("update_drawing")

    def update_drawing(self, task):
        if self.mouseWatcherNode.has_mouse():
            pos = self.mouseWatcherNode.get_mouse()
            self.strokes.draw_to(pos[0], 0, pos[1])

            # Update the drawing
            # Use a copy because create() calls reset().
            geom = LineSegs(self.strokes).create()
            try:
                # I don't know if this is more efficient than calling
                # geom.replace_node(self.sketch_np.node())
                self.sketch_np.remove_node()
            except AttributeError:
                pass
            self.sketch_np = self.render2d.attach_new_node(geom)

        return task.cont

    def clear_drawing(self):
        try:
            self.sketch_np.remove_node()
        except AttributeError:
            pass
        self.strokes.reset()

    def save_drawing(self, path="sketches/"):
        ls = LineSegs(self.strokes)
        ls.create()
        a = np.array(ls.get_vertices())
        filename = path + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        np.savetxt(filename, a)
