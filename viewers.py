#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom classes to improve on the basic Panda3D viewer.

@author: Robin Roussel
"""
import sys
from direct.showbase.ShowBase import ShowBase
from panda3d.core import AmbientLight, DirectionalLight
from panda3d.core import ShadeModelAttrib
from panda3d.core import Point2
from panda3d.core import LineSegs
from panda3d.core import NodePath

from coord_grid import ThreeAxisGrid

class TurntableViewer(ShowBase):
    """Provides a Blender-like 'turntable' viewer, more convenient than
    Panda3D's default trackball viewer.

    Source: 'camera/free.py' in https://launchpad.net/panda3dcodecollection
    """

    def __init__(self):
        super().__init__()

        self.disableMouse()

        # Camera movement
        self.mousePos = None
        self.startCameraMovement = False
        self.movePivot = False
        self.accept("mouse1", self.setMoveCamera, [True])
        self.accept("mouse1-up", self.setMoveCamera, [False])
        self.accept("mouse2", self.setMoveCamera, [True])
        self.accept("mouse2-up", self.setMoveCamera, [False])
        self.accept("mouse3", self.setAndMovePivot, [True])
        self.accept("mouse3-up", self.setAndMovePivot, [False])
        self.accept("shift", self.setMovePivot, [True])
        self.accept("shift-up", self.setMovePivot, [False])
        self.accept("shift-mouse2", self.setMoveCamera, [True])
        self.accept("shift-mouse2-up", self.setMoveCamera, [False])
        # Zoom
        self.accept("wheel_up", self.zoom, [True])
        self.accept("wheel_down", self.zoom, [False])
        self.acceptOnce("+", self.zoom, [True])
        self.acceptOnce("-", self.zoom, [False])

        # Control parameters
        # TODO set them from input parameters rather than hard-coded values.
        self.camDistance = 30.
        self.maxCamDistance = 100.
        self.minCamDistance = 2.  # Must be > 0
        self.zoomSpeed = 3.
        self.mouseSpeed = 100.

        # Pivot node
        self.pivot = self.render.attachNewNode("Pivot point")
        self.pivot.setPos(0, 0, 0)
        self.camera.reparentTo(self.pivot)

        # TODO check why the priority value is -4
        self.taskMgr.add(self.updateCam, "task_camActualisation", priority=-4)

    def setMoveCamera(self, moveCamera):
        if self.mouseWatcherNode.hasMouse():
            self.mousePos = self.mouseWatcherNode.getMouse()
        self.startCameraMovement = moveCamera

    def setMovePivot(self, movePivot):
        self.movePivot = movePivot

    def setAndMovePivot(self, move):
        self.setMovePivot(move)
        self.setMoveCamera(move)

    def zoom(self, zoomIn):
        if zoomIn:
            if self.camDistance > self.minCamDistance:
                # Prevent the distance from becoming negative
                self.camDistance -= min(self.zoomSpeed,
                                        self.camDistance - self.minCamDistance)
                # Reaccept the zoom in key
                self.acceptOnce("+", self.zoom, [True])
        else:
            if self.camDistance < self.maxCamDistance:
                self.camDistance += self.zoomSpeed
                # Reaccept the zoom out key
                self.acceptOnce("-", self.zoom, [False])

    def updateCam(self, task):
        if self.mouseWatcherNode.hasMouse():
            x = self.mouseWatcherNode.getMouseX()
            y = self.mouseWatcherNode.getMouseY()

            # Move the camera if a mouse key is pressed and the mouse moved
            if self.mousePos is not None and self.startCameraMovement:
                mouseMoveX = (self.mousePos.getX() - x) * (
                        self.mouseSpeed + self.taskMgr.globalClock.getDt())
                mouseMoveY = (self.mousePos.getY() - y) * (
                        self.mouseSpeed + self.taskMgr.globalClock.getDt())
                self.mousePos = Point2(x, y)

                if not self.movePivot:
                    # Rotate the pivot point
                    preP = self.pivot.getP()
                    self.pivot.setP(0)
                    self.pivot.setH(self.pivot, mouseMoveX)
                    self.pivot.setP(preP)
                    self.pivot.setP(self.pivot, mouseMoveY)
                else:
                    # Move the pivot point
                    self.pivot.setX(self.pivot, -mouseMoveX)
                    self.pivot.setZ(self.pivot,  mouseMoveY)

        # Set the camera zoom
        self.camera.setY(self.camDistance)
        # Always look at the pivot point
        self.camera.lookAt(self.pivot)

        return task.cont


def createAxes():
    axes = LineSegs()
    axes.setThickness(2)
    axesSize = .1

    axes.setColor((1, 0, 0, 1))
    axes.moveTo(axesSize, 0, 0)
    axes.drawTo(0, 0, 0)

    axes.setColor((0, 1, 0, 1))
    axes.moveTo(0, axesSize, 0)
    axes.drawTo(0, 0, 0)

    axes.setColor((0, 0, 1, 1))
    axes.moveTo(0, 0, axesSize)
    axes.drawTo(0, 0, 0)

    return NodePath(axes.create())


class Modeler(TurntableViewer):
    """Provides the look and feel of a basic 3D modeler.

    - Flat shading
    - Slightly visible wireframe
    - Directional light towards the object
    - Axes and 'ground' indicator

    TODO:
        - add option to reset the view
        - add option to place the camera s.t. all objects are visible
        - (optional) add a shaded background
    """

    def __init__(self):
        super().__init__()

        self.models = self.render.attachNewNode("Models")
        # Shading
        self.models.setAttrib(ShadeModelAttrib.make(ShadeModelAttrib.MFlat))
        self.models.setRenderModeFilledWireframe(.3)
        # Lights
        dlight = DirectionalLight('dlight')
        dlnp = self.camera.attachNewNode(dlight)
        dlnp.lookAt(-self.cam.getPos())
        self.render.setLight(dlnp)
        alight = AmbientLight("alight")
        alight.setColor(.1)
        self.render.setLight(self.render.attachNewNode(alight))
        # Background
        self.setBackgroundColor(.9, .9, .9)
        # Axes indicator (source: panda3dcodecollection, with modifications.)
        # Load the axes that should be displayed
        axes = createAxes()
        corner = self.aspect2d.attachNewNode("Axes indicator")
        corner.setPos(self.a2dLeft+.15, 0, self.a2dBottom+.1)
        axes.reparentTo(corner)
        # Make sure it will be drawn above all other elements
        axes.setDepthTest(False)
        axes.setBin("fixed", 0)
        # Now make sure it will stay in the correct rotation to render.
        self.axes = axes
        self.taskMgr.add(self.updateAxes, "update_axes", priority=-4)
        # Ground plane
        gridMaker = ThreeAxisGrid(xsize=10, ysize=10, zsize=0)
        gridMaker.gridColor = gridMaker.subdivColor = .35
        gridMaker.create().reparentTo(self.render)

        # Controls
        self.accept("escape", sys.exit)

    def updateAxes(self, task):
        # Point of reference for each rotation is super important here.
        # We want the axes have the same orientation wrt the screen (render2d),
        # as the orientation of the scene (render) wrt the camera.
        self.axes.setHpr(self.render2d, self.render.getHpr(self.camera))
        return task.cont
