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
        self.camDistance = 10.
        self.maxCamDistance = 100.
        self.minCamDistance = .01  # Must be > 0
        self.zoomSpeed = 5.
        self.mouseSpeed = 50.

        # Pivot node
        self.pivot = self.render.attachNewNode("Pivot Point")
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


class Modeler(TurntableViewer):
    """Provides the look and feel of a basic 3D modeler.

    - Flat shading
    - Slightly visible wireframe
    - Directional light towards the object

    TODO:
        - add axes indicator + plane
        - add option to reset the view
        - add option to place the camera s.t. all objects are visible
        - (optional) add a shaded background
    """

    def __init__(self):
        super().__init__()

        # View
        self.render.setAttrib(ShadeModelAttrib.make(ShadeModelAttrib.MFlat))
        self.render.setRenderModeFilledWireframe(.3)

        dlight = DirectionalLight('dlight')
        dlnp = self.camera.attachNewNode(dlight)
        dlnp.lookAt(-self.cam.getPos())
        self.render.setLight(dlnp)

        alight = AmbientLight("alight")
        alight.setColor(.1)
        self.render.setLight(self.render.attachNewNode(alight))

        self.setBackgroundColor(.9, .9, .9)

        # Controls
        self.accept("escape", sys.exit)
