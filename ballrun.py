#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ball run

@author: Robin Roussel
"""
import glob, os, sys

from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from panda3d.core import Vec3
from panda3d.core import TextNode
from panda3d.core import AmbientLight, DirectionalLight
from panda3d.bullet import (BulletWorld, BulletRigidBodyNode, BulletDebugNode,
                            BulletBoxShape, BulletSphereShape,
                            BulletTriangleMesh, BulletTriangleMeshShape)

from coord_grid import ThreeAxisGrid


def model2btm(model):
    """Returns a BulletTriangleMesh from the model's geometry."""
    gn = model.findAllMatches('**/+GeomNode').getPath(0).node()
    mesh = BulletTriangleMesh()
    ts = gn.getTransform()
    for geom in gn.getGeoms():
        mesh.addGeom(geom, True, ts)
    return mesh


class BallRun(ShowBase):
    def __init__(self):
        super().__init__(self)
        objects = self.render.attachNewNode("objects")
        objects.setH(-90)  # Rotate all objects to acommodate P3D conventions.
        ## Camera
        cam_pos = Vec3(0, -40, 10)
        self.cam.setPos(cam_pos)
        self.cam.lookAt(0, 0, 0)
        ## Lighting
        # Ambient
        alight = AmbientLight("alight")
        alight.setColor(.1)
        objects.setLight(self.render.attachNewNode(alight))
        # Directional
        dlight = DirectionalLight("dlight")
        dlight.setColor(.8)
        dlp = self.camera.attachNewNode(dlight)  # Attach to camera!
        dlp.lookAt(-cam_pos)
        objects.setLight(dlp)
        ## Physics
        world = BulletWorld()
        self.world = world
        world.setGravity(Vec3(0, 0, -9.81))
        ## Models
        self.models = {}
        self.load_models_in(self.models)
        # Planks
        model = self.models['plank']
        bounds = model.getTightBounds()
        halfdims = (bounds[1] - bounds[0]) / 2
        params = ((0,  3, 7, 0,  15, 0),
                  (0, -3, 4, 0, -15, 0),
                  (0,  3, 1, 0,  15, 0))
        for i, p in enumerate(params):
            node = BulletRigidBodyNode("plank_{}".format(i+1))
            node.addShape(BulletBoxShape(halfdims))
            np = objects.attachNewNode(node)
            np.setPosHpr(*p)
            world.attachRigidBody(node)
            model.instanceTo(np)
        # Ball
        model = self.models['ball']
        bounds = model.getTightBounds()
        halfdims = (bounds[1] - bounds[0]) / 2
        shape = BulletSphereShape(halfdims[0])
        node = BulletRigidBodyNode("ball")
        node.setMass(1.0)
        node.addShape(shape)
        np = objects.attachNewNode(node)
        np.setPos(0, 0, 8)
        world.attachRigidBody(node)
        model.reparentTo(np)
        # Goblet
        model = self.models['goblet']
        shape = BulletTriangleMeshShape(model2btm(model), dynamic=False)
        node = BulletRigidBodyNode("goblet")
        node.addShape(shape)
        np = objects.attachNewNode(node)
        np.setPosHpr(0, -6, -2, 0, -15, 0)
        world.attachRigidBody(node)
        model.reparentTo(np)
        np.setTwoSided(True)  # Show inside
        ## Events & controls
        self.taskMgr.add(self.update, 'update')
        self.accept('d', self.toggle_bullet_debug)
        self.accept('escape', sys.exit)
        self.accept('space', self.toggle_physics)
        self.play_physics = True
        ## Visual informations
        # Time
        self.world_time = 0.
        self.wtime_text = OnscreenText(
            parent=self.a2dTopLeft, align=TextNode.ALeft,
            pos=(0.05, -0.1), scale=.05)
        # Coordinate grid
        ThreeAxisGrid(gridstep=0, subdiv=0).create().reparentTo(self.render)

    def load_models_in(self, dic, from_="assets/"):
        for model_path in glob.iglob(from_ + "*.egg"):
            name = os.path.splitext(os.path.basename(model_path))[0]
            dic[name] = self.loader.loadModel(model_path)

    def toggle_bullet_debug(self):
        try:
            if self._debug_np.isHidden():
                self._debug_np.show()
            else:
                self._debug_np.hide()
        except AttributeError:
            dn = BulletDebugNode("debug")
            dn.showWireframe(True)
#            dn.showConstraints(True)
#            dn.showBoundingBoxes(True)
#            dn.showNormals(True)
            self._debug_np = self.render.attachNewNode(dn)
            self._debug_np.show()
            self.world.setDebugNode(dn)

    def toggle_physics(self):
        self.play_physics = not self.play_physics

    def update(self, task):
        if self.play_physics:
            dt = self.taskMgr.globalClock.getDt()
            self.world.doPhysics(dt)
            # Time
            self.world_time += dt
            self.wtime_text.setText(
                "World time: {:.1f}".format(self.world_time))
        return task.cont


def main():
    app = BallRun()
    app.run()

if __name__ == "__main__":
    main()
