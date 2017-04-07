#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Playing with Panda3D and Bullet

@author: Robin Roussel
"""
from direct.showbase.ShowBase import ShowBase
from panda3d.core import Vec3, NodePath
from panda3d.bullet import (BulletWorld, BulletPlaneShape, BulletRigidBodyNode,
                            BulletSphereShape)

from procedural import make_plane


class MyApp(ShowBase):
    def __init__(self):
        super().__init__(self)

        # Camera
        self.cam.setPos(20, 20, 40)
        self.cam.lookAt(0, 0, 0)

        # World
        world = BulletWorld()
        world.setGravity(Vec3(0, 0, -9.81))

        # Plane
        shape = BulletPlaneShape(Vec3(0, -0.1, 0.9), 1)
        node = BulletRigidBodyNode('Ground')
        node.addShape(shape)
        np = self.render.attachNewNode(node)
        np.setPos(0, 0, -2)
        world.attachRigidBody(node)
        model = NodePath(make_plane())
        model.reparentTo(np)

        # Box
        shape = BulletSphereShape(0.5)
        node = BulletRigidBodyNode('Ball')
        node.setMass(1.0)
        node.addShape(shape)
        np = self.render.attachNewNode(node)
        np.setPos(0, 0, 2)
        world.attachRigidBody(node)
        model = self.loader.loadModel('models/smiley.egg')
        model.flattenLight()
        model.reparentTo(np)

        self.world = world
        self.taskMgr.add(self.update, 'update')

    # Update
    def update(self, task, speedup=1):
      dt = self.taskMgr.globalClock.getDt()
      self.world.doPhysics(speedup*dt, 1, speedup/60)
      return task.cont

def main():
    app = MyApp()
    app.run()

if __name__ == "__main__":
    main()
