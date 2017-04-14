#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ball run

@author: Robin Roussel
"""
import glob
import os
import sys
import io

from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from panda3d.core import Vec3
from panda3d.core import OrthographicLens
from panda3d.core import TransparencyAttrib
from panda3d.core import TextNode
from panda3d.core import AmbientLight, DirectionalLight
from panda3d.bullet import (BulletWorld, BulletRigidBodyNode, BulletDebugNode,
                            BulletBoxShape, BulletSphereShape,
                            BulletTriangleMesh, BulletTriangleMeshShape)
import scipy.optimize as opt

from coord_grid import ThreeAxisGrid


def model2btm(model):
    """Returns a BulletTriangleMesh from the model's geometry."""
    gn = model.findAllMatches('**/+GeomNode').getPath(0).node()
    mesh = BulletTriangleMesh()
    ts = gn.getTransform()
    for geom in gn.getGeoms():
        mesh.addGeom(geom, True, ts)
    return mesh


class PandaPrinter: #(io.StringIO):
    """Hook that captures stuff printed to stdout and puts it in a TextNode."""
    def __init__(self, text_node, update):
        super().__init__()
        self.printer = text_node
        self.update = update

    def flush(self):
        pass

    def write(self, message):
#        super().write(message)
#        print(message.isspace(), file=sys.__stdout__)
        if not message.isspace():
            self.printer.setText(message)
            self.update()

class EndOptimFlag(Exception):
    def __init__(self, x):
        self.x = x


class BallRun(ShowBase):
    def __init__(self):
        super().__init__(self)
        objects = self.render.attachNewNode("objects")
        objects.setH(-90)  # Rotate all objects to acommodate P3D conventions.
        ## Camera
        cam_pos = Vec3(0, -40, 3)
        self.cam.setPos(cam_pos)
#        self.cam.lookAt(0, 0, 0)
        lens = OrthographicLens()
        lens.setFilmSize(4*6, 3*6)
        self.cam.node().setLens(lens)
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
        halfdims = (bounds[1] - bounds[0]) / 2.
        params = ((0.,  3., 7., 0.,  15., 0.),
                  (0., -3., 4., 0., -15., 0.),
                  (0.,  3.5, 1., 0.,  15., 0.))
        for i, p in enumerate(params):
            node = BulletRigidBodyNode("plank_{}".format(i+1))
            node.addShape(BulletBoxShape(halfdims))
            node_path = objects.attachNewNode(node)
            node_path.setPosHpr(*p)
            world.attachRigidBody(node)
            model.instanceTo(node_path)
        # Ball
        model = self.models['ball']
        bounds = model.getTightBounds()
        halfdims = (bounds[1] - bounds[0]) / 2
        shape = BulletSphereShape(halfdims[0])
        node = BulletRigidBodyNode("ball")
        node.setMass(1.)
        node.addShape(shape)
        node_path = objects.attachNewNode(node)
        node_path.setPos(0., 0., 9.)
        self.init_ball_coord = node_path.getTransform()
        world.attachRigidBody(node)
        model.reparentTo(node_path)
        # Goblet
        model = self.models['goblet']
        shape = BulletTriangleMeshShape(model2btm(model), dynamic=False)
        node = BulletRigidBodyNode("goblet")
        node.addShape(shape)
        node_path = objects.attachNewNode(node)
        node_path.setPosHpr(0., -6., -2., 0., -15., 0.)
        world.attachRigidBody(node)
        model.reparentTo(node_path)
        node_path.setTwoSided(True)  # Show inside
        node_path.setTransparency(TransparencyAttrib.M_alpha)  # Enable see through
        ## Events & controls
        self.taskMgr.add(self.update_goblet, "update_goblet")
        self.taskMgr.add(self.update_physics, "update_physics")
        self.taskMgr.add(self.update_text, "update_text")
        self.accept('d', self.toggle_bullet_debug)
        self.accept('r', self.reset_physics)
        self.accept('o', self.optimize)
        self.accept('escape', sys.exit)
        self.accept('space', self.toggle_physics)
        self.play_physics = False
        ## Visual informations
        # Time
        self.world_time = 0.
        self.wtime_text = OnscreenText(
            parent=self.a2dTopLeft, align=TextNode.A_left,
            pos=(.05, -.1), scale=.05)
        # Print output
        stdout_text = OnscreenText(
            parent=self.a2dBottomLeft, align=TextNode.A_left,
            pos=(.05, .1), scale=.05)
        sys.stdout = PandaPrinter(stdout_text, self.graphicsEngine.renderFrame)
        # Coordinate grid
        ThreeAxisGrid(gridstep=0, subdiv=0).create().reparentTo(self.render)

    def load_models_in(self, dic, from_="assets/"):
        for model_path in glob.iglob(from_ + "*.egg"):
            name = os.path.splitext(os.path.basename(model_path))[0]
            dic[name] = self.loader.loadModel(model_path)

    def optimize(self):
        # Initial parameter values
        goblet = self.render.find("objects/goblet")
        ball = self.render.find("objects/ball")
        plank_1 = self.render.find("objects/plank_1")
        plank_2 = self.render.find("objects/plank_2")
        plank_3 = self.render.find("objects/plank_3")
        x0 = [plank_1.getY(), plank_1.getZ(), plank_1.getP(),
              plank_2.getY(), plank_2.getZ(), plank_2.getP(),
              plank_3.getY(), plank_3.getZ(), plank_3.getP(),
              12.]
        # Additional parameters
        time_resol = 1 / 60
        # Convenience aliases
        world = self.world
        reset_physics = self.reset_physics
        # Optimization functions
        def set_state(x):
            y1, z1, p1, y2, z2, p2, y3, z3, p3, t = x
            plank_1.setPosHpr(0., y1, z1, 0., p1, 0.)
            plank_2.setPosHpr(0., y2, z2, 0., p2, 0.)
            plank_3.setPosHpr(0., y3, z3, 0., p3, 0.)
            world.doPhysics(t, int(t/time_resol + 1), time_resol)
        def objective(x):
            reset_physics()
            set_state(x)
            dist = goblet.getDistance(ball)
            if dist < 1.5:
                raise EndOptimFlag(x)
            return dist
        print("Starting optimization")
        try:
            res = opt.basinhopping(objective, x0, T=1., stepsize=.1, niter=30,
                                   disp=True, minimizer_kwargs={'tol':1.5})
            xf = res.x
        except EndOptimFlag as e:
            xf = e.x
        print("End of optimization")
        self.world_time = xf[-1]

    def reset_physics(self):
        bnp = self.render.find("objects/ball")
        bnp.setTransform(self.init_ball_coord)
        bn = bnp.node()
        bn.clearForces()
        bn.setLinearVelocity(0.)
        bn.setAngularVelocity(0.)
        self.world_time = 0.

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

    def update_goblet(self, task):
        gnp = self.render.find("objects/goblet")
        bnp = self.render.find("objects/ball")
        if gnp.getDistance(bnp) < 2.:
            gnp.setColorScale(0., 4., 0., .8)
        else:
            gnp.clearColorScale()
            gnp.setAlphaScale(.8)
        return task.cont

    def update_physics(self, task):
        if self.play_physics:
            dt = self.taskMgr.globalClock.getDt()
            self.world.doPhysics(dt)
            self.world_time += dt
        return task.cont

    def update_text(self, task):
        self.wtime_text.setText(
            "World time: {:.1f}".format(self.world_time))
        return task.cont


def main():
    app = BallRun()
    app.run()

if __name__ == "__main__":
    main()
