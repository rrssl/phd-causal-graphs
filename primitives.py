#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic primitives for the RGMs.

@author: Robin Roussel
"""
import numpy as np
from panda3d.core import GeomNode
import panda3d.bullet as bullet
import solid
#import trimesh

from utils import trimesh2panda, solid2panda


class Floor:
    """Create an infinite floor.

    Parameters
    ----------
    path : NodePath
        Path of the node in the scene tree where where objects are added.
    world : BulletWorld
        Physical world where the bullet nodes are added.
    make_geom : bool
        True if a visible geometry should be added to the scene.
    """

    def __init__(self, path, world, make_geom=False):
        self.parent_path = path
        self.world = world
        self.make_geom = make_geom

    def create(self):
        """Initialize all the objects in the block."""
        # Geometry
        if self.make_geom:
            vertices = np.array(
                    [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0],
                    dtype=np.float64
                    ).reshape(-1, 3)
            vertices = (vertices - [.5, .5, 0.]) * 100.
            faces = np.array(
                    [0, 1, 3, 1, 2, 3],
                    dtype=np.int64
                    ).reshape(-1, 3)
            vertex_normals = np.array(
                    [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
                    dtype=np.float64
                    ).reshape(-1, 3)
            floor_geom = trimesh2panda(vertices, faces, vertex_normals)
            floor_gn = GeomNode("floor_geom")
            floor_gn.add_geom(floor_geom)
        # Physics
        floor_bn = bullet.BulletRigidBodyNode("floor_solid")
        # TODO. Investigate whether PlaneShape really is the cause of the
        # problem. Maybe it's just a matter of collision margin?
        #  shape = bullet.BulletPlaneShape((0, 0, 1), .1)
        shape = bullet.BulletBoxShape((10, 10, .1))
        #  shape.set_margin(0.0001)
        floor_bn.add_shape(shape)
        # Add to the world
        self.world.attach(floor_bn)
        floor_path = self.parent_path.attach_new_node(floor_bn)
        floor_path.set_pos((0,0,-.1))
        if self.make_geom:
            floor_path.attach_new_node(floor_gn)


class BallRun:
    """Create a ball rolling on a track.

    Parameters
    ----------
    path : NodePath
        Path of the node in the scene tree where where objects are added.
    world : BulletWorld
        Physical world where the bullet nodes are added.
    ball_pos : (3,) floats
        Center of the ball.
    ball_rad : float
        Radius of the ball.
    ball_mass : float
        Mass of the ball.
    block_pos : (3,) floats
        Center of the support block.
    block_hpr : (3,) floats
        Heading-pitch-roll of the block.
    block_ext : (3,) floats
        Spatial half-extents of the block.
    """

    def __init__(self, path, world,
                 ball_pos, ball_rad, ball_mass,
                 block_pos, block_hpr, block_ext):

        self.parent_path = path
        self.world = world

        self.ball_pos = ball_pos
        self.ball_rad = ball_rad
        self.ball_mass = ball_mass

        self.block_pos = block_pos
        self.block_hpr = block_hpr
        self.block_ext = block_ext


    def create(self):
        """Initialize all the objects in the block."""
        # Ball
#        ball = trimesh.creation.uv_sphere(self.ball_rad, count=[12, 12])
        ball = solid.sphere(self.ball_rad, segments=2**4)
#        ball.visual.vertex_colors = (255, 0, 0)
        ball_geom = solid2panda(ball)
#        ball_geom = trimesh2panda(ball.vertices, ball.faces,
#                                  face_normals=ball.face_normals,
#                                  flat_shading=True)
#                                  colors=ball.visual.vertex_colors)
        ball_gn = GeomNode("ball_geom")
        ball_gn.add_geom(ball_geom)
        # Create physical model
        ball_shape = bullet.BulletSphereShape(self.ball_rad)
        ball_bn = bullet.BulletRigidBodyNode("ball_solid")
        ball_bn.set_mass(self.ball_mass)
        ball_bn.add_shape(ball_shape)
        # Add it to the world
        self.world.attach(ball_bn)
        ball_np = self.parent_path.attach_new_node(ball_bn)
        ball_np.attach_new_node(ball_gn)
        ball_np.set_pos(self.ball_pos)

        # Track
#        trans = trimesh.transformations.compose_matrix(angles=block_hpr,
#                                                    translate=block_pos)
#        block = trimesh.creation.box(block_ext, trans)
#        block = trimesh.creation.box(self.block_ext)
        block = solid.cube(tuple(self.block_ext), center=True)
        block_geom = solid2panda(block)
#        block.visual.vertex_colors = (255, 0, 0)
#        block_geom = trimesh2panda(block.vertices, block.faces,
#                                   face_normals=block.face_normals,
#                                   flat_shading=True)
#                                   colors=block.visual.vertex_colors)
        block_gn = GeomNode("block_geom")
        block_gn.add_geom(block_geom)
        # Create physical model
        block_shape = bullet.BulletBoxShape(self.block_ext*.5)
        block_bn = bullet.BulletRigidBodyNode("block_solid")
        block_bn.add_shape(block_shape)
        # Add it to the world
        self.world.attach(block_bn)
        block_np = self.parent_path.attach_new_node(block_bn)
        block_np.attach_new_node(block_gn)
        block_np.set_pos(self.block_pos)
        block_np.set_hpr(self.block_hpr)


class DominoMaker:
    """Factory class to create dominoes and add them to the scene.

    Parameters
    ----------
    path : NodePath
        Path of the node in the scene tree where where objects are added.
    world : BulletWorld
        Physical world where the bullet nodes are added.
    make_geom : bool
        Whether to give each domino a Geom or not.
    reuse_geom : bool
        If True, generated Geoms will be cached and reused if extents are
        the same.

    """

    def __init__(self, path, world, make_geom=True, reuse_geom=True):
        self.parent_path = path
        self.world = world
        self.make_geom = make_geom
        if make_geom:
            self.reuse_geom = reuse_geom
            if reuse_geom:
                self._geom_cache = {}
#        self.pos = np.array([])
#        self.head = np.array([])
#        self.extents = np.array([])
#        self.masses = np.array([])
#
#    def setup(self, pos, head, extents, masses):
#        """Setup the domino run.
#
#        Parameters
#        ----------
#        pos : (n,3) float array
#            Coordinates of the center of each domino.
#        head : (n,) float array
#            Heading of each domino.
#        extents : (n,3) array of floats or (3,) sequence of floats
#            Spatial extents, per domino or global.
#        masses : (n,) sequence of floats, or float
#            Masses, per domino or extrapolated from first domino.
#
#        """
#        self.pos = pos
#        self.head = head
#
#        extents = np.asarray(extents)
#        if extents.shape == (3,):
#            extents = np.tile(extents, (len(pos), 1))
#        self.extents = extents
#
#        # TODO. Extrapolate from extents.
#        masses = np.asarray(masses)
#        if masses.shape == ():
#            masses = np.tile(masses, (len(pos), 1))
#        self.masses = masses
#
#    def create(self):
#        """Initialize all the objects in the block."""
#        # Note: if all blocks were identical we could create a single geom and
#        # then call instance_to on the node.
#        add_domino = self.add_domino
#        for i, (p, h, e, m) in enumerate(
#                zip(self.pos, self.head, self.extents, self.masses)):
#            add_domino(p, h, e, m, "domino_{}".format(i))

    @staticmethod
    def make_domino(extents, prefix):
        block = solid.cube(tuple(extents), center=True)
        block_geom = solid2panda(block)
        block_gn = GeomNode(prefix+"_geom")
        block_gn.add_geom(block_geom)
        return block_gn

    def add_domino(self, pos, head, extents, mass, prefix):
        """Add a new domino to the scene.

        Parameters
        ----------
        pos : Vec3
            Coordinates of the center of the domino.
        head : float
            Heading of the domino.
        extents : Vec3
            Spatial extents of the domino.
        mass : float
            Mass of the domino.
        prefix : string
            Prefix name of the domino.

        Returns
        -------
        dom_np : NodePath
            Path to the BulletRigidBodyNode.

        """
        # Physics
        dom_bn = bullet.BulletRigidBodyNode(prefix+"_solid")
        # TODO. See if using the 'xform' parameter of add_shape wouldn't
        # actually be simpler than using set_pos and set_h later.
        # TODO. Apparently a single collision shape can be shared among
        # multiple objects. Would this make a significant impact in terms
        # of performance though? To investigate.
        # TODO. See if reducing the collision margin improves accuracy.
        # (Currently equal to min(half_dim)/10.)
        shape = bullet.BulletBoxShape(extents*.5)
        #  shape.set_margin(.0001)
        dom_bn.add_shape(shape)
        dom_bn.set_mass(mass)
        # Add it to the world
        self.world.attach(dom_bn)
        dom_np = self.parent_path.attach_new_node(dom_bn)
        dom_np.set_pos(pos)
        dom_np.set_h(head)
        # Geometry
        if self.make_geom:
            if self.reuse_geom:
                try:
                    self._geom_cache[extents].instance_to(dom_np)
                except KeyError:
                    dom_gn = self.make_domino(extents, prefix)
                    self._geom_cache[extents] = dom_np.attach_new_node(dom_gn)
            else:
                dom_gn = self.make_domino(extents, prefix)
                dom_np.attach_new_node(dom_gn)
        return dom_np
