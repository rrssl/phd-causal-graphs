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
        floor_bn.add_shape(bullet.BulletPlaneShape((0, 0, 1), 0))
        # Add to the world
        self.world.attach(floor_bn)
        floor_path = self.parent_path.attach_new_node(floor_bn)
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
        ball_path = self.parent_path.attach_new_node(ball_bn)
        ball_path.attach_new_node(ball_gn)
        ball_path.set_pos(self.ball_pos)

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
        block_path = self.parent_path.attach_new_node(block_bn)
        block_path.attach_new_node(block_gn)
        block_path.set_pos(self.block_pos)
        block_path.set_hpr(self.block_hpr)


class DominoRun:
    """Create a domino run.

    Parameters
    ----------
    path : NodePath
        Path of the node in the scene tree where where objects are added.
    world : BulletWorld
        Physical world where the bullet nodes are added.
    pos : (n,3) float array
        Coordinates of the center of each domino.
    head : (n,) float array
        Heading of each domino.
    extents : (n,3) array of floats or (3,) sequence of floats
        Spatial extents, per domino or global.
    masses : (n,) sequence of floats, or float
        Masses, per domino or extrapolated from first domino.
    """

    def __init__(self, path, world, pos, head, extents, masses):

        self.parent_path = path
        self.world = world

        self.pos = pos
        self.head = head

        extents = np.asarray(extents)
        if extents.shape == (3,):
            extents = np.tile(extents, (len(pos), 1))
        self.extents = extents

        # TODO. Extrpolate from extents.
        masses = np.asarray(masses)
        if masses.shape == ():
            masses = np.tile(masses, (len(pos), 1))
        self.masses = masses

    def create(self):
        """Initialize all the objects in the block."""
        # Note: if all blocks were identical we could create a single model and
        # then call instance_to on the node.

        for i, (p, h, e, m) in enumerate(
                zip(self.pos, self.head, self.extents, self.masses)):
            # Geometry
            block = solid.cube(tuple(e), center=True)
            block_geom = solid2panda(block)
            block_gn = GeomNode("domino_{}_geom".format(i))
            block_gn.add_geom(block_geom)
            # Physics
            block_bn = bullet.BulletRigidBodyNode("domino_{}_solid".format(i))
            block_bn.add_shape(bullet.BulletBoxShape(tuple(e*.5)))
            block_bn.set_mass(m)
            # Add it to the world
            self.world.attach(block_bn)
            block_path = self.parent_path.attach_new_node(block_bn)
            block_path.attach_new_node(block_gn)
            block_path.set_pos(*p)
            block_path.set_h(h)
