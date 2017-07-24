# -*- coding: utf-8 -*-
"""
Functions to create models procedurally within Panda3D.

Examples
--------
Adding the model to the scene graph:

>>> model = make_plane()
>>> render.attachNewNode(model).

Adding the model to an existing node:

>>> from panda3d.core import NodePath
>>> Nodepath(model).reparentTo(other_node_path)



@author: Robin Roussel
"""

from panda3d.core import (Geom, GeomVertexFormat, GeomVertexWriter,
                          GeomVertexData, GeomTriangles, GeomNode)

def make_plane(name="plane"):
    """Creates a plane from scratch. Returns a GeomNode."""

    # 1. Create GeomVertexData and add vertex information
    format = GeomVertexFormat.getV3()
    vdata = GeomVertexData("vertices", format, Geom.UHStatic)
    vdata.setNumRows(4)

    vertexWriter = GeomVertexWriter(vdata, "vertex")
    vertexWriter.addData3f(0, 0, 0)
    vertexWriter.addData3f(10, 0, 0)
    vertexWriter.addData3f(10, 10, 0)
    vertexWriter.addData3f(0, 10, 0)

    # 2. Make primitives and assign vertices to them
    tris = GeomTriangles(Geom.UHStatic)
    # First way of adding vertices: specify each index
    tris.addVertices(0, 1, 3)
    # Indicates that we have finished adding vertices for the first triangle.
    tris.closePrimitive()
    # Since the coordinates are in order we can use this convenience function.
    tris.addConsecutiveVertices(1, 3) #add vertex 1, 2 and 3
    tris.closePrimitive()

    # 3. Make a Geom object to hold the primitives.
    planeGeom = Geom(vdata)
    planeGeom.addPrimitive(tris)

    # 4. Put the Geom in a GeomNode.
    node = GeomNode(name)
    node.addGeom(planeGeom)

    return node
