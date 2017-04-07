#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions

@author: Robin Roussel
"""
from panda3d.core import (Geom, GeomVertexFormat, GeomVertexWriter,
                          GeomVertexData, GeomTriangles)

def trimesh2panda(vertices, triangles, normals=None, colors=None):
    """Takes triangular mesh data and returns a Panda3D Geom object."""
    # Choose the correct vertex data format
    has_colors = colors is not None
    has_normals = normals is not None
    if has_normals:
        if has_colors:
            fmt = GeomVertexFormat.getV3n3c4()
        else:
            fmt = GeomVertexFormat.getV3n3()
    elif has_colors:
        fmt = GeomVertexFormat.getV3c4()
    else:
        fmt = GeomVertexFormat.getV3()

    vdata = GeomVertexData("vertices", fmt, Geom.UHStatic)
    vdata.setNumRows(len(vertices))

    # Add vertex position
    writer = GeomVertexWriter(vdata, "vertex") # Name is not arbitrary here!
    for vertex in vertices:
        writer.addData3f(*vertex)

    # Add vertex normals
    if has_normals:
        writer = GeomVertexWriter(vdata, "normal") # Name is not arbitrary here!
        for normal in normals:
            writer.addData3f(*normal)

    # Add vertex color if there's one
    if has_colors:
        writer = GeomVertexWriter(vdata, "color") # Name is not arbitrary here!
        for color in colors:
            writer.addData4i(*color)

    # Make primitives and assign vertices to them
    gtris = GeomTriangles(Geom.UHStatic)
    for triangle in triangles:
        gtris.addVertices(*triangle)
        gtris.closePrimitive()

    # Make a Geom object to hold the primitives.
    geom = Geom(vdata)
    geom.addPrimitive(gtris)

    return geom
