"""
Mesh conversion functions

"""
from panda3d.core import (Geom, GeomVertexFormat, GeomVertexWriter,
                          GeomVertexData, GeomTriangles)
import solid
import trimesh


def trimesh2panda(vertices, triangles, vertex_normals=None, face_normals=None,
                  colors=None, flat_shading=False):
    """Takes triangular mesh data and returns a Panda3D Geom object."""
    assert(flat_shading == (face_normals is not None))
    # Flags
    normals = face_normals if flat_shading else vertex_normals
    has_normals = normals is not None
    has_colors = colors is not None
    # Choose the correct vertex data format
    if has_normals:
        if has_colors:
            fmt = GeomVertexFormat.getV3n3c4()
        else:
            fmt = GeomVertexFormat.getV3n3()
    elif has_colors:
        fmt = GeomVertexFormat.getV3c4()
    else:
        fmt = GeomVertexFormat.getV3()

    # For a proper rendering of flat shading, duplicate all vertices.
    if flat_shading:
        vertices = [vertices[i] for tri in triangles for i in tri]
        triangles = [[3*i, 3*i+1, 3*i+2] for i in range(len(triangles))]

    vdata = GeomVertexData("vertices", fmt, Geom.UHStatic)
    vdata.setNumRows(len(vertices))

    # Add vertex position
    writer = GeomVertexWriter(vdata, "vertex")  # 1.Name is not arbitrary here!
    for vertex in vertices:
        writer.addData3f(*vertex)

    # Add vertex normals
    if has_normals:
        writer = GeomVertexWriter(vdata, "normal")  # Same as (1)
        if flat_shading:
            for normal in normals:
                writer.addData3f(0, 0, 0)
                writer.addData3f(0, 0, 0)
                writer.addData3f(*normal)
        else:
            for normal in normals:
                writer.addData3f(*normal)

    # Add vertex color
    if has_colors:
        writer = GeomVertexWriter(vdata, "color")  # Same as (1)
        for color in colors:
            writer.addData4i(*color)

    # Make primitives and assign vertices to them
    gtris = GeomTriangles(Geom.UHStatic)
    if flat_shading:
        gtris.setShadeModel(Geom.SM_flat_last_vertex)
    for triangle in triangles:
        gtris.addVertices(*triangle)
        gtris.closePrimitive()

    # Make a Geom object to hold the primitives.
    geom = Geom(vdata)
    geom.addPrimitive(gtris)

    return geom


# TODO: Rather than using being hackish with trimesh, implement a proper
# method that generates a STL tempfile, converts it to an EGG tempfile, then
# loads it.
def solid2panda(model):
    scad = solid.scad_render(model).replace('$', '$$')
    data = trimesh.interfaces.scad.interface_scad([], scad)
    return trimesh2panda(data.vertices, data.faces,
                         face_normals=data.face_normals, flat_shading=True)
