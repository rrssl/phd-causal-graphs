#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparing different mesh generators on a simple 3D ball example.

@author: Robin Roussel
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def get_mesh_meshpy():
    # There is some documentation. Not sure, however, if just getting the
    # 'faces' argument is always going to give me the triangles I want.
    #
    # NOTE: combined with trimesh, we can probably directly use the output
    # from 'make_ball' to build a Trimesh object directly; which may be
    # interesting to use the extrusion functionality of meshpy. However,
    # solidpython is just as easy to install, can also be used with trimesh,
    # and provides more functionalities.
    import meshpy.geometry as geo
    from meshpy.tet import MeshInfo, build

    mesh_info = MeshInfo()
    shape = geo.make_ball(1., 10)
    #shape = geo.make_cylinder(1., 2., 20)
    builder = geo.GeometryBuilder()
    builder.add_geometry(*shape)
    builder.set(mesh_info)
    mesh = build(mesh_info)
    verts = np.array(mesh.points).T
    tris = np.array(mesh.faces)

    return verts, tris

def get_mesh_pygmsh():
    # Method is slow (involves writing to disk and executing an external
    # process), and on top of that, doesn't provide the desired output.
    # Documentation is very sparse so I don't understand how this works.
    import pygmsh as pg

    geom = pg.Geometry()
    geom.add_ball([0., 0., 0.], 1., lcar=.8, with_volume=False)
    points, cells, point_data, cell_data, field_data = pg.generate_mesh(
            geom, verbose=False)

    verts = np.array(points).T
    tris = np.array(cells['triangle']) # Curiously the 32 first triangles are good
    return verts, tris

def get_mesh_trimesh():
    # Very little documentation, buth the code itself is well commented and
    # easy to read.
    # Actually more graphics oriented than the two others, which are more
    # FEM oriented. This is much more straightforward to use.
    import trimesh as tri

    ball = tri.creation.uv_sphere(1., count=[10, 10])
    verts = np.array(ball.vertices).T
    tris = np.array(ball.faces)
    return verts, tris

def get_mesh_openscad():
    # Technically we use solidpython, an interfce to OpenSCAD, itself providing
    # an interface to CGAL's CSG functionalities, without the hassle of using
    # CGAL's bindings directly.
    # Fortunately, trimesh provides an interface to OpenSCAD.
    # Using OpenSCAD, however, should be restricted to complex geometries, as
    # the process requires some string parsing and intermediary tempfiles.
    # But for now let's just demonstrate a simple sphere.
    import solid as sol
    import trimesh as tri

    ball = sol.sphere(r=1., segments=16)
    # /!\ trimesh uses a string.Template to perform substitutions in the SCAD
    # script. There, keywords to be replaced are prepended with a '$', which
    # collides with the 'special variables' used by OpenSCAD. Solution: double
    # any '$' sign to escape it in the Template.
    scad = sol.scad_render(ball).replace('$', '$$')
    data = tri.interfaces.scad.interface_scad([], scad)
    verts = data['vertices'].T
    tris = data['faces']
    return verts, tris

def main():
#    verts, tris = get_mesh_meshpy()
#    verts, tris = get_mesh_pygmsh()
#    verts, tris = get_mesh_trimesh()
    verts, tris = get_mesh_openscad()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')
    ax.plot_trisurf(verts[0], verts[1], triangles=tris, Z=verts[2],
                    color='red', alpha=.9)
    ax.set_aspect('equal')
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
