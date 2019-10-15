"""
Functions to export the various layout specifications to a printable format.

"""
import os
from math import ceil

import cairosvg
import svgwrite
from panda3d.core import GeomVertexReader
from shapely.geometry import LineString


class VectorFile:

    def __init__(self, filename, dims_cm, stroke_width=.05):
        self.filename = filename
        width = str(dims_cm[0])
        height = str(dims_cm[1])
        self.stroke_width = stroke_width
        # Define viewBox with the same dimensions as 'size' to get units in cm
        # https://mpetroff.net/2013/08/analysis-of-svg-units/
        self.cont = svgwrite.Drawing(
                filename, size=(width + 'cm', height + 'cm'),
                viewBox='0 0 ' + width + ' ' + height)

    def add_circles(self, positions, radii, linecolor='black'):
        cont = self.cont
        group = cont.add(cont.g(fill='none', stroke=linecolor,
                         stroke_width=self.stroke_width))
        for pos, radius in zip(positions, radii):
            circle = cont.circle(center=(pos[0], pos[1]), r=radius)
            group.add(circle)

    def add_rectangles(self, positions, angles, sizes, linecolor='black'):
        cont = self.cont
        group = cont.add(cont.g(fill='none', stroke=linecolor,
                         stroke_width=self.stroke_width))
        for pos, angle, size in zip(positions, angles, sizes):
            rect = cont.rect(insert=(pos[0]-size[0]/2, pos[1]-size[1]/2),
                             size=size)
            rect.rotate(angle, pos)
            group.add(rect)

    def add_polyline(self, points, linecolor='black'):
        cont = self.cont
        cont.add(cont.polyline(
            points=points, fill='none', stroke=linecolor,
            stroke_width=self.stroke_width
        ))

    def add_text(self, text, position):
        cont = self.cont
        cont.add(cont.text(text, insert=position, style="font-size:1%"))

    def save(self):
        outname = os.path.splitext(self.filename)[0] + ".pdf"
        cairosvg.svg2pdf(self.cont.tostring(), write_to=outname)


def export_layout_to_pdf(scene, filename, sheetsize, plane='xy',
                         exclude=None, flip_u=False, flip_v=False):
    if exclude is None:
        exclude = []
    geom_nodes = scene.graph.find_all_matches("**/+GeomNode")
    objects = []
    min_u = min_v = float('inf')
    max_u = max_v = -min_u
    # First pass: retrieve the vertices.
    for node_path in geom_nodes:
        if node_path.name in exclude:
            continue
        geom_node = node_path.node()
        mat = node_path.get_net_transform().get_mat()
        objects.append([])
        for geom in geom_node.get_geoms():
            if geom.get_primitive_type() is not geom.PT_polygons:
                objects.pop()
                break
            vertex = GeomVertexReader(geom.get_vertex_data(), 'vertex')
            while not vertex.is_at_end():
                point = mat.xform_point(vertex.get_data3f())
                u = getattr(point, plane[0]) * 100 * (1, -1)[flip_u]
                v = getattr(point, plane[1]) * 100 * (1, -1)[flip_v]
                objects[-1].append([u, v])
                # Update the min and max.
                if u < min_u:
                    min_u = u
                if v < min_v:
                    min_v = v
                if u > max_u:
                    max_u = u
                if v > max_v:
                    max_v = v
    # Second pass: get the convex hulls.
    objects = [LineString(o).convex_hull.exterior.coords for o in objects]
    # Determine the size of the whole sheet: i.e., choose landscape or
    # portrait, and determine the number of sheets.
    su, sv = sheetsize
    nu_por = ceil((max_u - min_u)/su)
    nv_por = ceil((max_v - min_v)/sv)
    nu_lan = ceil((max_u - min_u)/sv)
    nv_lan = ceil((max_v - min_v)/su)
    if (nu_por + nv_por) <= (nu_lan + nv_lan):
        nu = nu_por
        nv = nv_por
    else:
        nu = nu_lan
        nv = nu_lan
        su, sv = sv, su
    sheetsize = (nu*su, nv*sv)
    # Create the vector file.
    vec = VectorFile(filename, sheetsize)
    # Add the cut lines.
    guides_color = "#DDDDDD"
    for i in range(1, nu):
        vec.add_polyline([[su*i, 0], [su*i, sheetsize[1]]],
                         linecolor=guides_color)
    for i in range(1, nv):
        vec.add_polyline([[0, sv*i], [sheetsize[0], sv*i]],
                         linecolor=guides_color)
    # Add the stitch guides.
    density = nu + nv
    points = []
    # (This could be rewritten with modulos, but would be harder to read.)
    for i in range(0, density):
        points.append([0, i*sheetsize[1]/density])
        points.append([i*sheetsize[0]/density, 0])
    for i in range(0, density):
        points.append([sheetsize[0], i*sheetsize[1]/density])
        points.append([i*sheetsize[0]/density, sheetsize[1]])
    for i in range(0, density):
        points.append([0, (density-i)*sheetsize[1]/density])
        points.append([i*sheetsize[0]/density, sheetsize[1]])
    for i in range(0, density):
        points.append([sheetsize[0], (density-i)*sheetsize[1]/density])
        points.append([i*sheetsize[0]/density, 0])
    vec.add_polyline(points, linecolor=guides_color)
    # Add the polylines.
    du = sheetsize[0]/2 - (min_u + max_u)/2
    dv = sheetsize[1]/2 - (min_v + max_v)/2
    for o in objects:
        o = [[u + du, v + dv] for u, v in o]
        vec.add_polyline(o)
    # Write the file.
    vec.save()
