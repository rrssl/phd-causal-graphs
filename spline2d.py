"""
Utility functions for 2D splines.

"""
import numpy as np
from panda3d.core import LineSegs
from scipy.integrate import quad
from scipy.interpolate import splprep, splev


def arclength(tck):
    """Return the total length of the (t, c, k) spline."""
    # In Cartesian coords, s = integral( sqrt(x'**2 + y'**2) )
    def speed(u):
        dx, dy = splev(u, tck, 1)
        return np.sqrt(dx*dx + dy*dy)
    return quad(speed, 0, 1)[0]


def get_smooth_path(ls, s=.1, prep=lambda x: x):
    """Convert the hand-drawn stroke to a spline on the ground plane.

    Parameters
    ----------
    ls : LineSegs
        Panda3D's polyline object.
    s : float, positive, optional
        Smoothing factor. Higher s => more smoothing.
    prep : callable, optional
        Preprocessing method applied to each vertex. Must accept and return
        a Point3.

    Returns
    -------
    tck : tuple
        Spline parameters; see scipy.interpolate.splprep for more details.
    """
    # Prepare the points for smoothing.
    ls = LineSegs(ls)
    ls.create()
    points = []
    vertices = ls.get_vertices()
    last_added = None
    for v in vertices:
        # Ensure that no two consecutive points are duplicates.
        # Alternatively we could do before the loop:
        # vertices = [v[0] for v in itertools.groupby(vertices)]
        if v != last_added:
            last_added = v
            v = prep(v)
            points.append((v[0], v[1]))
    # Smooth the trajectory.
    # scipy's splprep is more convenient here than panda3d's Rope class,
    # since it gives better control over curve smoothing.
    # TODO. Make changes in the code to handle a BSpline instead of tck.
#    t, c, k = splprep(np.array(points).T, s=s)[0]
#    c = np.column_stack(c)
#    return BSpline(t, c, k)
    return splprep(np.array(points).T, s=s)[0]


def get_spline_phi(u, tck):
    """Get the tangential angle, defined as tan(phi) = dy / dx.

    For convenience wrt Panda3D's conventions, phi is returned in degrees.

    """
    dx, dy = splev(u, tck, 1)
    return np.degrees(np.arctan2(dy, dx))


def get_spline_pos(u, tck, zoffset):
    """Convenience function to convert a sequence of parameter values to
    a sequence of 3D positions along the (t, c, k) spline.

    """
    return np.column_stack(splev(u, tck) + [np.full(u.size, zoffset)])


def show_spline2d(parent, tck, u, label="spline", color=(1, 1, 0, 1)):
    """Create a LineSegs instance representing the (t, c, k) spline, from a
    list of parameter values.

    """
    new_vertices = get_spline_pos(u, tck, 0)
    ls = LineSegs(label)
    ls.set_thickness(4)
    ls.set_color(color)
    ls.move_to(*new_vertices[0])
    for v in new_vertices[1:]:
        ls.draw_to(*v)
    parent.attach_new_node(ls.create())
    return ls
