"""
Experimenting with BSplines and Bezier curves.
We assume our BSplines to be:
    - univariate, with 2D control points
    - aperiodic (the first and last point are distinct)
    - clamped (the curve is tangent to the first and last legs at the first
    and last control points)

In that case, the theory is that the control points of a BSpline can be used
as control points for a sequence of Bezier curves, as long as each internal
knot has a multiplicity equal to the degree of the BSpline.

"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev, insert
import matplotlib.patches as patches
from matplotlib.path import Path


def increase_internal_knot_multiplicity(spline):
    """Increase the multiplicity of the internal knots so that it is equal
    to the degree of the BSpline."""
    k = spline[2]  # degree
    for t in spline[0][k+1:-(k+1)]:
        print("Inserting knot ", t)
        spline = insert(t, spline, m=(k-1))
    # NB. insert(), for some reason, appends k+1 empty values at the end.
    # This is not an issue with splev() (or even recursive calls to insert()),
    # but it becomes one when we want to visualize the control points,
    # or convert the BSpline to a sequence of Bezier curves.

    spline[1][0] = spline[1][0][:-(k+1)]  # this is not tuple assignment
    spline[1][1] = spline[1][1][:-(k+1)]
    return spline


def main():
    with open("xp/domino_design/data/latest/candidates.pkl", 'rb') as f:
        splines = pickle.load(f)
    spline = splines[2]
    spline2 = increase_internal_knot_multiplicity(spline)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=plt.figaspect(1/3),
                                        sharex=True, sharey=True,
                                        subplot_kw={'aspect':'equal'})

    ax1.plot(*splev(np.linspace(0, 1, 1000), spline), c='b')
    ax1.plot(spline[1][0], spline[1][1], marker='o', c='orange',
             markerfacecolor='b')

    ax2.plot(*splev(np.linspace(0,1,1000), spline2), c='g')
    ax2.plot(spline2[1][0], spline2[1][1], marker='o', c='r',
             markerfacecolor='g')

    verts = np.column_stack(spline2[1])
    codes = [Path.MOVETO] + [Path.CURVE4]*(len(verts)-1)
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor='none', lw=2)
    ax3.add_patch(patch)

    plt.show()


if __name__ == "__main__":
    main()
