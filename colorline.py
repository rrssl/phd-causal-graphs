"""
Defines a function colorline that draws a (multi-)colored 2D line with
coordinates x and y.  The color is taken from optional data in z, and creates a
LineCollection.

z can be:
 - empty, in which case a default coloring will be used based on the position
along the input arrays
 - a single number, for a uniform color [this can also be accomplished with the
usual plt.plot]
 - an array of the length of at least the same length as x, to color according
to this data
 - an array of a smaller length, in which case the colors are repeated along
the curve

The function colorline returns the LineCollection created, which can be
modified afterwards.

See also: plt.streamplot
Source: http://nbviewer.jupyter.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
"""
import numpy as np
from matplotlib.collections import LineCollection


def make_segments(x, y):
    """Create list of line segments from x and y coordinates, in the correct
    format for LineCollection.

    Returns an array of the form numlines x (points per line) x 2 (x and y).

    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    return segments


def colorline(ax, x, y, z=None, **kwargs):
    """Plot a colored line with coordinates x and y.

    Optionally specify colors in the array z.
    kwargs are LineCollection kwargs (cmap, linewidth, alpha, etc.).

    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, **kwargs)

    ax.add_collection(lc)

    return lc
