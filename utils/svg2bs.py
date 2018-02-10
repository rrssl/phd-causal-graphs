"""
Convert an SVG path to a BSpline path.

Parameters
----------
spath : string
  Path to the SVG file.
ns : int
  Number of samples used for BSpline fitting.
s : int
  Smoothing factor for BSpline fitting.

"""
import os
import pickle
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as mpath
import numpy as np
from pint import UnitRegistry
from scipy.interpolate import splev, splprep
import svgpathtools as spt
# from svgpathtools import disvg


def svg2mpl(path):
    verts = []
    codes = []
    for i, path_seg in enumerate(path):
        if i == 0:
            pt = path_seg.start
            verts.append([pt.real, pt.imag])
            codes.append(mpath.Path.MOVETO)
        if type(path_seg) is spt.Line:
            pt = path_seg.end
            verts.append([pt.real, pt.imag])
            codes.append(mpath.Path.LINETO)
        elif type(path_seg) is spt.CubicBezier:
            pt = path_seg.control1
            verts.append([pt.real, pt.imag])
            codes.append(mpath.Path.CURVE4)
            pt = path_seg.control2
            verts.append([pt.real, pt.imag])
            codes.append(mpath.Path.CURVE4)
            pt = path_seg.end
            verts.append([pt.real, pt.imag])
            codes.append(mpath.Path.CURVE4)
    return mpath.Path(verts, codes)


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return
    spath = sys.argv[1]
    ns = int(sys.argv[2])
    s = float(sys.argv[3])

    # Load the path
    paths, _, svg_attr = spt.svg2paths(spath, return_svg_attributes=True)

    # Convert the path from whatever user units to meters
    ureg = UnitRegistry()
    width = ureg(svg_attr['width']).to(ureg.meter).magnitude
    height = ureg(svg_attr['height']).to(ureg.meter).magnitude
    viewbox = [float(val) for val in svg_attr['viewBox'].split()]
    x_factor = width / viewbox[2]
    y_factor = height / viewbox[3]
    path = paths[0]
    path = spt.Path(
            *[spt.bpoints2bezier(
                [complex(pt.real*x_factor, pt.imag*y_factor)
                 for pt in seg.bpoints()])
              for seg in path])
    print(path)

    # Compute samples
    u = np.linspace(0, 1, ns)
    samples = np.array([path.point(t) for t in u])
    samples = np.array([samples.real, samples.imag])
    der = np.array([path.derivative(t) for t in u])  # for comparison
    # Fit BSpline
    tck = splprep(samples, u=u, s=s)[0]
    print(tck)
    fitted_samples = np.array(splev(u, tck))
    fitted_der = np.array(splev(u, tck, 1))
    #  fitted_der /= np.linalg.norm(fitted_der, axis=0)

    # Display
    fig, axes = plt.subplots(1, 3, figsize=plt.figaspect(1/3))

    patch = patches.PathPatch(svg2mpl(path), facecolor='none', lw=2)
    # Trick to have a correct legend
    axes[0].plot([], [], c='k', lw=2, label="Original SVG path")
    axes[0].add_patch(patch)
    axes[0].plot(fitted_samples[0], fitted_samples[1], '--', c='y',
                 label="Fitted BSpline")
    #  axes[0].scatter(samples[0], samples[1], marker='x', s=10, c='r')
    axes[0].set_aspect('equal')
    axes[0].legend()
    axes[0].set_title("Fitted BSpline with {} samples\n"
                      "and a smoothing factor of {}".format(ns, s))
    axes[1].plot(u, der.real)
    axes[1].plot(u, der.imag)
    axes[1].legend()
    axes[1].set_title("Derivative of the original path")
    axes[2].plot(u, fitted_der[0], label="$dx/dt$")
    axes[2].plot(u, fitted_der[1], label="$dy/dt$")
    axes[2].legend()
    axes[2].set_title("Derivative of the fitted BSpline")

    plt.show()

    filename = os.path.splitext(spath)[0] + ".pkl"
    with open(filename, 'wb') as f:
        pickle.dump([tck], f)


if __name__ == "__main__":
    main()
