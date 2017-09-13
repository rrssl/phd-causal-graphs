"""
Global parameters.

"""
SAMPLING_NDIM = 3
timestep = 1. / 120.    # [s]
maxtime = 2.            # [s]
density = 650.          # [kg/m3]
h = .05                 # height [m]
w = h / 3.              # width [m]
t = h / 10.             # thickness [m]

X_MIN = t
X_MAX = 1.5 * h
Y_MIN = -w*1.1
Y_MAX = w*1.1
A_MIN = -90
A_MAX = 90
