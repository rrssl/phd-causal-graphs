"""
Parameters
----------
spath : string
  Path to the spline.
ndoms : int
  Number of dominoes to put on the path.

"""
import os
import pickle
import sys

from matplotlib import cm
#  import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from shapely.geometry import box
from shapely.affinity import rotate, translate

from config import t, w
sys.path.insert(0, os.path.abspath("../.."))
import spline2d as spl  # noqa
from xp.calibration.doms2pdf import export_domino_run  # noqa
from xp.domino_design.methods import equal_spacing  # noqa
from xp.domino_predictors import DominoRobustness  # noqa
from xp.viewdoms import DominoViewer  # noqa


FREE_ANGLE = 0


class DominoSequence:
    def __init__(self, coords):
        self.coords = coords

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, key):
        return self.coords[key]


class DominoPath(DominoSequence):
    def __init__(self, u, spline):
        self.u = u
        self.spline = spline
        coords = np.zeros((len(u), 3))
        coords[:, 0], coords[:, 1] = spl.splev(u, spline)
        coords[:, 2] = spl.splang(u, spline)
        super().__init__(coords)


class CustomViewer(DominoViewer):
    def __init__(self):
        super().__init__()
        self.cam_distance = 1
        self.min_cam_distance = .1
        self.camLens.set_near(0.01)
        self.zoom_speed = .1


def get_robustness_colored_path(u, spline, rob):
    """Create a colored path showing robustness evolving smoothly along it.

    This is for visual purposes only. There is no such thing as "pathwise
    robustness". The values are interpolated from the pairwise robustness.

    """
    s = spl.arclength(spline, u)
    s_ = (s[:-1]+s[1:])/2
    approx_rob = spl.splprep([rob], u=s_, s=0.)[0]
    dense_u = np.linspace(0, 1, 10*len(s))
    dense_rob = spl.splev(spl.arclength(spline, dense_u), approx_rob)[0]
    dense_rob_colors = cm.viridis(dense_rob)
    return dense_u, spline, dense_rob_colors


class OptimModel:
    """Model used to map the optimizer vector to parameters usable by other
    functions.

    Some constraints are implicitly built into the model (e.g fixed dominoes).
    """
    def __init__(self, n1, s1):
        self.n1 = n1
        self.s1 = s1
        # Init c1
        c1 = np.zeros((n1, 3))
        c1[0, 0], c1[0, 1] = spl.splev(0, s1)
        c1[0, 2] = spl.splang(0, s1)
        c1[-1, 0], c1[-1, 1] = spl.splev(1, s1)
        c1[-1, 2] = spl.splang(1, s1)
        self.c1 = c1

        self._last_x = None

    def update(self, x):
        if np.array_equal(x, self._last_x):
            return
        if not np.isfinite(x).all():
            print("Invalid x = ", x)
            return
        n1_ = self.n1-2  # first and last domino are fixed
        c1 = self.c1

        if FREE_ANGLE:
            c1[1:-1, 0], c1[1:-1, 1] = spl.splev(x[:n1_], self.s1)
            c1[1:-1, 2] = x[n1_:] * 360 - 180
        else:
            c1[1:-1, 0], c1[1:-1, 1] = spl.splev(x, self.s1)
            c1[1:-1, 2] = spl.splang(x, self.s1)

        self._last_x = x.copy()

    @property
    def u1(self):
        n1_ = self.n1-2  # first and last domino are fixed
        return np.concatenate(([0], self._last_x[:n1_], [1]))


class Objective:
    def __init__(self, model, rob_estimator):
        self.model = model
        self.rob_estimator = rob_estimator

    def __call__(self, x):
        self.model.update(x)
        r1 = -self.rob_estimator(self.model.c1).sum()
        return r1


class Objective2:
    def __init__(self, model, rob_estimator):
        self.model = model
        self.rob_estimator = rob_estimator

    def __call__(self, x):
        self.model.update(x)
        p1 = self.rob_estimator._transform(self.model.c1)
        return -np.sum(p1**2)


class BoundsConstraint:
    def __init__(self, bounds):
        self.bounds = np.asarray(bounds).T

    def __call__(self, x):
        return np.concatenate((x - self.bounds[0], self.bounds[1] - x))


class SequenceConstraint:
    def __init__(self, model):
        self.model = model

    def __call__(self, x):
        self.model.update(x)
        diff = np.diff(self.model.u1)
        return diff


class SequenceConstraintBinary(SequenceConstraint):
    def __call__(self, x):
        self.model.update(x)
        u1 = self.model.u1
        return (u1[1:] > u1[:-1]).all()


class NonPenetrationConstraint:
    def __init__(self, model):
        self.model = model
        self.base = box(-t/2, -w/2, t/2,  w/2)

    def _get_penetrations(self, coords):
        base = self.base
        b1 = [translate(rotate(base, a), x, y) for x, y, a in coords]
        return [-b1j.intersection(b1k).area / (t * w)
                for b1j, b1k in zip(b1[:-1], b1[1:])]

    def __call__(self, x):
        self.model.update(x)
        pen = self._get_penetrations(self.model.c1)
        return pen


class NonPenetrationConstraintBinary(NonPenetrationConstraint):
    def _get_penetrations(self, coords):
        base = self.base
        b1 = [translate(rotate(base, a), x, y) for x, y, a in coords]
        return not any(
                b1j.intersects(b1k) for b1j, b1k in zip(b1[:-1], b1[1:]))


class ValidityConstraint:
    def __init__(self, model, rob_estimator):
        self.model = model
        self.rob_estimator = rob_estimator

    def __call__(self, x):
        self.model.update(x)
        r1 = self.rob_estimator(self.model.c1)
        return r1


class BasinHoppingStepTaker:
    def __init__(self, model, stepsize=0.1, seed=None):
        self.spline = model.s1
        if seed is not None:
            np.random.seed(seed)
        self.stepsize = stepsize
        self.base = box(-t/2, -w/2, t/2,  w/2)

    def __call__(self, x):
        s = self.stepsize
        spline = self.spline
        base = self.base
        #  b = self.bounds
        #  x += np.random.uniform(-s, s, x.shape)
        #  np.clip(x, b[0], b[1], out=x)
        for i in range(len(x)):
            if i == 0:
                prev_x = 0
            else:
                prev_x = x[i-1]
            if i == len(x) - 1:
                next_x = 1
            else:
                next_x = x[i+1]

            mini = x[i] - s*(x[i] - prev_x)
            maxi = x[i] + s*(next_x - x[i])

            while True:
                cand_x = np.random.uniform(mini, maxi, 1)
                u = np.array([prev_x, cand_x, next_x])
                coords = np.column_stack(
                        spl.splev(u, spline)+[spl.splang(u, spline)])
                b = [translate(rotate(base, ai), xi, yi)
                     for xi, yi, ai in coords]
                if not (b[0].intersects(b[1]) and b[1].intersects(b[2])):
                    x[i] = cand_x
                    break
        return x


class BasinHoppingStepTester:
    def __init__(self, tests):
        self.tests = tests

    def __call__(self, **kwargs):
        x = kwargs['x_new']
        return all(test(x) for test in self.tests)


def run_optim(init_doms, rob_predictor, method='minimize'):
    if FREE_ANGLE:
        x0 = np.concatenate((init_doms.u[1:-1], init_doms.coords[1:-1, 2]))
    else:
        x0 = init_doms.u[1:-1]
    model = OptimModel(len(init_doms), init_doms.spline)
    objective = Objective(model, rob_predictor)
    seq_cons = SequenceConstraint(model)
    nonpen_cons = NonPenetrationConstraint(model)
    #  val_cons = ValidityConstraint(model, rob_predictor)
    minimizer_kwargs = {
            'bounds': [(0., 1.)] * len(x0),
            'constraints': [
                    {'type': 'ineq', 'fun': seq_cons},
                    {'type': 'ineq', 'fun': nonpen_cons},
                    #  {'type': 'ineq', 'fun': val_cons},
                    ],
            'options': {'disp': True},
        }
    if method == 'minimize':
        res = opt.minimize(objective, x0, **minimizer_kwargs)
    elif method == 'basinhopping':
        accept_test = BasinHoppingStepTester((
            SequenceConstraintBinary(model),
            NonPenetrationConstraintBinary(model)
            ))
        take_step = BasinHoppingStepTaker(model, seed=123)
        res = opt.basinhopping(
                objective, x0, niter=10, T=10.0, stepsize=0.1,
                minimizer_kwargs=minimizer_kwargs,
                take_step=take_step, accept_test=accept_test,
                disp=True, seed=123)
    else:
        print("Invalid method name")
        return
    print("Sequence constraint: ", seq_cons(res.x))
    print("Non-penetration constraint: ", nonpen_cons(res.x))
    model.update(res.x)
    return DominoPath(model.u1, model.s1)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    # Load the path.
    spath = sys.argv[1]
    n_doms = int(sys.argv[2])
    with open(spath, 'rb') as f:
        spline = pickle.load(f)[0]
    spline_shifted = spl.translate(spline, (0, .1))
    spline_shifted2 = spl.translate(spline, (0, .2))

    # Distribute dominoes linearly along this path.
    base_u = equal_spacing(spline, n_doms)
    base_doms = DominoPath(base_u, spline)

    # Distribute dominoes manually along this path.
    manu_u = [0., 0.09, 0.28, 0.36, 0.46, 0.59, 0.69, 0.8, 0.92, 1.]
    manu_doms = DominoPath(manu_u, spline_shifted)

    # Set up and run optimization.
    rob_predictor = DominoRobustness()
    init_doms = DominoPath(base_u, spline_shifted2)
    best_doms = run_optim(init_doms, rob_predictor, method='basinhopping')

    # Show simulation in the different cases.
    viewer = CustomViewer()
    # Base
    base_rob = rob_predictor(base_doms.coords)
    print("Base robustness: ", base_rob, base_rob.sum())
    max_rob = abs(base_rob).max()
    viewer.add_domino_run_from_spline(base_u, spline)
    viewer.add_path(
            *get_robustness_colored_path(base_u, spline, base_rob/max_rob))
    # Manual
    manu_rob = rob_predictor(manu_doms.coords)
    print("Manu robustness: ", manu_rob, manu_rob.sum())
    viewer.add_domino_run(manu_doms.coords)
    viewer.add_path(
            *get_robustness_colored_path(
                manu_u, spline_shifted, manu_rob/max_rob))
    # Optimized
    best_rob = rob_predictor(best_doms.coords)
    print("Best robustness: ", best_rob, best_rob.sum())
    viewer.add_domino_run(best_doms.coords)
    viewer.add_path(
            *get_robustness_colored_path(
                best_doms.u, spline_shifted2, best_rob/max_rob))
    try:
        viewer.run()
    except SystemExit:
        viewer.destroy()

    # Export the result for printing.
    basename = os.path.splitext(spath)[0]
    sheetsize = (21, 29.7)
    export_domino_run(basename + "-base_layout", base_doms.coords, sheetsize)
    export_domino_run(basename + "-best_layout", best_doms.coords, sheetsize)
    export_domino_run(basename + "-manu_layout", manu_doms.coords, sheetsize)
    np.save(basename + "-base_layout", base_doms.coords)
    np.save(basename + "-best_layout", best_doms.coords)
    np.save(basename + "-manu_layout", manu_doms.coords)


if __name__ == "__main__":
    main()
