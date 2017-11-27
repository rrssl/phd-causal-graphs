"""
Calibrate the simulator.

"""
import os
import pickle
import sys

from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
from panda3d.core import Vec3, Point3
import pandas
import scipy.optimize as opt
#  from sklearn.externals import joblib
import skopt
import stochopy
sys.path.insert(0, os.path.abspath("../.."))
from _ext import go_amp  # noqa

from xp.config import MAX_WAIT_TIME, TOPPLING_ANGLE, t, h  # noqa
import xp.simulate as simu  # noqa
from viewers import PhysicsViewer  # noqa


MAKE_VISUALS = 1
BASES = ["data/20171104/I", "data/20171104/U", "data/20171104/O"]
SPLINE_EXT = ".pkl"
DOMS_EXT = "-doms.npz"
REF_TIMES_EXT = "-groundtruth.txt"
FPS = 480  # Hz
OPTIM_SOLVERS = ('scipy_DE',
                 'skopt_GP', 'skopt_GBRT', 'skopt_forest',
                 'stochopy_PSO', 'stochopy_CPSO', 'stochopy_CMAES',
                 )


class CustomViewer(PhysicsViewer):
    def __init__(self, world, dominoes, push_duration, push_dir, push_point):
        super().__init__()
        self.cam_distance = 2
        self.min_cam_distance = .2
        self.camLens.set_near(.1)
        self.zoom_speed = .2

        self.push_duration = push_duration
        self.push_dir = push_dir
        self.push_point = push_point

        for domino in dominoes:
            self.models.attach_new_node(domino)
        self.first_dom = dominoes[0]
        self.world = world

    def update_physics(self, task):
        if self.play_physics and self.world_time <= self.push_duration:
            self.first_dom.apply_force(self.push_dir, self.push_point)
        return super().update_physics(task)


class Objective:
    def __init__(self, data_dicts):
        self.n_events = 2
        self.n_shapes = len(data_dicts)
        self.distributions = [dd['doms'] for dd in data_dicts]
        self.splines = [dd['spline'] for dd in data_dicts]
        self.push_times = [dd['ref_times'][0] for dd in data_dicts]
        self.A_ref = np.array([dd['ref_times'][1:] for dd in data_dicts])
        self.E = np.zeros((self.n_shapes, self.n_events))
        self.push_point = Point3(-t/2, 0, h/3)
        self.denorm_factor = np.array([1/10, 1, 1, 1, 1/10])

    def __call__(self, x, _visual=False, _print=False):
        parameters = x * self.denorm_factor
        # Reset world for each shape
        worlds = [simu.setup_dominoes_from_path(
            distrib, spline, tilt_first_dom=False, _make_geom=_visual)[1]
            for distrib, spline in zip(self.distributions, self.splines)
            ]

        # Compute the N_shapes-by-N_events error matrix
        for i in range(self.n_shapes):
            self.E[i] = self.A_ref[i] - run_simu(
                    worlds[i],
                    floor_friction=parameters[2],
                    domino_friction=parameters[1],
                    domino_restitution=parameters[3],
                    domino_angular_damping=parameters[4],
                    push_magnitude=parameters[0],
                    push_duration=self.push_times[i],
                    push_point=self.push_point,
                    _visual=_visual*(i == 2))
        if _print:
            print("Error matrix:")
            print(self.E)

        out = np.sum(self.E**2)
        #  try:
        #      old_out = self._cache[tuple(x)]
        #  except AttributeError:
        #      self._cache = {}
        #      self._cache[tuple(x)] = out
        #  except KeyError:
        #      self._cache[tuple(x)] = out
        #  else:
        #      if out != old_out:
        #          print("Non deterministic result!!")
        return out


def gen_data():
    data_dicts = []
    for base in BASES:
        # Spline
        with open(base + SPLINE_EXT, 'rb') as f:
            spline = pickle.load(f)[0]
        # Distribution
        doms = np.load(base + DOMS_EXT)['arr_0']
        # Times
        table = pandas.read_csv(base + REF_TIMES_EXT)
        pen_times = (table['pen_leaves'] - table['pen_touches']) / table['fps']
        run_times = (
                table['dn-1_touches_dn'] - table['d1_touches_d2']
                ) / table['fps']
        last_topple_times = (
                table['dn_touches_floor'] - table['dn-1_touches_dn']
                ) / table['fps']
        ref_times = [pen_times.mean(),
                     run_times.mean(),
                     last_topple_times.mean()]
        ref_times_std = [pen_times.std(),
                         run_times.std(),
                         last_topple_times.std()]

        data_dicts.append({
            'spline': spline,
            'doms': doms,
            'ref_times': ref_times,
            'ref_times_std': ref_times_std,
            'n_trials': table.shape[0]
            })
    n_shapes = len(data_dicts)

    if MAKE_VISUALS and 0:
        # Show statistics of ground truth times
        fig, ax = plt.subplots(figsize=plt.figaspect(.5))
        barwidth = .5
        inter_plot_width = barwidth * (n_shapes + 1)
        shapes = ("$I$", "$U$", "$\Omega$")
        for i, dd in enumerate(data_dicts):
            x = np.arange(n_shapes)*inter_plot_width + i*barwidth
            y = dd['ref_times']
            yerr = dd['ref_times_std']
            label = "{} shape ($N_{{dom}}={}$, $N_{{trials}}={}$)".format(
                    shapes[i], len(dd['doms']), dd['n_trials'])
            ax.bar(x, y, barwidth, yerr=yerr, capsize=3, label=label)
        ax.set_title("Duration of some events for each path shape (mean+std)")
        ax.set_xticks(x-(barwidth*i/2))
        ax.set_xticklabels(("Initial push",
                            "From first to last collision",
                            "Last topple"))
        ax.set_ylabel("Time (s)")
        ax.legend()

    return data_dicts


def run_simu(world,
             floor_friction=.5,
             domino_friction=.5,
             domino_restitution=0.,
             domino_angular_damping=0.,
             push_magnitude=.01,
             push_duration=.1,
             push_point=Point3(0),
             timestep=1/FPS,
             _visual=False):
    # Extract the rigid bodies and change their physical parameters.
    floor, *dominoes = world.get_rigid_bodies()
    floor.set_friction(floor_friction)
    for domino in dominoes:
        domino.set_friction(domino_friction)
        domino.set_restitution(domino_restitution)
        domino.set_angular_damping(domino_angular_damping)
    push_dir = Vec3(push_magnitude, 0, 0)

    if _visual:
        app = CustomViewer(
                world, dominoes, push_duration, push_dir, push_point)
        try:
            app.run()
        except SystemExit:
            app.destroy()

    # Run the simulation.
    time = 0.
    n = len(dominoes)
    first_collision_time = MAX_WAIT_TIME
    last_collision_time = MAX_WAIT_TIME
    last_topple_time = MAX_WAIT_TIME
    prev_toppled_id = -1
    prev_toppled_time = 0.
    while True:
        # Check early termination condition
        if prev_toppled_id < n-1:
            next_r = dominoes[prev_toppled_id+1].get_transform().get_hpr()[2]
            if next_r >= TOPPLING_ANGLE:
                prev_toppled_id += 1
                prev_toppled_time = time
        if (time - prev_toppled_time) > MAX_WAIT_TIME:
            break
        # Detect events and record their time
        if (first_collision_time == MAX_WAIT_TIME and
                world.contact_test_pair(
                    dominoes[0], dominoes[1]).get_num_contacts()):
            first_collision_time = time
        elif (last_collision_time == MAX_WAIT_TIME and
                world.contact_test_pair(
                    dominoes[-2], dominoes[-1]).get_num_contacts()):
                last_collision_time = time
        #  print(dominoes[-1].get_transform().get_hpr()[2])
        if (last_topple_time == MAX_WAIT_TIME and
                dominoes[-1].get_transform().get_hpr()[2] >= 89.5):
            last_topple_time = time - last_collision_time
            break
        # Apply push
        if time <= push_duration:
            dominoes[0].apply_force(push_dir, push_point)
        # Step the world
        world.do_physics(timestep, 2, timestep)
        time += timestep
    return last_collision_time - first_collision_time, last_topple_time


def run_bruteforce_optim(fun):
    push_magnitude = .1
    domino_restitution = 0.
    domino_angular_damping = 0.
    num_points = 50
    dom_friction_rng = floor_friction_rng = np.linspace(0, 2, num_points)
    filename = "energy-wrt-friction-{}fps.npy".format(FPS)
    try:
        values = np.load(filename)
    except FileNotFoundError:
        grid_x, grid_y = np.meshgrid(dom_friction_rng, floor_friction_rng)
        values = [fun([push_magnitude,
                       x,
                       y,
                       domino_restitution,
                       domino_angular_damping])
                  for x, y in zip(grid_x.flat, grid_y.flat)]
        values = np.reshape(values, grid_x.shape)
        np.save(filename, values)
    min_id = values.argmin()
    min_x, min_y = np.unravel_index(min_id, values.shape)
    best_x = np.array([
            push_magnitude,
            dom_friction_rng[min_x],
            floor_friction_rng[min_y],
            domino_restitution,
            domino_angular_damping]) * fun.denorm_factor

    if MAKE_VISUALS and 0:
        # Visualize energy
        fig, ax = plt.subplots()
        im = ax.imshow(values, extent=[0, 2, 0, 2], origin='lower')
        fig.colorbar(im)
        grid_x, grid_y = np.meshgrid(dom_friction_rng, floor_friction_rng)
        cs = ax.contour(grid_x, grid_y, values, [.01, .1, 1],
                        cmap=plt.cm.gray_r, norm=LogNorm())
        ax.clabel(cs)
        ax.set_xlabel("Domino material 'friction'")
        ax.set_ylabel("Floor material 'friction'")
        ax.set_title("Optimization energy wrt friction -- {} FPS\n".format(FPS)
                     + "(Initial push force is fixed at "
                     "{}N)".format(push_magnitude))

    return best_x, values[min_x, min_y]


def run_optim(fun, bounds, solver, x0=None, maxiter=100, seed=None,
              disp=False):
    bounds = np.asarray(bounds)
    param_vectors = []

    def cbk(x, *args, **kwargs):
        if type(x) is np.ndarray:
            param_vectors.append(x.copy())
            print("X = {}, f(X) = {}".format(
                param_vectors[-1]*fun.denorm_factor,
                fun(param_vectors[-1])
                ))
        elif type(x) is opt.OptimizeResult:
            param_vectors.append(x.x.copy())

    if solver == 'scipy_L-BFGS-B':
        if x0 is None:
            x0 = bounds[:, 0]
        res = opt.minimize(fun, x0, bounds=bounds, method='L-BFGS-B',
                           options={'maxiter': maxiter, 'disp': disp},
                           callback=cbk)
    if solver == 'scipy_DE':
        res = opt.differential_evolution(fun, bounds, maxiter=maxiter,
                                         disp=disp, seed=seed, callback=cbk)
    if solver == 'scipy_BH':
        if x0 is None:
            x0 = bounds[:, 0]
        res = opt.basinhopping(fun, x0, niter=maxiter,
                               minimizer_kwargs={'method': 'L-BFGS-B',
                                                 'bounds': bounds},
                               disp=disp, seed=seed, callback=cbk)
    if solver == 'skopt_GP':
        res = skopt.gp_minimize(fun, bounds, n_calls=maxiter, x0=x0,
                                verbose=disp, random_state=seed,
                                callback=cbk, n_jobs=-1)
    if solver == 'skopt_GBRT':
        res = skopt.gbrt_minimize(fun, bounds, n_calls=maxiter, x0=x0,
                                  verbose=disp, random_state=seed,
                                  callback=cbk, n_jobs=-1)
    if solver == 'skopt_forest':
        res = skopt.forest_minimize(fun, bounds, n_calls=maxiter, x0=x0,
                                    verbose=disp, random_state=seed,
                                    callback=cbk, n_jobs=-1)
    if solver.startswith('stochopy'):
        ea = stochopy.Evolutionary(fun, lower=bounds[:, 0], upper=bounds[:, 1],
                                   max_iter=maxiter, random_state=seed,
                                   snap=True)
        if solver == 'stochopy_DE':
            if x0 is not None:
                x0 = np.array([x0]*ea._popsize)
            res = ea.optimize(xstart=x0, solver='de')
        if solver == 'stochopy_PSO':
            if x0 is not None:
                x0 = np.array([x0]*ea._popsize)
            res = ea.optimize(xstart=x0, solver='pso')
        if solver == 'stochopy_CPSO':
            if x0 is not None:
                x0 = np.array([x0]*ea._popsize)
            res = ea.optimize(xstart=x0, solver='cpso')
        if solver == 'stochopy_CMAES':
            res = ea.optimize(xstart=x0, solver='cmaes', sigma=.1)

        param_vectors = [ea.models[m, :, i]
                         for i, m in enumerate(ea.energy.argmin(axis=0))]
    if solver == 'AMPGO':
        if x0 is None:
            x0 = bounds[:, 0]
        res = go_amp.AMPGO(fun, x0, bounds=bounds, totaliter=maxiter,
                           disp=disp, seed=seed, callback=cbk)

    xf = res.x if type(res) is opt.OptimizeResult else res[0]
    if param_vectors:
        if (len(param_vectors) < maxiter
                and not np.allclose(param_vectors[-1], xf)):
            param_vectors.append(xf)
    else:
        param_vectors = [xf]

    return param_vectors


def find_best_solver(fun, bounds, maxiter):
    filename = "param-vectors-wrt-solver.npz"
    try:
        param_vectors = np.load(filename)
    except FileNotFoundError:
        param_vectors = {
                solver: run_optim(fun, bounds, maxiter=maxiter,
                                  solver=solver, seed=123, disp=True)
                for solver in OPTIM_SOLVERS
                }
        np.savez(filename, **param_vectors)
    filename = "fun-vals-wrt-solver.npz"
    try:
        solvers_fun_vals = np.load(filename)
    except FileNotFoundError:
        solvers_fun_vals = {
                solver: [fun(x) for x in param_vectors[solver]]
                for solver in OPTIM_SOLVERS
                }
        np.savez(filename, **solvers_fun_vals)

    if MAKE_VISUALS and 0:
        fig, ax = plt.subplots()
        for solver in OPTIM_SOLVERS:
            ax.plot(solvers_fun_vals[solver],  label=solver)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Log of objective function")
        ax.set_yscale('log')
        ax.set_title("Convergence of each solver")
        ax.legend()

    best_solver_id = np.argmin([solvers_fun_vals[solver][-1]
                                for solver in OPTIM_SOLVERS])
    best_solver = OPTIM_SOLVERS[best_solver_id]
    best_x = param_vectors[best_solver][-1]
    best_f = solvers_fun_vals[best_solver][-1]
    return best_solver, best_x, best_f


def find_toppling_bounds(floor_friction=.5,
                         domino_friction=.5,
                         domino_restitution=0.,
                         domino_angular_damping=0.,
                         timestep=1/FPS,
                         _visual=False):
    distances = np.linspace(1.6*t, 1.2*h, 100)
    last_dom_topples = np.zeros_like(distances)
    n_doms = 2
    coords = np.zeros((n_doms, 3))
    for i, distance in enumerate(distances):
        coords[:, 0] = np.arange(n_doms) * distance
        _, world = simu.setup_dominoes(coords, _make_geom=_visual)
        run_time, last_topple_time = run_simu(
                world,
                floor_friction,
                domino_friction,
                domino_restitution,
                domino_angular_damping,
                push_magnitude=0.,
                push_duration=-1,
                timestep=timestep,
                _visual=_visual)
        last_dom_topples[i] += last_topple_time < MAX_WAIT_TIME
    #  print(last_dom_topples)
    min_dist = distances[last_dom_topples.argmax()]
    max_dist = distances[len(distances) - last_dom_topples[::-1].argmax() - 1]
    return min_dist, max_dist


def main():
    # Load all files
    data_dicts = gen_data()

    # Initialize optimization
    objective = Objective(data_dicts)
    # Ensure reproducibility
    #  objective([0.51414948, 0.47084592, 0.92685865], _visual=0)
    _temp1 = [.1, .7, .5, .1, .1]
    #  _temp2 = [.0001, .6, .6, .2, .1]
    assert objective(_temp1) == objective(_temp1)
    #  assert objective(_temp2) == objective(_temp2)
    #  assert objective(_temp1) == objective(_temp1)

    # Find a good value with brute force.
    best_x_brute, best_fun_val_brute = run_bruteforce_optim(objective)
    print("Optim result (bruteforce): {}, with energy: {}".format(
        best_x_brute, best_fun_val_brute))

    # Compare solvers and pick the best
    #  x0 = [.1, .5, .5]
    bounds = [[.01, 1.], [.1, 1.5], [.1, 1.5], [0., 1.], [0., 1.]]
    maxiter = 100
    best_solver, best_x, best_f = find_best_solver(objective, bounds, maxiter)
    print("Optim params: {}, with energy: {}, obtained with solver {}".format(
        best_x*objective.denorm_factor, best_f, best_solver))
    #  objective([.0535, .5, .5, 0., 0.], _print=1, _visual=0)
    #  objective(best_x, _print=1, _visual=0)

    # Find the fall/no-fall distances.
    params = best_x * objective.denorm_factor
    bounds = find_toppling_bounds(
            domino_friction=params[1],
            domino_restitution=params[3],
            domino_angular_damping=params[4],
            floor_friction=params[2],
            timestep=1/(2*FPS),
            _visual=0)
    print("Toppling bounds: {:.2f}cm -- {:.2f}cm".format(
        bounds[0]*100, bounds[1]*100))

    # TODO. Produce 240FPS videos to compare to real footage.

    if MAKE_VISUALS:
        plt.show()


if __name__ == "__main__":
    main()
