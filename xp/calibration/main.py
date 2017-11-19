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
import skopt
import stochopy
sys.path.insert(0, os.path.abspath("../.."))
from _ext import go_amp  # noqa

from xp.config import MAX_WAIT_TIME, TOPPLING_ANGLE, t, h  # noqa
from xp.simulate import setup_dominoes_from_path  # noqa


MAKE_VISUALS = 0
BASES = ["data/20171104/I", "data/20171104/U", "data/20171104/O"]
SPLINE_EXT = ".pkl"
DOMS_EXT = "-doms.npz"
REF_TIMES_EXT = "-groundtruth.txt"
FPS = 480  # Hz


def run_simu(world,
             floor_friction=.5,
             domino_friction=.5,
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
    push_dir = Vec3(push_magnitude, 0, 0)

    if _visual:
        from viewers import PhysicsViewer

        class CustomViewer(PhysicsViewer):
            def update_physics(self, task):
                if self.play_physics and self.world_time <= push_duration:
                    dominoes[0].apply_force(push_dir, push_point)
                return super().update_physics(task)

        app = CustomViewer()
        app.cam_distance = 2
        app.min_cam_distance = .2
        app.camLens.set_near(.1)
        app.zoom_speed = .2
        for domino in dominoes:
            app.models.attach_new_node(domino)
        #  doms_np.reparent_to(app.models)
        app.world = world
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


OPTIM_SOLVERS = ('scipy_L-BFGS-B', 'scipy_DE', 'scipy_BH',
                 'skopt_GP', 'skopt_GBRT', 'skopt_forest',
                 'stochopy_DE', 'stochopy_PSO', 'stochopy_CPSO',
                 'stochopy_CMAES',
                 'AMPGO'
                 )


def run_optim(fun, bounds, x0=None, maxiter=100, solver='butt', seed=None):
    bounds = np.asarray(bounds)
    param_vectors = []

    def cbk(x, *args, **kwargs):
        if type(x) is np.ndarray:
            param_vectors.append(x.copy())
        elif type(x) is opt.OptimizeResult:
            param_vectors.append(x.x.copy())

    if solver == 'scipy_L-BFGS-B':
        if x0 is None:
            x0 = bounds[:, 0]
        opt.minimize(fun, x0, bounds=bounds, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'disp': True},
                     callback=cbk)
    if solver == 'scipy_DE':
        opt.differential_evolution(fun, bounds, maxiter=maxiter,
                                   disp=True, seed=seed, callback=cbk)
    if solver == 'scipy_BH':
        if x0 is None:
            x0 = bounds[:, 0]
        opt.basinhopping(fun, x0, niter=maxiter,
                         minimizer_kwargs={'method': 'L-BFGS-B',
                                           'bounds': bounds},
                         disp=True, seed=seed, callback=cbk)
    if solver == 'skopt_GP':
        skopt.gp_minimize(fun, bounds, n_calls=maxiter, x0=x0,
                          verbose=True, random_state=seed,
                          callback=cbk, n_jobs=-1)
    if solver == 'skopt_GBRT':
        skopt.gbrt_minimize(fun, bounds, n_calls=maxiter, x0=x0,
                            verbose=True, random_state=seed,
                            callback=cbk, n_jobs=-1)
    if solver == 'skopt_forest':
        skopt.forest_minimize(fun, bounds, n_calls=maxiter, x0=x0,
                              verbose=True, random_state=seed,
                              callback=cbk, n_jobs=-1)
    if solver.startswith('stochopy'):
        ea = stochopy.Evolutionary(fun, lower=bounds[:, 0], upper=bounds[:, 1],
                                   max_iter=maxiter, random_state=seed,
                                   snap=True)
        if solver == 'stochopy_DE':
            ea.optimize(xstart=x0, solver='de')
        if solver == 'stochopy_PSO':
            ea.optimize(xstart=x0, solver='pso')
        if solver == 'stochopy_CPSO':
            ea.optimize(xstart=x0, solver='cpso')
        if solver == 'stochopy_CMAES':
            ea.optimize(xstart=x0, solver='cmaes', sigma=.1)

        param_vectors = [ea.models[m, :, i]
                         for i, m in enumerate(ea.energy.argmin(axis=0))]
    if solver == 'AMPGO':
        if x0 is None:
            x0 = bounds[:, 0]
        res = go_amp.AMPGO(fun, x0, bounds=bounds, totaliter=maxiter,
                           disp=True, seed=seed, callback=cbk)
        if not np.allclose(param_vectors[-1], res[0]):
            param_vectors.append(res[0])
    if solver == 'butt':
        print("Haha, butt.")

    return param_vectors


def main():
    # Load all files
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

    if MAKE_VISUALS:
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

    # Initialize optimization
    n_events = 2
    push_times = [dd['ref_times'][0] for dd in data_dicts]
    A_ref = np.array([dd['ref_times'][1:] for dd in data_dicts])
    E = np.zeros((n_shapes, n_events))
    push_point = Point3(-t/2, 0, h/3)
    # TODO. Define world and objects once, reset them at each run, check
    # that you get the same results.
    worlds = [setup_dominoes_from_path(
        dd['doms'], dd['spline'], tilt_first_dom=False, _make_geom=True)[1]
        for dd in data_dicts]
    bodies_cache = [
            [body, body.get_transform()]
            for world in worlds for body in world.get_rigid_bodies()
            if not body.is_static()]

    def objective(x, _visual=False):
        #  print(x)
        # Reset world for each shape
        for world in worlds:
            for man in world.get_manifolds():
                man.clear_manifold()
        for body, xform in bodies_cache:
            body.set_transform(xform)
            body.set_linear_velocity(0)
            body.set_angular_velocity(0)
            body.set_active(True)

        # Compute the N_shapes-by-N_events distance matrix
        for i in range(len(E)):
            E[i] = A_ref[i] - run_simu(worlds[i],
                                       floor_friction=x[2],
                                       domino_friction=x[1],
                                       push_magnitude=x[0]/100,
                                       push_duration=push_times[i],
                                       push_point=push_point,
                                       _visual=_visual*(i == 2))
        #  print(E)
        return np.sum(E**2)

    # Ensure reproducibility
    #  objective([0.51414948, 0.47084592, 0.92685865], _visual=0)
    assert objective([.01, .7, .5]) == objective([.01, .7, .5])

    # Evaluate the objective at an arbitrary value.
    push_magnitude = .01
    num_points = 50
    dom_friction_rng = floor_friction_rng = np.linspace(0, 2, num_points)
    filename = "energy-wrt-friction-{}fps.npy".format(FPS)
    try:
        values = np.load(filename)
    except FileNotFoundError:
        grid_x, grid_y = np.meshgrid(dom_friction_rng, floor_friction_rng)
        values = [objective([push_magnitude, x, y])
                  for x, y in zip(grid_x.flat, grid_y.flat)]
        values = np.reshape(values, grid_x.shape)
        np.save(filename, values)
    min_id = values.argmin()
    min_x, min_y = np.unravel_index(min_id, values.shape)
    print("Optim result (bruteforce): {}, with energy: {}".format(
        [push_magnitude, dom_friction_rng[min_x], floor_friction_rng[min_y]],
        values[min_x, min_y]))

    if MAKE_VISUALS:
        # Visualize energy
        fig, ax = plt.subplots()
        im = ax.imshow(values, extent=[0, 2, 0, 2], origin='lower')
        fig.colorbar(im)
        cs = ax.contour(grid_x, grid_y, values, [.01, .1, 1],
                        cmap=plt.cm.gray_r, norm=LogNorm())
        ax.clabel(cs)
        ax.set_xlabel("Domino material 'friction'")
        ax.set_ylabel("Floor material 'friction'")
        ax.set_title("Optimization energy wrt friction -- {} FPS\n".format(FPS)
                     + "(Initial push force is fixed at "
                     "{}N)".format(push_magnitude))

    # Compare solvers and pick the best
    #  x0 = [.01, .5, .5]
    bounds = [[.1, 10.], [.1, 2.], [.1, 2.]]
    maxiter = 2
    filename = "param-vectors-for-each-solver.npy"
    try:
        param_vectors = np.load(filename)
    except FileNotFoundError:
        param_vectors = {
                solver: run_optim(objective, bounds, maxiter=maxiter,
                                  solver=solver, seed=123)
                for solver in OPTIM_SOLVERS
                }
        np.savez(filename, **param_vectors)
    print(param_vectors)
    for solver_name in OPTIM_SOLVERS:
        solvers_fun_vals = [objective(x) for x in param_vectors[solver_name]]

    if MAKE_VISUALS:
        fig, ax = plt.subplots()
        xi = np.arange(1, maxiter+1)
        for i, solver_name in enumerate(OPTIM_SOLVERS):
            ax.plot(xi, solvers_fun_vals[i], label=solver_name)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Objective function")
        ax.set_title("Convergence of each solver")
        ax.legend()

    best_solver_id = np.argmin([fv[-1] for fv in solvers_fun_vals])
    best_solver = OPTIM_SOLVERS[best_solver_id]
    best_x = param_vectors[best_solver][-1]
    best_f = solvers_fun_vals[best_solver_id][-1]
    print("Optim params: {}, with energy: {}, obtained with solver {}".format(
        best_x, best_f, best_solver))

    # TODO. Check that the fall/no-fall distances are coherent with reality.

    # TODO. Produce 240FPS videos to compare to real footage.

    if MAKE_VISUALS:
        plt.show()


if __name__ == "__main__":
    main()
