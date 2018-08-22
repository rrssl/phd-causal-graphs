import os
import sys

import numpy as np
from panda3d.core import Vec4
from sklearn.externals import joblib

sys.path.insert(0, os.path.abspath(".."))
import gui.visual as vis  # noqa: E402
from gui.viewers import ScenarioViewer  # noqa: E402
from xp.scenarios import BallPlankDominoes  # noqa: E402

PATH = "data/robustness_estimation/ballplankdominoes-estimator.pkl"


def main():
    # Create the initial scenario and initialize the viewer.
    X0 = BallPlankDominoes.sample_valid(1, max_trials=10)[0]
    scenario = BallPlankDominoes(X0, make_geom=True, ndoms=5)
    ball = scenario.scene.find("ball*")
    plank = scenario.scene.find("plank*")
    app = ScenarioViewer(scenario, 480)
    app.min_cam_distance = 1.
    app.camLens.set_near(0.1)
    app.pivot.set_h(180)
    # Import the estimator
    estimator = joblib.load(PATH)
    scaler = estimator.named_steps['standardscaler']
    svc = estimator.named_steps['svc']
    print(scaler)
    # Sample the local coordinates and compute the field values.
    n = 10
    X0t = scaler.transform([X0])[0]
    Xt_axes = [np.s_[x0t-5:x0t+5:(2*n+1)*1j] for x0t in X0t]
    Xt_grid = np.mgrid[Xt_axes]
    Xt_list = Xt_grid.reshape(Xt_grid.shape[0], -1).T
    F = svc.decision_function(Xt_list)
    F.shape = Xt_grid[0].shape
    X = scaler.inverse_transform(Xt_list).T
    X.shape = Xt_grid.shape
    # Compute the corresponding global values.
    idx = n
    idy = n
    ida = n
    #  A_global = A
    # Create the visual field.
    frame = (X[0, 0, 0, 0], X[0, -1, 0, 0], X[1, 0, 0, 0], X[1, 0, -1, 0])
    ratio = vis.get_aspect_ratio(frame)
    card = vis.ImageCard("field", frame=Vec4(frame),
                         resol=128*np.asarray(ratio))
    app.visual.attach_new_node(card.generate())
    fplot = vis.Function2DPlot(datalims=frame, size=ratio, levels=[0.])
    card.set_image(fplot.update_data(
        X[0, :, :, ida], X[1, :, :, ida], F[:, :, ida]
    ))

    def move_plank(move):
        nonlocal idx, idy, ida
        if move == 'x-':
            idx = max(idx-1, 0)
        if move == 'x+':
            idx = min(idx+1, X.shape[1]-1)
        if move == 'y-':
            idy = max(idy-1, 0)
        if move == 'y+':
            idy = min(idy+1, X.shape[2]-1)
        new_scene = BallPlankDominoes.init_scenario(X[:, idx, idy, ida])[0]
        plank.set_pos(new_scene.find("plank*").get_pos())
        ball.set_pos(new_scene.find("ball*").get_pos())
        app._create_cache()
    app.accept('arrow_left', move_plank, ['x-'])
    app.accept('arrow_right', move_plank, ['x+'])
    app.accept('arrow_down', move_plank, ['y-'])
    app.accept('arrow_up', move_plank, ['y+'])

    try:
        app.run()
    except SystemExit:
        app.destroy()


if __name__ == "__main__":
    main()
