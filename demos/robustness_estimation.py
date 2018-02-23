import os
import sys

import numpy as np
from panda3d.core import Vec4

sys.path.insert(0, os.path.abspath(".."))
import gui.visual as vis  # noqa: E402
from gui.viewers import ScenarioViewer  # noqa: E402
from xp.scenarios import BallPlankDominoes  # noqa: E402


def main():
    sample = BallPlankDominoes.sample_valid(1, max_trials=10)[0]
    scenario = BallPlankDominoes(sample, make_geom=True, ndoms=5)
    app = ScenarioViewer(scenario)

    n = 10
    X_lims = (-.05, .05)
    Y_lims = (-.1, .1)
    X, Y = np.meshgrid(
        np.linspace(*X_lims, num=2*n+1),
        np.linspace(*Y_lims, num=2*n+1),
    )
    frame = X_lims + Y_lims
    ratio = vis.get_aspect_ratio(frame)
    card = vis.ImageCard("field", frame=Vec4(frame),
                         resol=128*np.asarray(ratio))
    card_np = app.visual.attach_new_node(card.generate())
    card_np.set_pos(app.models.find("**/plank*").get_pos())
    fplot = vis.Function2DPlot(datalims=frame, size=ratio, levels=[.5])

    def generate_field_data(task):
        Z = np.sinc(10*np.sqrt(X**2 + Y**2)) * np.cos(task.frame/30)
        card.set_image(fplot.update_data(X, Y, Z))
        return task.cont
    app.task_mgr.add(generate_field_data, "gen_data")

    try:
        app.run()
    except SystemExit:
        app.destroy()


if __name__ == "__main__":
    main()
