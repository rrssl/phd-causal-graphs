import os
import sys
from panda3d.core import Point3, Vec3

sys.path.insert(0, os.path.abspath(".."))
import core.primitives as prim  # noqa:E402
from xp.dominoes.templates import create_circular_arc  # noqa:E402
from gui.viewers import PhysicsViewer  # noqa:E402


def main():
    app = PhysicsViewer()
    app.cam_distance = 1
    app.min_cam_distance = .01
    app.camLens.set_near(.01)
    app.zoom_speed = .01

    floor = prim.Plane(name="floor")
    floor.create()
    floor.attach_to(app.models, app.world)

    ball = prim.Ball(name="ball", radius=.01, geom=True, mass=.01)
    ball.create().set_pos(Point3(0, 0, .4))
    ball.attach_to(app.models, app.world)

    lever = prim.Lever(
        name="lever", extents=[.05, .1, .01],
        pivot_pos_hpr=[0, 0, 0, 0, 0, 90],
        geom=True, mass=.1, angular_damping=.1
    )
    lever.create().set_pos(Point3(0, -.03, .3))
    lever.attach_to(app.models, app.world)

    goblet = prim.Goblet(
        name="goblet", extents=[.1, .05, .03],
        geom=True, mass=.1
    )
    goblet.create().set_pos_hpr(Point3(0, .05, .02), Vec3(0, -40, 0))
    goblet.attach_to(app.models, app.world)

    coords = create_circular_arc([-.1, .01], .1, -90, -0, 10)
    run = prim.DominoRun(
        name="run", extents=[.005, .015, .04], coords=coords,
        geom=True, mass=.005
    )
    run.create()
    run.attach_to(app.models, app.world)

    pulley_ball = prim.Ball(
        name="ball", radius=.01, geom=True, mass=.02
    )
    pulley_ball.create().set_pos(Point3(0, .15, .2))
    pulley_ball.attach_to(app.models, app.world)
    pulley_cube = prim.Box(
        name="box", extents=[.02, .02, .02], geom=True, mass=.01
    )
    pulley_cube.create().set_pos(Point3(0, .27, .2))
    pulley_cube.attach_to(app.models, app.world)
    rope_pulley = prim.RopePulley(
        name="rope-pulley",
        first_object=pulley_ball, second_object=pulley_cube,
        rope_length=.30, pulley_coords=[[0, .16, .25], [0, .25, .3]],
        geom=True
    )
    rope_pulley.create()
    rope_pulley.attach_to(app.models, app.world)
    app.models.ls()

    app.run()


if __name__ == "__main__":
    main()
