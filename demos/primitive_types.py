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
    floor.create(geom=None, phys=True, parent=app.models, world=app.world)

    ball = prim.Ball(name="ball", radius=.01, mass=.01)
    ball.create(
        geom='LD', phys=True, parent=app.models, world=app.world
    ).set_pos(Point3(0, 0, .4))

    lever = prim.Lever(
        name="lever", extents=[.05, .1, .01],
        pivot_pos=[0, 0, 0], pivot_hpr=[0, 0, 90], pivot_extents=[.004, .06],
        mass=.1, angular_damping=.1
    )
    lever.create(
        geom='LD', phys=True, parent=app.models, world=app.world
    ).set_pos(Point3(0, -.03, .3))

    goblet = prim.Goblet(name="goblet", extents=[.1, .05, .03], mass=.1)
    goblet.create(
        geom='LD', phys=True, parent=app.models, world=app.world
    ).set_pos_hpr(Point3(0, .05, .02), Vec3(0, -40, 0))

    coords = create_circular_arc([-.1, .01], .1, -90, -0, 10)
    run = prim.DominoRun(
        name="run", extents=[.005, .015, .04], coords=coords, mass=.005
    )
    run.create(geom='LD', phys=True, parent=app.models, world=app.world)

    pulley_ball = prim.Ball(name="ball", radius=.01, mass=.2)
    pulley_ball_path = pulley_ball.create(
        geom='LD', phys=True, parent=app.models, world=app.world
    )
    pulley_ball_path.set_pos(Point3(0, .16, .2))
    pulley_cube = prim.Box(name="box", extents=[.02, .02, .02], mass=.19)
    pulley_cube_path = pulley_cube.create(
        geom='LD', phys=True, parent=app.models, world=app.world
    )
    pulley_cube_path.set_pos(Point3(0, .28, .2))
    rope_pulley = prim.RopePulley(
        name="rope-pulley",
        comp1_pos=Point3(0, 0, .01), comp2_pos=Point3(0, 0, .01),
        rope_length=.30, pulleys=[[0, .17, .25], [0, .25, .3]],
    )
    rope_pulley.create(
        geom='LD', phys=True, parent=app.models, world=app.world,
        components=(pulley_ball_path, pulley_cube_path)
    )

    app.run()


if __name__ == "__main__":
    main()
