import os
import sys
from panda3d.core import Point3, Vec3

sys.path.insert(0, os.path.abspath(".."))
import core.primitives as prim  # noqa:E402
from xp.dominoes.templates import create_wave  # noqa:E402
from gui.viewers import PhysicsViewer  # noqa:E402


def main():
    app = PhysicsViewer()
    app.cam_distance = 1
    app.min_cam_distance = .01
    app.camLens.set_near(.01)
    app.zoom_speed = .01
    app.pivot.set_hpr(180, 0, 0)

    ball = prim.Ball(name="ball", radius=.02, geom=True)
    ball.create().set_pos(Point3(-.15, 0, .1))
    ball.attach_to(app.models, app.world)

    box = prim.Box(name="box", extents=[.05, .03, .01], geom=True)
    box.create().set_pos(Point3(-.05, 0, .1))
    box.attach_to(app.models, app.world)

    cylinder = prim.Cylinder(name="cylinder", extents=[.01, .04], geom=True)
    cylinder.create().set_pos_hpr(Point3(.05, 0, .1), Vec3(0, 0, 90))
    cylinder.attach_to(app.models, app.world)

    coords = create_wave([-.05, 0], 0, .095, .025, 18)
    run = prim.DominoRun(
        name="run", extents=[.0025, .008, .012], coords=coords, geom=True
    )
    run.create().set_pos_hpr(Point3(.15, 0, .1), Vec3(0, 90, 0))
    run.attach_to(app.models, app.world)

    goblet = prim.Goblet(name="goblet", extents=[.05, .025, .015], geom=True)
    goblet.create().set_pos(Point3(-.15, 0, -.025))
    goblet.attach_to(app.models, app.world)

    track = prim.Track(name="track", extents=[.06, .02, .006, .002], geom=True)
    track.create().set_pos(Point3(-.05, 0, 0))
    track.attach_to(app.models, app.world)

    lever = prim.Lever(
        name="lever", extents=[.05, .025, .01],
        pivot_pos=[0, 0, 0], pivot_hpr=[0, 90, 0], pivot_extents=[.003, .04],
        geom=True
    )
    lever.create().set_pos(Point3(.05, 0, 0))
    lever.attach_to(app.models, app.world)

    pulley = prim.Pulley(
        name="pulley", extents=[.01, .02],
        pivot_pos=[0, 0, 0], pivot_hpr=[0, 0, 0], pivot_extents=[.003, .04],
        geom=True
    )
    pulley.create().set_pos_hpr(Point3(.15, 0, 0), Vec3(0, 0, 90))
    pulley.attach_to(app.models, app.world)

    rp_cyl = prim.Cylinder(
        name="rp-cyl", extents=[.01, .02], geom=True, mass=1
    )
    rp_cyl.create().set_pos(Point3(0, 0, -.05))
    rp_cube = prim.Box(
        name="rp-box", extents=[.02, .02, .02], geom=True
    )
    rp_cube.create().set_pos(Point3(.1, 0, -.05))
    rope_pulley = prim.RopePulley(
        name="rp",
        obj1=rp_cyl, obj2=rp_cube,
        obj1_pos=Point3(0, 0, .01), obj2_pos=Point3(0, 0, .01),
        rope_length=.18, pulley_coords=[[0, 0, 0], [.1, 0, 0]],
        geom=True
    )
    rope_pulley.create().set_pos(Point3(-.15, 0, -.08))
    rope_pulley.attach_to(app.models, app.world)
    rp_cyl.attach_to(rope_pulley.path, app.world)
    rp_cube.attach_to(rope_pulley.path, app.world)

    rpp_cyl = prim.Cylinder(
        name="rpp-cyl", extents=[.01, .02], geom=True, mass=1
    )
    rpp_cyl.create().set_pos(Point3(0, 0, -.05))
    rpp_cube = prim.Box(
        name="rpp-box", extents=[.05, .02, .005], geom=True
    )
    rpp_cube.create().set_pos(Point3(.1, -.015, -.05))
    rope_pulley = prim.RopePulleyPivot(
        name="rpp",
        obj1=rpp_cyl, obj2=rpp_cube,
        obj1_pos=Point3(0, 0, .01), obj2_pos=Point3(0, .015, 0),
        rope_length=.20, pulley_coords=[[0, 0, 0], [.1, 0, 0]],
        pivot_extents=[.0015, .03], rot_dir=1, coiled_length=.01,
        geom=True
    )
    rope_pulley.create().set_pos(Point3(.05, 0, -.08))
    rope_pulley.attach_to(app.models, app.world)
    rpp_cyl.attach_to(rope_pulley.path, app.world)
    rpp_cube.attach_to(rope_pulley.path, app.world)

    app.run()


if __name__ == "__main__":
    main()
