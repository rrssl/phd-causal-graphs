import os
import subprocess
import sys

from panda3d.core import NodePath, Point3, Vec3

sys.path.insert(0, os.path.abspath(".."))
import core.primitives as prim  # noqa:E402
from xp.dominoes.templates import create_wave  # noqa:E402
from gui.viewers import PhysicsViewer  # noqa:E402

GEOM = 'HD'
VISUAL = 1
XFORMS = {
    "ball": [Point3(-.15, 0, .1), Vec3(0)],
    "box": [Point3(-.05, 0, .1), Vec3(0)],
    "cylinder": [Point3(.05, 0, .1), Vec3(0, 0, 90)],
    "run": [Point3(.15, 0, .1), Vec3(0, 90, 0)],
    "goblet": [Point3(-.15, 0, -.025), Vec3(0)],
    "track": [Point3(-.05, 0, 0), Vec3(0)],
    "lever": [Point3(.05, 0, 0), Vec3(0)],
    "pulley": [Point3(.15, 0, 0), Vec3(0, 0, 90)],
    "rp": [Point3(-.15, 0, -.08), Vec3(0)],
    "rpp": [Point3(.05, 0, -.08), Vec3(0)],
}


def main():
    world = prim.World()
    scene = NodePath("scene")

    ball = prim.Ball(name="ball", radius=.02, geom=GEOM, phys=False)
    ball.create()
    ball.attach_to(scene, world)

    box = prim.Box(name="box", extents=[.05, .03, .01], geom=GEOM, phys=False)
    box.create()
    box.attach_to(scene, world)

    cylinder = prim.Cylinder(name="cylinder", extents=[.01, .04], geom=GEOM,
                             phys=False)
    cylinder.create()
    cylinder.attach_to(scene, world)

    coords = create_wave([-.05, 0], 0, .095, .025, 18)
    run = prim.DominoRun(
        name="run", extents=[.0025, .008, .012], coords=coords,
        geom=GEOM, phys=False
    )
    run.create()
    run.attach_to(scene, world)

    goblet = prim.Goblet(name="goblet", extents=[.05, .025, .015], geom=GEOM,
                         phys=False)
    goblet.create()
    goblet.attach_to(scene, world)

    track = prim.Track(name="track", extents=[.06, .02, .006, .002], geom=GEOM,
                       phys=False)
    track.create()
    track.attach_to(scene, world)

    lever = prim.Lever(
        name="lever", extents=[.05, .025, .01],
        pivot_pos=[0, 0, 0], pivot_hpr=[0, 90, 0], pivot_extents=[.003, .04],
        geom=GEOM, phys=False
    )
    lever.create()
    lever.attach_to(scene, world)

    pulley = prim.Pulley(
        name="pulley", extents=[.01, .02],
        pivot_pos=[0, 0, 0], pivot_hpr=[0, 0, 0], pivot_extents=[.003, .04],
        geom=GEOM, phys=False
    )
    pulley.create()
    pulley.attach_to(scene, world)

    rp_cyl = prim.Cylinder(
        name="rp-cyl", extents=[.01, .02], geom=GEOM, phys=False, mass=1
    )
    rp_cyl.create().set_pos(Point3(0, 0, -.05))
    rp_cube = prim.Box(
        name="rp-box", extents=[.02, .02, .02], geom=GEOM, phys=False
    )
    rp_cube.create().set_pos(Point3(.1, 0, -.05))
    rope_pulley = prim.RopePulley(
        name="rp",
        obj1=rp_cyl, obj2=rp_cube,
        obj1_pos=Point3(0, 0, .01), obj2_pos=Point3(0, 0, .01),
        rope_length=.18, pulley_coords=[[0, 0, 0], [.1, 0, 0]],
        geom=GEOM, phys=False
    )
    rope_pulley.create()
    rope_pulley.attach_to(scene, world)
    rp_cyl.attach_to(rope_pulley.path, world)
    rp_cube.attach_to(rope_pulley.path, world)

    rpp_cyl = prim.Cylinder(
        name="rpp-cyl", extents=[.01, .02], geom=GEOM, phys=False, mass=1
    )
    rpp_cyl.create().set_pos(Point3(0, 0, -.05))
    rpp_cube = prim.Box(
        name="rpp-box", extents=[.05, .02, .005], geom=GEOM, phys=False
    )
    rpp_cube.create().set_pos(Point3(.1, -.015, -.05))
    rope_pulley = prim.RopePulleyPivot(
        name="rpp",
        obj1=rpp_cyl, obj2=rpp_cube,
        obj1_pos=Point3(0, 0, .01), obj2_pos=Point3(0, .015, 0),
        rope_length=.20, pulley_coords=[[0, 0, 0], [.1, 0, 0]],
        pivot_extents=[.0015, .03], rot_dir=1, coiled_length=.01,
        geom=GEOM, phys=False
    )
    rope_pulley.create()
    rope_pulley.attach_to(scene, world)
    rpp_cyl.attach_to(rope_pulley.path, world)
    rpp_cube.attach_to(rope_pulley.path, world)

    if VISUAL:
        app = PhysicsViewer(world=world)
        app.cam_distance = 1
        app.min_cam_distance = .01
        app.camLens.set_near(.01)
        app.zoom_speed = .01
        app.pivot.set_hpr(180, 0, 0)

        scene.reparent_to(app.models)
        for child in scene.get_children():
            child.set_pos_hpr(*XFORMS[child.get_name()])

        app.run()
    else:
        dir_ = "all_primitives"
        if not os.path.exists(dir_):
            os.mkdir(dir_)
        for child in scene.get_children():
            name = os.path.join(dir_, child.get_name())
            child.write_bam_file(name)
            subprocess.run(["bam2egg", "-o", name + ".egg", name])
            os.remove(name)


if __name__ == "__main__":
    main()
