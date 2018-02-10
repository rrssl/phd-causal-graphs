from panda3d.core import Point3, Vec3

from primitives import Plane, Ball, Goblet, Lever, DominoRun
from xp.domino_syncing.domino_templates import create_circular_arc
from viewers import PhysicsViewer


def main():
    app = PhysicsViewer()
    app.cam_distance = 1
    app.min_cam_distance = .01
    app.camLens.set_near(.01)
    app.zoom_speed = .01

    floor = Plane(name="floor")
    floor.create()
    floor.attach_to(app.models, app.world)

    ball = Ball(name="ball", radius=.01, geom=True, mass=.01)
    ball.create().set_pos(Point3(0, 0, .4))
    ball.attach_to(app.models, app.world)

    lever = Lever(name="lever", extents=[.05, .1, .01], geom=True,
                  mass=.1, angular_damping=.1)
    lever.create().set_pos(Point3(0, -.03, .3))
    lever.attach_to(app.models, app.world)

    goblet = Goblet(name="goblet", extents=[.1, .05, .03], geom=True, mass=.1)
    goblet.create().set_pos_hpr(Point3(0, .05, .02), Vec3(0, -40, 0))
    goblet.attach_to(app.models, app.world)

    coords = create_circular_arc([-.1, .01], .1, -90, -0, 100)
    run = DominoRun(name="run", extents=[.005, .015, .04], coords=coords,
                    geom=True, mass=.005)
    run.create()
    run.attach_to(app.models, app.world)

    app.run()


if __name__ == "__main__":
    main()
