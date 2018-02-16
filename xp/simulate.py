"""
This module provides the necessary functions to simulate a domino run.

"""
from panda3d.core import load_prc_file_data, Vec4

from .config import TIMESTEP


# The next line avoids a "memory leak" that notably happens when
# BulletWorld.do_physics is called a huge number of times out of the
# regular Panda3D task process. In a nutshell, objects transforms are
# cached and compared by pointer to avoid expensive recomputation; the
# cache is configured to flush itself at the end of each frame, which never
# happens when we don't use frames. The solutions are: don't use the cache
# ("transform-cache 0"), or don't defer flushing to the end of the frame
# ("garbage-collect-states 0"). See
# http://www.panda3d.org/forums/viewtopic.php?t=15645 for a discussion.
load_prc_file_data("", "garbage-collect-states 0")


class Simulation:

    def __init__(self, scenario, observers=None, timestep=TIMESTEP):
        self.scenario = scenario
        self.observers = [] if observers is None else observers
        self.timestep = timestep

    def run(self):
        """Run the simulation until the termination condition is met."""
        ts = self.timestep
        world = self.scenario.world
        terminate = self.scenario.terminate
        time = 0.
        while not terminate(time):
            for obs in self.observers:
                obs(time)
            world.do_physics(ts, 2, ts)
            time += ts

    def run_visual(self):
        """Run the simulation in visual mode."""
        from gui.viewers import PhysicsViewer

        app = PhysicsViewer()
        scenario = self.scenario
        scenario.scene.reparent_to(app.models)
        app.world = scenario.world
        status = None

        def update_status(task):
            scenario.terminate(app.world_time)
            nonlocal status
            if scenario.terminate.status != status:
                status = scenario.terminate.status
                if status == 'success':
                    scenario.scene.set_color(Vec4(0, 1, 0, 1))
                elif status == 'timeout':
                    scenario.scene.set_color(Vec4(1, 0, 0, 1))
                else:
                    scenario.scene.clear_color()
            return task.cont
        app.task_mgr.add(update_status, "update_status")

        def reset():
            scenario.terminate.reset()
            app.reset_physics()
        app.accept('r', reset)

        try:
            app.run()
        except SystemExit:
            app.destroy()
