"""
This module provides the necessary functions to simulate a domino run.

"""
from panda3d.core import load_prc_file_data

from .config import TIMESTEP
from gui.viewers import ScenarioViewer


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
        while True:
            # We want to call the observers _before_ breaking.
            for obs in self.observers:
                obs(time)
            if terminate(time):
                break
            world.do_physics(ts, 2, ts)
            time += ts

    def run_visual(self):
        """Run the simulation in visual mode."""
        app = ScenarioViewer(self.scenario, frame_rate=1/self.timestep)
        try:
            app.run()
        except SystemExit:
            app.destroy()
