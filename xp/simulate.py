"""
This module provides the class to simulate a scenario.

"""
from panda3d.core import TransformState

from xp.config import TIMESTEP, TIMEOUT
from gui.viewers import ScenarioViewer


class Simulation:

    def __init__(self, scenario, observers=None, timestep=TIMESTEP):
        self.scenario = scenario
        self.observers = [] if observers is None else observers
        self.timestep = timestep

    def run(self, timeout=TIMEOUT):
        """Run the simulation until the termination condition is met."""
        ts = self.timestep
        world = self.scenario.world
        terminate = self.scenario.terminate
        time = 0.
        while time < timeout:
            # We want to call the observers _before_ normally breaking.
            for obs in self.observers:
                obs(time)
            if terminate.update_and_check(time):
                break
            world.do_physics(ts, 2, ts)
            time += ts
        else:
            print("Simulator timed out")
        # Transforms are globally cached by default. Out of the regular
        # Panda3D task process, we need to empty this cache by hand when
        # running a large number of simulations, to avoid memory overflow.
        TransformState.garbage_collect()

    def run_visual(self, **viewer_kwargs):
        """Run the simulation in visual mode."""
        app = ScenarioViewer(self.scenario, frame_rate=1/self.timestep,
                             **viewer_kwargs)
        try:
            app.run()
        except SystemExit:
            app.destroy()
