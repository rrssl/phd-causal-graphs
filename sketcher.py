#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sketcher app

@author: Robin Roussel
"""
from direct.showbase.ShowBase import ShowBase
from panda3d.core import load_prc_file_data

from uimixins import Drawable


class Sketcher(Drawable, ShowBase):

    def __init__(self):
        ShowBase.__init__(self)
        self.disable_mouse()
        Drawable.__init__(self, color=(0, 0, 0, 1))
        self.set_background_color((1, 1, 1, 1))

        # Controls
        self.accept("q", self.userExit)
        self.accept("s", self.save_drawing)
        self.accept("c", self.clear_drawing)
        self.start_drawing()

    def start_drawing(self):
        self.accept("mouse1", self.set_draw, [True])
        self.accept("mouse1-up", self.set_draw, [False])

    def stop_drawing(self):
        self.ignore("mouse1")
        self.ignore("mouse1-up")
        self.clear_drawing()


def main():
    load_prc_file_data("", "win-size 1000 1000")
    load_prc_file_data("", "window-title Sketcher")

    app = Sketcher()
    app.run()


if __name__ == "__main__":
    main()
