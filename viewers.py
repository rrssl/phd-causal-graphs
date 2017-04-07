#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom classes to improve on the basic Panda3D viewer.

@author: Robin Roussel
"""
from direct.showbase.ShowBase import ShowBase
from panda3d.core import AmbientLight, DirectionalLight
from panda3d.core import ShadeModelAttrib

class Modeler(ShowBase):
    """Provides the look and feel of a basic 3D modeler.

    - Flat shading
    - Slightly visible wireframe
    - Directional light towards the object
    """

    def __init__(self):
        super().__init__(self)

        self.render.setAttrib(ShadeModelAttrib.make(ShadeModelAttrib.MFlat))
        self.render.setRenderModeFilledWireframe(.3)

        dlight = DirectionalLight('dlight')
        dlnp = self.camera.attachNewNode(dlight)
        dlnp.lookAt(-self.cam.getPos())
        self.render.setLight(dlnp)

        alight = AmbientLight("alight")
        alight.setColor(.1)
        self.render.setLight(self.render.attachNewNode(alight))

        self.setBackgroundColor(.9, .9, .9)
