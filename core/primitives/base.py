from functools import partial

import panda3d.bullet as bt
from panda3d.core import NodePath, PythonCallbackObject, Vec3


class BulletRootNodePath(NodePath):
    """Special NodePath, parent to bt nodes, that propagates transforms."""

    def __init__(self, *args):
        super().__init__(*args)

        xforms = ['set_pos', 'set_hpr', 'set_pos_hpr',
                  'set_x', 'set_y', 'set_z', 'set_h', 'set_p', 'set_r']

        for xform in xforms:
            setattr(self, xform,
                    partial(self.propagate_xform, xform=xform))

    def propagate_xform(self, *args, xform=''):
        getattr(super(), xform)(*args)
        for child in self.get_children():
            if isinstance(child.node(), bt.BulletBodyNode):
                child.node().set_transform_dirty()


class CallbackSequence(list):
    """Allows to define a sequence of callbacks to give to BulletWorld"""
    def __call__(self, callback_data):
        for cb in self:
            cb(callback_data)
        callback_data.upcall()  # just to be safe


class PrimitiveBase:
    """Base class for all primitives.

    Parameters
    ----------
    name : string
      Name of the primitive.
    geom : {None, 'LD', 'HD'}, optional
      Quality of the visible geometry. None by default (i.e. not visible).
    phys : bool, optional
      Whether this primitive participates in the simulation or not. If false,
      bt_props arguments are ignored. True by default.
    bt_props : dict, optional
      Dictionary of Bullet properties (mass, restitution, etc.). Basically
      the method set_key is called for the Bullet body, where "key" is each
      key of the dictionary. Empty by default (i.e. Bullet default values).

    """

    def __init__(self, name, **bt_props):
        self.name = name
        self.bt_props = bt_props

    @staticmethod
    def _attach(path=None, parent=None, bodies=None, constraints=None,
                physics_callback=None, world=None):
        """Attach the object to the scene and world.

        Parameters
        ----------
        path : NodePath, optional
          Path of the root of the instantiated object(s).
        parent : NodePath, optional
          Path of the node in the scene tree where where objects are added.
        bodies : sequence of bt.BulletRigidBodyNode, optional
          Rigid bodies.
        constraints: sequence of bt.BulletConstraint, optional
          Constraints between rigid bodies.
        physics_callback: callable, optional
          Function to call after each simulation step.
        world : World, optional
          Physical world where the rigid bodies and constraints are added.

        """
        if path is not None and parent is not None:
            path.reparent_to(parent)
        if world is not None:
            if bodies:
                for body in bodies:
                    world.attach(body)
            if constraints:
                for cs in constraints:
                    world.attach_constraint(cs, linked_collision=True)
                    cs.set_debug_draw_size(.05)
            if physics_callback is not None:
                world._callbacks.append(physics_callback)

    # def reset(self):
    #     path = None
    #     if phys:
    #         self.bodies = []
    #         self.constraints = []
    #         physics_callback = None

    def create(self, geom, phys, parent=None, world=None):
        raise NotImplementedError

    def _set_properties(self, bullet_object):
        for key, value in self.bt_props.items():
            getattr(bullet_object, "set_" + key)(value)


class World(bt.BulletWorld):
    """The world in which the primitives live."""

    def __init__(self):
        super().__init__()
        # Trick to have several physics callbacks. Note that the callback
        # object must not be a method of World, otherwise you get a circular
        # reference leading to a memory leak when you instantiate many worlds
        # at once.
        self._callbacks = CallbackSequence()
        self.set_tick_callback(
            PythonCallbackObject(self._callbacks), is_pretick=True
        )

    def set_gravity(self, gravity):
        gravity = Vec3(*gravity)
        super().set_gravity(gravity)
