import pickle

import bpy
from bpy_extras.io_utils import ImportHelper


class StatesImporter(bpy.types.Operator, ImportHelper):
    bl_idname = "custom.states_importer"
    bl_label = "Import"
    filename_ext = ".pkl"

    def execute(self, context):
        path = self.properties.filepath
        self.report({'INFO'}, "Importing {}".format(path))
        import_states(path)
        return {'FINISHED'}


def register():
    bpy.utils.register_module(__name__)


def unregister():
    bpy.utils.unregister_module(__name__)


def import_states(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    fps = data['metadata']['fps']
    states = data['states']
    # Set keyframes
    scene = bpy.context.scene
    max_frame = 0
    for o in scene.objects:
        try:
            o_states = states[o.name]
        except KeyError:
            continue
        try:
            o.game.properties['save_scale']
            has_scale = True
        except KeyError:
            has_scale = False
        print("Keyframing {}".format(o))
        o.rotation_mode = 'QUATERNION'
        for state in o_states:
            t, x, y, z, w, i, j, k, *_ = state
            frame = int(t*fps) + 1
            if has_scale:
                sx, sy, sz = state[-3:]
                o.scale = (sx, sy, sz)
                o.keyframe_insert(data_path='scale', frame=frame)
            o.location = (x, y, z)
            o.rotation_quaternion = (w, i, j, k)
            o.keyframe_insert(data_path='location', frame=frame)
            o.keyframe_insert(data_path='rotation_quaternion', frame=frame)
        # Keep track of max frame.
        # Do it after the for loop because highest frame is always last!
        if frame > max_frame:
            max_frame = frame
    # Set time remapping
    render = scene.render
    new_fps = render.fps
    print("Remapping {}FPS to {}FPS".format(fps, new_fps))
    render.frame_map_old = fps
    render.frame_map_new = new_fps
    scene.frame_end = max_frame * new_fps // fps


if __name__ == "__main__":
    register()
    bpy.ops.custom.states_importer('INVOKE_DEFAULT')
