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
        states = pickle.load(f)
    # Set keyframes
    scene = bpy.context.scene
    for o in scene.objects:
        try:
            anim_id = o.game.properties['anim_id'].value
        except KeyError:
            continue
        try:
            o.game.properties['save_scale']
            has_scale = True
        except KeyError:
            has_scale = False
        o_states = states[anim_id]
        parent = o.parent
        print("Keyframing {}".format(parent))
        parent.rotation_mode = 'QUATERNION'
        for fi, state in enumerate(o_states):
            if has_scale:
                _, x, y, z, w, i, j, k, sx, sy, sz = state
                parent.scale = (sx, sy, sz)
                parent.keyframe_insert(data_path='scale', frame=fi+1)
            else:
                _, x, y, z, w, i, j, k = state
            parent.location = (x, y, z)
            parent.rotation_quaternion = (w, i, j, k)
            parent.keyframe_insert(data_path='location', frame=fi+1)
            parent.keyframe_insert(data_path='rotation_quaternion', frame=fi+1)
    # Set time remapping
    render = scene.render
    old_fps = int(1 / (o_states[1][0] - o_states[0][0]))
    new_fps = render.fps
    print("Remapping {}FPS to {}FPS".format(old_fps, new_fps))
    render.frame_map_old = old_fps // new_fps
    render.frame_map_new = 1
    scene.frame_end = len(o_states) // render.frame_map_old


if __name__ == "__main__":
    register()
    bpy.ops.custom.states_importer('INVOKE_DEFAULT')
