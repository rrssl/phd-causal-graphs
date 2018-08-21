import bpy

context = bpy.context
scene = context.scene


def clean_up(objects):
    for o in objects:
        if o.type == 'MESH':
            scene.objects.active = o
            bpy.ops.mesh.customdata_custom_splitnormals_clear()
        if o.type == 'EMPTY':
            o.hide = True


def main():
    init = context.active_object
    clean_up(context.selected_objects)
    scene.objects.active = init


if __name__ == "__main__":
    main()
