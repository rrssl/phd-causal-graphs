import bpy

context = bpy.context
scene = context.scene

init = context.active_object

for o in context.selected_objects:
    if o.type == 'MESH':
        scene.objects.active = o
        bpy.ops.mesh.customdata_custom_splitnormals_clear()
    if o.type == 'EMPTY':
        o.hide = True

scene.objects.active = init