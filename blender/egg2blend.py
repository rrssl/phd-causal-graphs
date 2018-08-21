"""
Use me with `blender -b -P path/to/blender/egg2blend.py -- file.egg`

"""
import bpy
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from clean_up_scene import clean_up  # noqa: E402


egg_path = os.path.abspath(sys.argv[-1])
egg_dir, egg_name = os.path.split(egg_path)
blend_out = os.path.splitext(egg_path)[0] + ".blend"
# The following arguments are redundant but work...
bpy.ops.import_scene.egg(filepath=egg_path, directory=egg_dir,
                         files=[{'name': egg_name}])
# Clean up panda3D idiosyncrasies.
clean_up(bpy.context.scene.objects)
# Save file.
bpy.ops.wm.save_mainfile(filepath=blend_out)
