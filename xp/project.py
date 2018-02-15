import sys
from pathlib import Path

_import_path = str(Path(__file__).parents[2].resolve())
print("Adding {} to PATH".format(_import_path))
if _import_path not in sys.path:
    sys.path.insert(0, _import_path)
from code import *  # noqa: F401, F403
