#!/bin/bash
# maskinfo cell definitions (protocol_maskinfo.md v2 2, 9 item 2): the single source of the frozen parameters and of
# the per-task interactive-object list P. Usage:
#   tools/mi_cells.sh params        -> export lines MI_LAG_STEPS / MI_OUTLINE_W / MI_PROTECT (eval'd by the sbatch)
#   tools/mi_cells.sh spec <C0..C7> -> full spec JSON of a cell (also written into every dump as mr_spec / mi_spec)
#   tools/mi_cells.sh pset          -> the frozen P list (lists/mi_pset.json, enumerated by tools/mi_pset.py)
# Frozen values (protocol v3 2.1): lag 16 (agentview IoU(16) median 0.026 / 0.031 << 0.7, cached-state check);
# width 3 = 1 inner + 2 outer (the v2 width-2 rule fired: Spatial C4 / C1 edge 0.787 / 0.756 < 0.8).
T=$(cd "$(dirname "$0")/.." && pwd)
MI_LAG_STEPS=16
MI_OUTLINE_W=3
MI_PROTECT=interactive
case "$1" in
  params) echo "export MI_LAG_STEPS=$MI_LAG_STEPS MI_OUTLINE_W=$MI_OUTLINE_W MI_PROTECT=$MI_PROTECT" ;;
  spec) cd $T && python3 -c "import sys, json; sys.path.insert(0, '.'); import importlib.util as u
s = u.spec_from_file_location('mi', 'embody/maskinfo.py')
import types; sys.modules['embody'] = types.ModuleType('embody'); sys.modules['embody'].__path__ = ['embody']
from embody import maskinfo as mi
print(json.dumps(mi.mi_spec('$2', $MI_LAG_STEPS, $MI_OUTLINE_W, '$MI_PROTECT'), indent=1))" ;;
  pset) cat $T/lists/mi_pset.json ;;
  *) echo "usage: $0 params | spec <cell> | pset"; exit 1 ;;
esac
