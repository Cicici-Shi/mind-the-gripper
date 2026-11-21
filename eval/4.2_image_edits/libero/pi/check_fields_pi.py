"""Field dry-run with the FROZEN readout code (deviation 10 (a)): imports analysis/prereg_maskinfo_frozen_copy.py
(byte copy of $PROJ_ROOT/maskinfo/analysis/prereg_maskinfo.py, sha256 0a184235...) and runs its own per-file structure
check preflight_block() -- REQUIRED_FIELDS, scalar labels mi_cell / mi_block == path, success dtype (value never
printed), gpu / host, mi_pad_geoms, action / mi_obj_pos / mi_grip_pos / mi_contact_* shapes, same init across cells --
on smoke npz. The only change is the KEY UNIVERSE (module globals TASKS / SHARDS / DESIGN_KEYS / CELLS), because
smoke keys are deliberately outside the formal ep 20-29. No field list is re-typed here.
Usage: python -I tools/check_fields_pi.py RUNS BLOCK TASKS EPS CELLS     e.g. out/smoke_runs pi05_spatial 0 40 C0,...,C8
Prints only: per block OK / REFUSE <code> <message>. Never prints success, lengths or counts of anything per episode.
"""
import importlib.util
import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("prereg_frozen", os.path.join(root, "analysis", "prereg_maskinfo_frozen_copy.py"))
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)

runs, block = sys.argv[1], sys.argv[2]
tasks = [int(x) for x in sys.argv[3].split(",")]
eps = [int(x) for x in sys.argv[4].split(",")]
cells = sys.argv[5].split(",")
P.TASKS = tasks
P.SHARDS = [("smoke", eps)]
P.DESIGN_KEYS = [(t, e) for t in tasks for e in eps]
P.CELLS = cells
print("REQUIRED_FIELDS (frozen):", ",".join(P.REQUIRED_FIELDS))
try:
    files, ignored = P.preflight_block(runs, block)
    print(f"FIELDS_OK block={block} cells={len(files)} keys_per_cell={sorted(len(v) for v in files.values())} ignored={ignored}")
except P.Refuse as r:
    print(f"FIELDS_REFUSE block={block} code={r.code} msg={r.msg}")
    sys.exit(1)
