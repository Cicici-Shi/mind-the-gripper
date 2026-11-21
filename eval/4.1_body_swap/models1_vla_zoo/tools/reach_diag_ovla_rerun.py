"""Entry for tools/reach_diag_m1.py on the OVLA re-run dumps (Fable add1 4.4). reach_diag_m1.py is imported unchanged;
only its key finder (runs/<model>/<cell>) and output dir are redirected to out/ovla_rerun/<tag>/.
Usage: python reach_diag_ovla_rerun.py --rerun-dir out/ovla_rerun/main verify|diag --model openvla_rerun
       --suite libero_spatial --cells g_tw_fc --offsets 0 [--tag ...]
"""
import glob, importlib.util, os, re, sys

Z = os.path.expanduser("$PROJ_ROOT/vla_zoo")
i = sys.argv.index("--rerun-dir")
RD = os.path.abspath(sys.argv[i + 1])
del sys.argv[i:i + 2]
assert os.path.realpath(RD).startswith(os.path.realpath(os.path.join(Z, "out", "ovla_rerun"))), RD

spec = importlib.util.spec_from_file_location("reach_diag_m1", os.path.join(Z, "tools", "reach_diag_m1.py"))
rd = importlib.util.module_from_spec(spec)
sys.modules["reach_diag_m1"] = rd
spec.loader.exec_module(rd)


def keys_of(model, cell, offsets, trials=5):
    keys = {(t, e) for t in range(10) for o in offsets for e in range(o, o + trials)}
    out = {}
    for f in glob.glob(os.path.join(RD, cell, "t*_e*_i*.npz")):
        m = re.search(r"t(\d+)_e(\d+)_i(\d+)", os.path.basename(f))
        k = (int(m.group(1)), int(m.group(2)))
        if k in keys:
            out[k] = f
    return out


rd.keys_of = keys_of
rd.OUT = os.path.join(RD, "reach_diag")
if __name__ == "__main__":
    print(f"reach_diag_m1 (unchanged) on {RD}; output {rd.OUT}", flush=True)
    rd.main()
