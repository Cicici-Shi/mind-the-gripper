"""models1 (2026-11-04): run the byte-copied OFT eval script run_libero_eval_m1.py with the m1_fastsync patch applied to
embody_toy BodyEnv (M1_FASTSYNC=1). The script itself is unchanged; it later imports embody.body_env from the same
EMBODY_DIR and gets the already-patched module. Usage: python launch_m1.py <args of run_libero_eval_m1.py>"""
import os, runpy, sys
sys.path.insert(0, os.path.expanduser(os.environ["EMBODY_DIR"]))
if os.environ.get("M1_FASTSYNC") == "1":
    sys.path.insert(0, os.path.expanduser("$PROJ_ROOT/vla_zoo/common"))
    import m1_fastsync
    m1_fastsync.patch()
    print("M1_FASTSYNC patched", flush=True)
here = os.path.dirname(os.path.abspath(__file__))
sys.argv = [os.path.join(here, "run_libero_eval_m1.py")] + sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
