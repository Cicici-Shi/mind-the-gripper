"""Aggregate maskinfo-pi load-test progress files (lines "unix_time cumulative_steps cumulative_errors" per client).
Prints ONLY the total executed control steps per second of all clients over [t0, t1] and the error delta.
Usage: python tools/tput_agg.py T0 T1 FILE...   (or: python tools/tput_agg.py --ready N FILE... -> exit 0 when N files
show steps > 0)."""
import sys


def rows(f):
    out = []
    try:
        for line in open(f):
            p = line.split()
            if len(p) == 3:
                out.append((float(p[0]), int(p[1]), int(p[2])))
    except OSError:
        pass
    return out


def at(r, t):
    """last sample at or before t (None if none)."""
    best = None
    for x in r:
        if x[0] <= t:
            best = x
    return best


if sys.argv[1] == "--ready":
    n = int(sys.argv[2])
    ok = sum(1 for f in sys.argv[3:] if any(s > 0 for _, s, _ in rows(f)))
    sys.exit(0 if ok >= n else 1)
t0, t1 = float(sys.argv[1]), float(sys.argv[2])
steps = errs = 0
dt = []
for f in sys.argv[3:]:
    r = rows(f)
    a, b = at(r, t0), at(r, t1)
    if a is None or b is None:
        print(f"TPUT_MISSING {f}")
        continue
    steps += b[1] - a[1]
    errs += b[2] - a[2]
    dt.append(b[0] - a[0])
span = sum(dt) / len(dt) if dt else float("nan")
print(f"TPUT clients={len(sys.argv) - 3} window_s={span:.1f} steps_per_s={steps / span:.2f} errors={errs}")
