"""next (2026-11-06): sustained closed-loop load test of policy servers with REAL requests (pickled by
zoo_client_next.py --dump-inputs during the blind smoke; any cell, no outcome involved).
For each tier of N concurrent client processes (zero think time, i.e. maximum offered load), every process keeps one
websocket open and sends requests back-to-back for --dur seconds; completions in [t0 + --warm, t0 + --dur] count.
Prints per tier: N, completed queries/s (effective throughput), client RTT p50/p95, server infer_ms p50, errors.
Clients are spread round-robin over --ports (several server processes on one GPU).
Usage: python loadgen_next.py --host H --ports p1,p2 --inputs DIR [--tiers 1,2,4,8,12,16] [--dur 75] [--warm 10] --out J
"""
import argparse, glob, json, multiprocessing as mp, os, pickle, random, time

import numpy as np


def worker(host, port, files, t_start, t_end, q):
    from openpi_client import websocket_client_policy
    import websockets.sync.client as _wsc
    _c = _wsc.connect
    _wsc.connect = lambda *a, **k: _c(*a, **{"ping_interval": None, **k})
    pool = [pickle.load(open(f, "rb")) for f in files]
    for el in pool:
        if "new_episode" in el:
            el["new_episode"] = False
    rng = random.Random(port * 1000 + os.getpid())
    out, err = [], 0
    try:
        cl = websocket_client_policy.WebsocketClientPolicy(host, port)
    except Exception as e:
        q.put({"rows": [], "err": 1, "msg": repr(e)})
        return
    while time.time() < t_start:
        time.sleep(0.01)
    while time.time() < t_end:
        el = pool[rng.randrange(len(pool))]
        t = time.time()
        try:
            r = cl.infer(el)
            st = (r.get("server_timing") or {}).get("infer_ms")
            out.append((time.time(), (time.time() - t) * 1000, st))
        except Exception as e:
            err += 1
            time.sleep(1)
            try:
                cl = websocket_client_policy.WebsocketClientPolicy(host, port)
            except Exception:
                pass
    q.put({"rows": out, "err": err})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--ports", required=True)
    ap.add_argument("--inputs", required=True, help="dir tree with *.pkl requests")
    ap.add_argument("--tiers", default="1,2,4,8,12,16")
    ap.add_argument("--dur", type=float, default=75)
    ap.add_argument("--warm", type=float, default=10)
    ap.add_argument("--max-files", type=int, default=40)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.inputs, "**", "*.pkl"), recursive=True))
    random.Random(0).shuffle(files)
    files = files[: a.max_files]
    assert files, a.inputs
    ports = [int(p) for p in a.ports.split(",")]
    res = []
    for n in [int(x) for x in a.tiers.split(",")]:
        q = mp.Queue()
        t0 = time.time() + 15          # connection set-up before the window
        ps = [mp.Process(target=worker, args=(a.host, ports[i % len(ports)], files, t0, t0 + a.dur, q)) for i in range(n)]
        for p in ps:
            p.start()
        got = [q.get() for _ in ps]
        for p in ps:
            p.join()
        rows = [r for g in got for r in g["rows"] if r[0] >= t0 + a.warm]
        lat = [r[1] for r in rows]
        srv = [r[2] for r in rows if r[2] is not None]
        win = a.dur - a.warm
        row = {"label": a.label, "clients": n, "servers": len(ports), "qps": round(len(rows) / win, 3),
               "rtt_p50": round(float(np.percentile(lat, 50)), 1) if lat else None,
               "rtt_p95": round(float(np.percentile(lat, 95)), 1) if lat else None,
               "srv_ms_p50": round(float(np.percentile(srv, 50)), 1) if srv else None,
               "errors": sum(g["err"] for g in got), "window_s": win, "n_files": len(files)}
        print(json.dumps(row), flush=True)
        res.append(row)
        with open(a.out, "a") as f:
            f.write(json.dumps(row) + "\n")
    return res


if __name__ == "__main__":
    main()
