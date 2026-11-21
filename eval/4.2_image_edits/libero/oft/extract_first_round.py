"""Extract the first-round (ep 20-29) constants needed by prereg_maskinfo_confirm.py §3 from verdict_maskinfo.json.
Usage: python3 extract_first_round.py verdict_maskinfo.json out.json   (pure copy of stored values; no re-analysis)"""
import hashlib, json, sys
src, dst = sys.argv[1], sys.argv[2]
raw = open(src, "rb").read()
V = json.loads(raw)
out = {"source": {"file": "verdict_maskinfo.json", "sha256": hashlib.sha256(raw).hexdigest(),
                  "keys": "task 0-9 x ep 20-29 (protocol_maskinfo.md v3)"}, "blocks": {}}
for b in ["oft_spatial", "ofti_spatial", "oft_object", "ofti_object"]:
    r = V["blocks"][b]
    est = {k: {"point": r["estimates"][k]["point"], "ci95": r["estimates"][k]["ci95"]} for k in ("F", "P1", "P2", "D3")}
    dc = {c: {"point": r["D_c"][c]["point"], "ci95": r["D_c"][c]["ci95"], "j_label": r["J"]["labels"][c]}
          for c in ("C1", "C2", "C3")}
    n = r["n_universe"]
    succ = {c: int(round(r["sr"][c] * n / 100.0)) for c in ("C2", "C3", "C6")}
    out["blocks"][b] = {"status": r["status"], "is_A": r["is_A"], "row_letters": r["row_letters"], "n_universe": n,
                        "labels": {k: r["labels"][k] for k in ("P1", "P2", "D3", "G", "P1_word")},
                        "estimates": est, "F_lo95": r["estimates"]["F"]["ci95"][0], "D_c": dc,
                        "successes": succ}
json.dump(out, open(dst, "w"), ensure_ascii=False, indent=1, sort_keys=True)
