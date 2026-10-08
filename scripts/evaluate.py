import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from cams import cfg, data
from cams.evaluation import citation, faith, localize, quality, stats


def main():
    a = argparse.ArgumentParser()
    a.add_argument("cmd")
    a.add_argument("--cfg", default="configs/default.yaml")
    a.add_argument("--data")
    a.add_argument("--pred")
    a.add_argument("--cache")
    a.add_argument("--sys", default="cams")
    a.add_argument("--judge", default="eval")
    a.add_argument("--metrics", default="align,summac,fact")
    a.add_argument("--spec")
    a.add_argument("--out")
    a.add_argument("--n", type=int)
    a.add_argument("-o", "--set", action="append", default=[])
    x = a.parse_args()
    c = cfg.load(x.cfg, x.set)
    xs = {str(e["id"]): e for e in data.load(x.data, x.n)} if x.data else {}
    P = [p for p in data.read_jsonl(x.pred) if str(p["id"]) in xs] if x.pred else []
    R = []

    if x.cmd == "cite":
        R = [citation.cite_eval(p, xs[str(p["id"])]["docs"], c, x.judge) for p in P]
        print(json.dumps({k: citation.agg(R, k) for k in ["rec", "prec", "tok", "span", "uncited"]}))
    elif x.cmd == "loc":
        if x.pred:
            Q = P
        else:
            Q = [localize.localize(e, c, x.sys, x.cache) for e in xs.values()]
        for q in Q:
            g = localize.gold_tok(xs[str(q["id"])])
            r = {"id": q["id"]}
            for t in c["eval"]["taus"]:
                s = localize.score_topic(q["pred"], g, t)
                r["span@%g" % t], r["multi@%g" % t] = s["span"], s["multi"]
            R.append(r)
        print(json.dumps({k: localize.total(R, k) for k in R[0] if k != "id"} if R else {}))
    elif x.cmd == "faith":
        ds = [xs[str(p["id"])]["docs"] for p in P]
        ss = [p["summary"] for p in P]
        m = {}
        for k in x.metrics.split(","):
            m[k] = {"align": faith.alignscore, "summac": faith.summac, "fact": faith.factscore}[k](ds, ss, c)
        R = [dict({"id": p["id"]}, **{k: float(v[i]) for k, v in m.items()}) for i, p in enumerate(P)]
        print(json.dumps({k: float(np.mean(v)) for k, v in m.items()}))
    elif x.cmd == "qual":
        rf = [xs[str(p["id"])]["summary"] for p in P]
        hy = [p["summary"] for p in P]
        rg = quality.rouge(rf, hy)
        bs = quality.bertscore(rf, hy)
        cv = [quality.coverage(r, h, c)[0] for r, h in zip(rf, hy)] if "cov" in x.metrics else [None] * len(P)
        R = [dict({"id": p["id"], "bert": b, "cov": v}, **g) for p, g, b, v in zip(P, rg, bs, cv)]
        print(json.dumps({k: float(np.mean([r[k] for r in R])) for k in R[0] if k != "id" and R[0][k] is not None} if R else {}))
    elif x.cmd == "cmp":
        sp = json.load(open(x.spec))
        cm = []
        for s in sp:
            A = {str(r["id"]): r for r in data.read_jsonl(s["a"])}
            B = {str(r["id"]): r for r in data.read_jsonl(s["b"])}
            ids = sorted(set(A) & set(B))
            f = (lambda r: r[s["key"]][0:2]) if s.get("part") == "p" else (lambda r: r[s["key"]][2:4]) if s.get("part") == "r" else (lambda r: r[s["key"]])
            cm.append(dict(s, A=[f(A[i]) for i in ids], Bv=[f(B[i]) for i in ids], n=len(ids)))
        R = stats.table(cm, c["eval"]["B"], c["eval"]["m"], c["eval"]["alpha"], c["eval"]["seed"])
        for r in R:
            print(json.dumps(r))
    else:
        raise SystemExit(x.cmd)
    if x.out:
        data.write_jsonl(x.out, R)


if __name__ == "__main__":
    main()
