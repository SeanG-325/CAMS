import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from cams import cfg, data, release
from cams import selection as SL
from cams.baselines import e2e
from cams.pipeline import postprocess, prepare, summarize, summarize_posthoc


def done(p):
    if not p or not os.path.exists(p):
        return set()
    return {str(json.loads(l)["id"]) for l in open(p) if l.strip()}


def feats(xs, c, cache):
    X, y = [], []
    for ex in xs:
        st = prepare(ex, c, "full", cache)
        if not len(st["G"]):
            continue
        X.append(st["F"])
        y.append(SL.silver(st["G"], ex["summary"], c))
    return np.vstack(X), np.concatenate(y)


def gen(xs, p, f):
    sk = done(p)
    with open(p, "a") as g:
        for ex in xs:
            if str(ex["id"]) in sk:
                continue
            g.write(json.dumps(f(ex)) + "\n")
            g.flush()


def main():
    a = argparse.ArgumentParser()
    a.add_argument("cmd")
    a.add_argument("--cfg", default="configs/default.yaml")
    a.add_argument("--data")
    a.add_argument("--dev")
    a.add_argument("--cache")
    a.add_argument("--sel")
    a.add_argument("--pred")
    a.add_argument("--out")
    a.add_argument("--mode", default="full")
    a.add_argument("--ab", default="")
    a.add_argument("--n", type=int)
    a.add_argument("--ntr", type=int)
    a.add_argument("--thetas", default="")
    a.add_argument("--oracle", action="store_true")
    a.add_argument("--no-ss", action="store_true")
    a.add_argument("-o", "--set", action="append", default=[])
    x = a.parse_args()
    c = cfg.load(x.cfg, x.set)
    xs = data.load(x.data, x.n) if x.data else []
    ab = tuple(t for t in x.ab.split(",") if t)

    if x.cmd == "prep":
        for ex in xs:
            prepare(ex, c, x.mode, x.cache)
        print(len(xs))
    elif x.cmd == "train":
        X, y = feats(xs[:x.ntr] if x.ntr else xs, c, x.cache)
        Xd, yd = feats(data.load(x.dev, x.n), c, x.cache)
        s = SL.train_selector(X, y, Xd, yd, c, not x.no_ss)
        SL.save(s, x.out)
        print(json.dumps({"n": len(y), "pos": float(y.mean()), "dev": s["dev"], "lr": s["lr"], "ne": s["ne"]}))
    elif x.cmd == "sum":
        sel = None if x.oracle else SL.load(x.sel)
        gen(xs, x.out, lambda ex: summarize(ex, c, sel, ab, x.cache, (lambda st: SL.silver(st["G"], ex["summary"], c)) if x.oracle else None))
    elif x.cmd == "posthoc":
        sel = SL.load(x.sel)
        gen(xs, x.out, lambda ex: summarize_posthoc(ex, c, sel, x.cache))
    elif x.cmd == "e2e":
        gen(xs, x.out, lambda ex: e2e(ex, c))
    elif x.cmd == "post":
        P = {str(r["id"]): r for r in data.read_jsonl(x.pred)}
        gen([ex for ex in xs if str(ex["id"]) in P], x.out, lambda ex: postprocess(P[str(ex["id"])], ex, c, x.cache))
    elif x.cmd == "sweep":
        sel = SL.load(x.sel)
        for t in [float(v) for v in x.thetas.split(",") if v]:
            c["sel"]["theta"] = t
            gen(xs, "%s.%g.jsonl" % (x.out, t), lambda ex: summarize(ex, c, sel, ab, x.cache))
    elif x.cmd == "release":
        print(release.export([prepare(ex, c, "full", x.cache) for ex in xs], x.out))
    else:
        raise SystemExit(x.cmd)


if __name__ == "__main__":
    main()
