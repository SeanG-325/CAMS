import json
import os

import numpy as np

from . import models as M
from .cluster import cluster_claims, gemb, merge_spans
from .conflict import find_conflicts
from .extract import extract_all
from .features import FEATS, featurize, span_text
from .rewrite import arrange, parse, rewrite
from .selection import mmr, score_clusters
from .textutil import sents
from .verify import ev, verify


def _rd(p):
    if p and os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def _wr(p, x):
    if p:
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        with open(p, "w") as f:
            json.dump(x, f)


def prepare(ex, c, mode="full", cache=None):
    D = ex["docs"]
    q = (lambda s: os.path.join(cache, "%s.%s.json" % (ex["id"], s))) if cache else (lambda s: None)
    a = _rd(q("claims"))
    if a is None:
        C, nd = extract_all(D, c)
        a = {"C": C, "nd": nd}
        _wr(q("claims"), a)
    C = a["C"]
    E = M.emb(c)([x["claim"] for x in C]) if C else np.zeros((0, 1))
    b = _rd(q(mode))
    if b is None:
        G, Eg, lab = cluster_claims(C, c, E, mode)
        X = find_conflicts(C, E, lab, c)
        F = featurize(D, C, G, Eg, X, c)
        b = {"G": G, "X": X, "F": F.tolist(), "lab": [int(v) for v in lab]}
        _wr(q(mode), b)
    G = b["G"]
    return {"id": ex["id"], "D": D, "C": C, "E": E, "G": G, "Eg": gemb(E, G), "X": b["X"], "F": np.array(b["F"], float).reshape(-1, len(FEATS)), "lab": b["lab"], "nd": a["nd"]}


def out(st, xs, raw=None, pool=None):
    pool = st["G"] if pool is None else pool
    r = []
    for x in xs:
        sp = merge_spans([y for i in x["g"] for y in pool[i]["spans"]])
        r.append({"text": x["text"], "g": [int(i) for i in x["g"]], "spans": [list(y) for y in sp], "rep": x.get("rep", 0), "log": x.get("log", [])})
    return {"id": st["id"], "summary": " ".join(y["text"] for y in r), "sents": r, "raw": raw, "conf": st["X"]}


def choose(st, c, sel=None, ab=(), oracle=None):
    G = st["G"]
    s = np.asarray(oracle, float) if oracle is not None else score_clusters(sel, st["F"])
    ok = None
    if "ss" not in ab and (sel is None or sel.get("ss", True)):
        ok = [g.get("ss", 1.0) >= c["sel"]["ss_floor"] for g in G]
    S = mmr(G, s, st["Eg"], st["X"], c, ok)
    return arrange(S, G, s, st["C"], st["D"], c)


def summarize(ex, c, sel=None, ab=(), cache=None, oracle=None):
    st = prepare(ex, c, "none" if "cluster" in ab else "full", cache)
    S = choose(st, c, sel, ab, oracle(st) if callable(oracle) else oracle)
    t, lab = rewrite(S, st["G"], st["X"], c)
    xs = parse(t, lab)
    if "verify" not in ab:
        xs = verify(xs, st["G"], st["Eg"], st["D"], c)
    return out(st, xs, t)


def summarize_posthoc(ex, c, sel=None, cache=None):
    st = prepare(ex, c, "full", cache)
    G, D = st["G"], st["D"]
    S = choose(st, c, sel)
    t, _ = rewrite(S, G, st["X"], c, marks=False)
    v = M.sup(c, "ver")
    tau = c["ver"]["tau"]
    ss = sents(t)
    xs = []
    if ss and G:
        Q = M.emb(c)(ss)
        for s, q in zip(ss, Q):
            k = [int(i) for i in np.argsort(-(st["Eg"] @ q))[:c["ver"]["topk"]]]
            z = v.score([ev(D, [G[i]]) for i in k], [s] * len(k))
            j = int(np.argmax(z))
            xs.append({"text": s, "g": [k[j]] if z[j] >= tau else []})
    else:
        xs = [{"text": s, "g": []} for s in ss]
    return out(st, xs, t)


def postprocess(o, ex, c, cache=None):
    st = prepare(ex, c, "full", cache)
    D = st["D"]
    pool = list(st["G"])
    xs, nt = [], []
    for y in o["sents"]:
        ids = []
        for sp in y["spans"]:
            pool.append({"claim": span_text(D, [sp]), "spans": [tuple(sp)]})
            nt.append(pool[-1]["claim"])
            ids.append(len(pool) - 1)
        xs.append({"text": y["text"], "g": ids})
    En = M.emb(c)(nt) if nt else None
    Ep = st["Eg"] if En is None else (np.vstack([st["Eg"], En]) if len(st["Eg"]) else En)
    return out(st, verify(xs, pool, Ep, D, c), o.get("raw"), pool)
