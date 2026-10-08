import numpy as np

from .. import models as M
from ..baselines import loc_retrieval, loc_top2
from ..cluster import merge_spans
from ..features import span_text
from ..pipeline import prepare
from ..textutil import c2t, toks


def gold_tok(ex):
    O = [toks(d) for d in ex["docs"]]
    r = []
    for p in ex["props"]:
        g = []
        for k, a, b in p["gold"]:
            t = c2t(O[k], a, b)
            if t:
                g.append((k, t[0], t[1]))
        r.append(g)
    return r


def loc_cams(p, st, c):
    G = st["G"]
    if not G:
        return []
    q = M.emb(c)([p])[0]
    k = [int(i) for i in np.argsort(-(st["Eg"] @ q))[:c["ver"]["topk"]]]
    z = M.sup(c, "ver").score([span_text(st["D"], G[i]["spans"]) for i in k], [p] * len(k))
    return merge_spans(G[k[int(np.argmax(z))]]["spans"])


def localize(ex, c, sysn, cache=None):
    if sysn == "cams":
        st, f = prepare(ex, c, "full", cache), loc_cams
    elif sysn == "nocluster":
        st, f = prepare(ex, c, "none", cache), loc_cams
    elif sysn == "retrieval":
        st, f = prepare(ex, c, "full", cache), loc_retrieval
    elif sysn == "top2":
        st, f = prepare(ex, c, "full", cache), loc_top2
    else:
        raise ValueError(sysn)
    return {"id": ex["id"], "pred": [[list(y[:3]) for y in f(p["text"], st, c)] for p in ex["props"]]}


def iou(a, b):
    if a[0] != b[0]:
        return 0.0
    i = max(0, min(a[2], b[2]) - max(a[1], b[1]))
    u = max(a[2], b[2]) - min(a[1], b[1])
    return i / u if u > 0 else 0.0


def span_counts(P, G, tau):
    return [sum(any(iou(p, g) >= tau for g in G) for p in P), len(P), sum(any(iou(p, g) >= tau for p in P) for g in G), len(G)]


def src_counts(P, G, tau):
    dg = {g[0] for g in G}
    dp = {p[0] for p in P}
    h = {d for d in dp & dg if any(iou(p, g) >= tau for p in P if p[0] == d for g in G if g[0] == d)}
    return [len(h), len(dp), len(h), len(dg)]


def score_topic(pred, gold, tau):
    s = [0, 0, 0, 0]
    m = [0, 0, 0, 0]
    for P, G in zip(pred, gold):
        P = [tuple(x) for x in P]
        G = [tuple(x) for x in G]
        s = [a + b for a, b in zip(s, span_counts(P, G, tau))]
        if len({g[0] for g in G}) >= 2:
            m = [a + b for a, b in zip(m, src_counts(P, G, tau))]
    return {"span": s, "multi": m}


def prf(x):
    p = x[0] / x[1] if x[1] else 0.0
    r = x[2] / x[3] if x[3] else 0.0
    return p, r, 2 * p * r / (p + r) if p + r else 0.0


def total(rows, k):
    return prf([sum(r[k][i] for r in rows) for i in range(4)])
