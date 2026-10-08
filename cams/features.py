import numpy as np

from . import models as M
from .conflict import ents

FEATS = ["size", "ndoc", "fdoc", "K", "pos", "posm", "len", "nent", "cent", "nconf", "ss_rep", "ss_mean", "ss_min"]
SS = ["ss_rep", "ss_mean", "ss_min"]


def span_text(D, sp):
    return " ".join(D[x[0]][x[3]:x[4]] for x in sorted(tuple(y) for y in sp))


def self_support(D, C, G, c):
    f = M.sup(c, "sel")
    z = f.score([D[x["doc"]][x["cs"]:x["ce"]] for x in C], [x["claim"] for x in C]) if C else np.zeros(0)
    g = f.score([span_text(D, x["spans"]) for x in G], [x["claim"] for x in G]) if G else np.zeros(0)
    return z, g


def featurize(D, C, G, Eg, X, c):
    K = len(D)
    L = [max(len(d), 1) for d in D]
    z, zg = self_support(D, C, G, c)
    nc = np.zeros(len(G))
    for e in X:
        nc[e["a"]] += 1
        nc[e["b"]] += 1
    S = Eg @ Eg.T if len(G) else np.zeros((0, 0))
    F = []
    for i, g in enumerate(G):
        ps = [C[j]["cs"] / L[C[j]["doc"]] for j in g["m"]]
        ss = [z[j] for j in g["m"]]
        ct = (S[i].sum() - S[i, i]) / max(len(G) - 1, 1)
        g["ss"] = float(zg[i])
        F.append([len(g["m"]), len(g["docs"]), len(g["docs"]) / K, K, min(ps), float(np.mean(ps)), len(g["claim"].split()), len(ents(g["claim"])), ct, nc[i], zg[i], float(np.mean(ss)), float(min(ss))])
    return np.array(F, dtype=float).reshape(-1, len(FEATS))
