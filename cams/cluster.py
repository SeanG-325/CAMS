import numpy as np

from . import models as M


def merge_spans(sp):
    r = []
    for x in sorted(tuple(int(v) for v in y) for y in sp):
        if r and r[-1][0] == x[0] and x[1] <= r[-1][2]:
            p = r[-1]
            r[-1] = (p[0], p[1], max(p[2], x[2]), p[3], max(p[4], x[4]))
        else:
            r.append(x)
    return r


def spans_of(C, ix):
    return merge_spans([(C[i]["doc"], C[i]["ts"], C[i]["te"], C[i]["cs"], C[i]["ce"]) for i in ix])


def _med(E, ix):
    if len(ix) == 1:
        return ix[0]
    return ix[int((E[ix] @ E[ix].T).sum(1).argmax())]


def _same(C, E, a, b, c, mm):
    x, y = _med(E, a), _med(E, b)
    k = (min(x, y), max(x, y))
    if k not in mm:
        p = M.nli3(c).ent([C[x]["claim"], C[y]["claim"]], [C[y]["claim"], C[x]["claim"]])
        mm[k] = bool(min(p) >= c["clu"]["ent"])
    return mm[k]


def gemb(E, G):
    if not G:
        return np.zeros((0, E.shape[1] if E.ndim == 2 else 1))
    V = np.stack([E[g["m"]].mean(0) for g in G])
    return V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)


def cluster_claims(C, c, E=None, mode="full"):
    n = len(C)
    if E is None:
        E = M.emb(c)([x["claim"] for x in C])
    G = [[i] for i in range(n)]
    if n > 1 and mode != "none":
        V = (E @ E.T).astype(float)
        np.fill_diagonal(V, -np.inf)
        B = np.zeros((n, n), bool)
        on = np.ones(n, bool)
        w = np.ones(n)
        th = c["clu"]["sim"]
        mm = {}
        while True:
            A = np.where(B | ~on[:, None] | ~on[None, :], -np.inf, V)
            i, j = np.unravel_index(int(np.argmax(A)), A.shape)
            if not A[i, j] >= th:
                break
            if mode == "full" and not _same(C, E, G[i], G[j], c, mm):
                B[i, j] = B[j, i] = True
                continue
            V[i] = (w[i] * V[i] + w[j] * V[j]) / (w[i] + w[j])
            V[:, i] = V[i]
            V[i, i] = -np.inf
            G[i] = G[i] + G[j]
            G[j] = []
            w[i] += w[j]
            on[j] = False
            B[i, :] = False
            B[:, i] = False
    G = [sorted(g) for g in G if g]
    lab = np.zeros(n, int)
    R = []
    for i, g in enumerate(G):
        r = _med(E, g)
        for j in g:
            lab[j] = i
        R.append({"id": i, "m": [int(j) for j in g], "rep": int(r), "claim": C[r]["claim"], "spans": spans_of(C, g), "docs": sorted({int(C[j]["doc"]) for j in g})})
    return R, gemb(E, R), lab
