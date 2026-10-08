import pickle

import numpy as np

from . import models as M
from .features import FEATS, SS
from .textutil import lcs_f, sents


def cols(ss=True):
    return [i for i, f in enumerate(FEATS) if ss or f not in SS]


def silver(G, ref, c):
    rs = sents(ref)
    n = len(G)
    if not rs or not n:
        return np.zeros(n, int)
    s = c["sel"]["silver"]
    z = M.sup(c, "sel").score([r for r in rs for _ in G], [g["claim"] for _ in rs for g in G]).reshape(len(rs), n).max(0)
    y = (z >= s["tau"]).astype(int)
    bd = np.where((z >= s["lo"]) & (z < s["tau"]))[0]
    if len(bd):
        e = M.emb(c)
        es = (e([G[i]["claim"] for i in bd]) @ e(rs).T).max(1)
        for k, i in enumerate(bd):
            if es[k] >= s["emb"] or max(lcs_f(G[i]["claim"], r) for r in rs) >= s["rouge"]:
                y[i] = 1
    return y


def train_selector(X, y, Xd, yd, c, ss=True):
    import lightgbm as lgb
    k = cols(ss)
    s = c["sel"]
    b = None
    for lr in s["lr"]:
        for ne in s["n_est"]:
            m = lgb.LGBMClassifier(objective="binary", num_leaves=s["leaves"], learning_rate=lr, n_estimators=ne, random_state=s["seed"], verbose=-1)
            m.fit(X[:, k], y, eval_set=[(Xd[:, k], yd)], eval_metric="binary_logloss", callbacks=[lgb.early_stopping(s["es"], verbose=False)])
            v = m.best_score_["valid_0"]["binary_logloss"]
            if b is None or v < b[0]:
                b = (v, m, lr, ne)
    return {"m": b[1], "k": k, "ss": ss, "dev": b[0], "lr": b[2], "ne": b[3]}


def save(sel, p):
    with open(p, "wb") as f:
        pickle.dump(sel, f)


def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def score_clusters(sel, F):
    if not len(F):
        return np.zeros(0)
    return sel["m"].predict_proba(F[:, sel["k"]])[:, 1]


def mmr(G, s, Eg, X, c, ok=None):
    lam, th, pol = c["sel"]["lam"], c["sel"]["theta"], c["sel"]["policy"]
    nb = {i: set() for i in range(len(G))}
    for e in X:
        nb[e["a"]].add(e["b"])
        nb[e["b"]].add(e["a"])
    L = [i for i in range(len(G)) if (ok is None or ok[i]) and not (pol == "defer" and nb[i])]
    S = []
    while L:
        r = [s[i] - lam * (max(float(Eg[i] @ Eg[j]) for j in S) if S else 0.0) for i in L]
        k = int(np.argmax(r))
        if r[k] < th:
            break
        g = L.pop(k)
        S.append(g)
        if pol == "both":
            for h in sorted(nb[g]):
                if h not in S:
                    S.append(h)
                    if h in L:
                        L.remove(h)
    return S
