import numpy as np


def stat(x, ratio):
    if ratio:
        return x[..., 0].sum(-1) / np.maximum(x[..., 1].sum(-1), 1e-12)
    return x.mean(-1)


def boot(a, b, B=10000, seed=0, ratio=False, sc=1.0, bs=500):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    g = np.random.default_rng(seed)
    d0 = (stat(b, ratio) - stat(a, ratio)) * sc
    ds = []
    for i in range(0, B, bs):
        ix = g.integers(0, n, (min(bs, B - i), n))
        ds.append((stat(b[ix], ratio) - stat(a[ix], ratio)) * sc)
    ds = np.concatenate(ds)
    lo, hi = np.percentile(ds, [2.5, 97.5])
    p = min(1.0, 2 * min((np.sum(ds <= 0) + 1) / (B + 1), (np.sum(ds >= 0) + 1) / (B + 1)))
    se = (hi - lo) / 2 / 1.96
    return {"a": float(stat(a, ratio) * sc), "b": float(stat(b, ratio) * sc), "d": float(d0), "lo": float(lo), "hi": float(hi), "p": float(p), "se": float(se), "z": float(d0 / se) if se > 0 else (0.0 if d0 == 0 else float("inf"))}


def holm(ps, alpha=0.05, m=None):
    m = max(m or 0, len(ps))
    r = [False] * len(ps)
    for i, j in enumerate(np.argsort(ps)):
        if ps[j] <= alpha / (m - i):
            r[j] = True
        else:
            break
    return r


def dagger(d, ok, bounded=False, e=5.0, eb=0.05):
    return bool(ok and abs(d) >= (eb if bounded else e) - 1e-9)


def table(cmps, B=10000, m=24, alpha=0.05, seed=0):
    r = [dict(x, **boot(x["A"], x["Bv"], B, seed, x.get("ratio", False), x.get("sc", 1.0))) for x in cmps]
    h = holm([x["p"] for x in r], alpha, m)
    for x, k in zip(r, h):
        x["holm"] = k
        x["dagger"] = dagger(x["d"], k, x.get("bounded", False))
        x.pop("A")
        x.pop("Bv")
    return r
