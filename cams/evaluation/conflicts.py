import numpy as np


def rates(rows):
    a = [r for r in rows if r.get("kind") == "divergent"]
    b = [r for r in rows if r.get("kind") == "conflict"]
    sf = [r for r in b if r["surfaced"]]
    return {"divergent": np.mean([r["covered"] for r in a]) if a else 0.0, "surfaced": len(sf) / len(b) if b else 0.0, "attrib": np.mean([r["attrib_ok"] for r in sf]) if sf else 0.0}


def cohen(x, y):
    x, y = list(x), list(y)
    L = sorted(set(x) | set(y))
    n = len(x)
    po = sum(a == b for a, b in zip(x, y)) / n
    pe = sum((x.count(l) / n) * (y.count(l) / n) for l in L)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def fleiss(Z):
    Z = [list(z) for z in Z]
    L = sorted({v for z in Z for v in z})
    n = len(Z[0])
    N = len(Z)
    T = np.array([[z.count(l) for l in L] for z in Z], float)
    p = T.sum(0) / (N * n)
    Pi = ((T ** 2).sum(1) - n) / (n * (n - 1))
    pb, pe = Pi.mean(), (p ** 2).sum()
    return (pb - pe) / (1 - pe) if pe < 1 else 1.0


def timing(rows, cap=120):
    r = {}
    for x in rows:
        if x["sec"] <= cap:
            r.setdefault(x["sys"], []).append(x["sec"])
    return {k: (float(np.mean(v)), float(np.median(v)), len(v)) for k, v in r.items()}
