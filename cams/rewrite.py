import re

from . import models as M
from . import prompts as P
from .textutil import date

MK = re.compile(r"\[\s*(g\d+(?:\s*,\s*g\d+)*)\s*\]")


def arrange(S, G, s, C, D, c):
    if c["rw"]["order"] != "time":
        return sorted(S, key=lambda g: -s[g])
    p = {g: min(C[j]["cs"] / max(len(D[C[j]["doc"]]), 1) for j in G[g]["m"]) for g in S}
    xs = sorted(S, key=lambda g: p[g])
    k, ks = None, []
    for g in xs:
        d = date(G[g]["claim"])
        if d:
            k = (d[0] or (k[0] if k else 0), d[1] or 0, d[2] or 0)
        ks.append(k)
    f = next((x for x in ks if x), (0, 0, 0))
    ks = [x or f for x in ks]
    return [g for _, g in sorted(zip(ks, xs), key=lambda z: z[0])]


def rewrite(S, G, X, c, marks=True):
    lab = {"g%d" % (i + 1): g for i, g in enumerate(S)}
    inv = {g: l for l, g in lab.items()}
    ix = {g: i for i, g in enumerate(S)}
    X = [e for e in X if e["a"] in inv and e["b"] in inv]
    if marks:
        p = P.rewrite([(l, G[g]["claim"]) for l, g in lab.items()], [(inv[e["a"]], inv[e["b"]]) for e in X], c["rw"]["words"])
    else:
        p = P.rewrite_plain([G[g]["claim"] for g in S], [(ix[e["a"]], ix[e["b"]]) for e in X], c["rw"]["words"])
    return M.llm(c)(p, c["llm"]["max_tokens"]["rewrite"]), lab


def parse(t, lab):
    t = t or ""
    r, p = [], 0
    for m in MK.finditer(t):
        s = re.sub(r"\s+", " ", t[p:m.start()]).strip().lstrip(".;:,-*• ").strip()
        p = m.end()
        g = list(dict.fromkeys(lab[x.strip()] for x in m.group(1).split(",") if x.strip() in lab))
        if not g or not re.search(r"[A-Za-z]", s):
            continue
        if s[-1] not in ".!?\"'":
            s += "."
        r.append({"text": s, "g": g})
    return r
