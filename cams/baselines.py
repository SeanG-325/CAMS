import re

import numpy as np

from . import models as M
from . import prompts as P
from .textutil import c2t, chunks, sents, toks

CT = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def passages(D, n=120):
    r = []
    for k, d in enumerate(D):
        o = toks(d)
        for a, b in chunks(d, n, 0):
            t = c2t(o, a, b)
            if t:
                r.append((k, t[0], t[1], a, b))
    return r


def e2e(ex, c):
    D = ex["docs"]
    ps = passages(D, c["e2e"]["psg"])
    t = M.llm(c)(P.e2e([D[x[0]][x[3]:x[4]] for x in ps], c["rw"]["words"]), c["llm"]["max_tokens"]["rewrite"])
    u = re.sub(r"([.!?])\s*((?:\[\d+(?:\s*,\s*\d+)*\]\s*)+)", lambda m: " " + m.group(2).strip() + m.group(1) + " ", t or "")
    r = []
    for s in sents(u):
        ids = []
        for m in CT.finditer(s):
            ids += [int(x) - 1 for x in m.group(1).split(",")]
        ids = [i for i in dict.fromkeys(ids) if 0 <= i < len(ps)]
        s = re.sub(r"\s+", " ", CT.sub("", s)).strip()
        s = re.sub(r"\s+([.!?,;:])", r"\1", s)
        if re.search(r"[A-Za-z]", s):
            r.append({"text": s, "g": [], "spans": [list(ps[i]) for i in ids]})
    return {"id": ex["id"], "summary": " ".join(x["text"] for x in r), "sents": r, "raw": t}


def claim_pool(st):
    return [(x["doc"], x["ts"], x["te"], x["cs"], x["ce"]) for x in st["C"]]


def rank_spans(p, st, c, k=None):
    C = st["C"]
    if not C:
        return [], np.zeros(0)
    q = M.emb(c)([p])[0]
    ix = [int(i) for i in np.argsort(-(st["E"] @ q))[:k or c["ver"]["topk"]]]
    D = st["D"]
    z = M.sup(c, "ver").score([D[C[i]["doc"]][C[i]["cs"]:C[i]["ce"]] for i in ix], [p] * len(ix))
    o = np.argsort(-z)
    return [ix[j] for j in o], z[o]


def loc_retrieval(p, st, c):
    if not st["C"]:
        return []
    q = M.emb(c)([p])[0]
    i = int(np.argmax(st["E"] @ q))
    return [claim_pool(st)[i]]


def loc_top2(p, st, c):
    ix, _ = rank_spans(p, st, c)
    if not ix:
        return []
    pl = claim_pool(st)
    r = [pl[ix[0]]]
    for i in ix[1:]:
        if pl[i][0] != r[0][0]:
            r.append(pl[i])
            break
    return r
