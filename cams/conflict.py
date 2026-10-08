import re

from . import models as M

_N = None
NUM = re.compile(r"\d+(?:[.,]\d+)*")
CAP = re.compile(r"\b[A-Z][\w\-]*(?:\s+[A-Z][\w\-]*)*")
STOP = {"The", "A", "An", "In", "On", "At", "He", "She", "It", "They", "This", "That", "These", "Those", "But", "And", "Its", "His", "Her", "Their", "According", "After", "Before", "When", "While"}


def ents(t):
    global _N
    if _N is None:
        try:
            import spacy
            _N = spacy.load("en_core_web_sm")
        except Exception:
            _N = False
    if _N:
        return {e.text.lower() for e in _N(t).ents}
    r = {x.replace(",", "") for x in NUM.findall(t)}
    r |= {m.group().lower() for m in CAP.finditer(t) if m.group() not in STOP}
    return r


def find_conflicts(C, E, lab, c):
    n = len(C)
    if n < 2:
        return []
    S = E @ E.T
    X = [ents(x["claim"]) for x in C]
    q = []
    for a in range(n):
        for b in range(a + 1, n):
            if lab[a] == lab[b] or C[a]["doc"] == C[b]["doc"]:
                continue
            if S[a, b] >= c["conf"]["cand_sim"] or X[a] & X[b]:
                q.append((float(S[a, b]), a, b))
    q.sort(reverse=True)
    q = q[:c["conf"]["max_pairs"]]
    if not q:
        return []
    P = [C[a]["claim"] for _, a, b in q] + [C[b]["claim"] for _, a, b in q]
    H = [C[b]["claim"] for _, a, b in q] + [C[a]["claim"] for _, a, b in q]
    z = M.nli3(c).con(P, H)
    m = len(q)
    r = {}
    for i, (_, a, b) in enumerate(q):
        p = float(max(z[i], z[i + m]))
        if p >= c["conf"]["p"]:
            k = (int(min(lab[a], lab[b])), int(max(lab[a], lab[b])))
            if p > r.get(k, (0.0,))[0]:
                r[k] = (p, a, b)
    return [{"a": k[0], "b": k[1], "p": v[0], "pair": [int(v[1]), int(v[2])]} for k, v in sorted(r.items())]
