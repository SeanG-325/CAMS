from rapidfuzz import fuzz

from .textutil import c2t, norm, toks


def find_span(d, q, rho=0.85, o=None, lo=0, hi=None):
    o = toks(d) if o is None else o
    hi = len(d) if hi is None else hi
    q = (q or "").strip()
    if not q:
        return None
    rs = list(dict.fromkeys([(lo, hi), (0, len(d))]))
    for a, b in rs:
        i = d.find(q, a, b)
        if i >= 0:
            t = c2t(o, i, i + len(q))
            return (i, i + len(q)) + t + ("exact",) if t else None
    nq, _ = norm(q)
    if not nq:
        return None
    for a, b in rs:
        nd, m = norm(d[a:b])
        if not nd:
            continue
        al = fuzz.partial_ratio_alignment(nq, nd)
        if al is None or al.score / 100.0 < rho:
            continue
        x, y = al.dest_start, al.dest_end
        while x < y and nd[x] == " ":
            x += 1
        while y > x and nd[y - 1] == " ":
            y -= 1
        if x >= y:
            continue
        i, j = a + m[x], a + m[y - 1] + 1
        t = c2t(o, i, j)
        if t:
            return (i, j) + t + ("fuzzy",)
    return None
