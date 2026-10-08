import json
import re

from . import models as M
from . import prompts as P
from .locate import find_span
from .textutil import chunks, toks


def pjson(t):
    t = re.sub(r"```(?:json)?", "", t or "").strip()
    for a, b in ((0, len(t)), (t.find("["), t.rfind("]") + 1)):
        if a < 0 or b <= a:
            continue
        try:
            return json.loads(t[a:b])
        except Exception:
            pass
    return None


def extract_doc(d, k, c):
    f = M.llm(c)
    o = toks(d)
    cs = chunks(d, c["pre"]["chunk"], c["pre"]["overlap"])
    hd = next((l.strip() for l in d.split("\n") if l.strip()), "")
    r, z, nd = [], set(), 0
    for a, b in cs:
        x = None
        for _ in range(2):
            x = pjson(f(P.extract(d[a:b], "d%d" % (k + 1), hd if a > 0 else ""), c["llm"]["max_tokens"]["extract"]))
            if isinstance(x, list):
                break
        if not isinstance(x, list):
            continue
        for e in x:
            if not isinstance(e, dict):
                continue
            t = str(e.get("claim") or "").strip()
            q = str(e.get("quote") or "").strip()
            if not t or not q:
                continue
            s = find_span(d, q, c["pre"]["rho"], o, a, b)
            if s is None:
                nd += 1
                continue
            u = (s[0], s[1], t.lower())
            if u in z:
                continue
            z.add(u)
            r.append({"claim": t, "quote": q, "doc": k, "cs": s[0], "ce": s[1], "ts": s[2], "te": s[3], "how": s[4]})
    return r, nd


def extract_all(D, c):
    C, nd = [], 0
    for k, d in enumerate(D):
        r, n = extract_doc(d, k, c)
        C += r
        nd += n
    for i, x in enumerate(C):
        x["id"] = i
    return C, nd
