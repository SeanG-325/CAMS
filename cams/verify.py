import numpy as np

from . import models as M
from . import prompts as P
from .extract import pjson
from .features import span_text
from .rewrite import parse


def ev(D, us):
    return span_text(D, {tuple(x) for u in us for x in u["spans"]})


def facts(t, c):
    x = pjson(M.llm(c)(P.facts(t), c["llm"]["max_tokens"]["facts"]))
    x = [str(f).strip() for f in x if str(f).strip()] if isinstance(x, list) else []
    return x or [t]


def check(t, us, D, c):
    tau = c["ver"]["tau"]
    p = ev(D, us)
    fs = facts(t, c)
    z = M.sup(c, "ver").score([p] * (len(fs) + 1), [t] + fs)
    return bool(z[0] >= tau), [f for f, y in zip(fs, z[1:]) if y < tau], [f for f, y in zip(fs, z[1:]) if y >= tau]


def reretrieve(t, need, good, cur, pool, Ep, D, c):
    if not len(pool) or not len(Ep):
        return None
    v = M.sup(c, "ver")
    tau = c["ver"]["tau"]
    q = M.emb(c)([t])[0]
    k = [int(i) for i in np.argsort(-(Ep @ q))[:c["ver"]["topk"]]]
    cd = list(dict.fromkeys(k + list(cur)))
    T = [ev(D, [pool[i]]) for i in cd]
    nw = []
    for f in need:
        z = v.score(T, [f] * len(T))
        j = int(np.argmax(z))
        if z[j] < tau:
            return None
        nw.append(cd[j])
    kp = []
    for i in cur:
        if good and (v.score([ev(D, [pool[i]])] * len(good), good) >= tau).any():
            kp.append(i)
    return list(dict.fromkeys(kp + nw))


def regen(x, bad, pool, c):
    lb = {"g%d" % (j + 1): i for j, i in enumerate(x["g"])}
    mk = "[" + ", ".join(lb) + "]"
    y = (M.llm(c)(P.regen([pool[i]["claim"] for i in x["g"]], x["text"], bad, mk), c["llm"]["max_tokens"]["rewrite"]) or "").strip()
    if not y or y.upper().startswith("NONE"):
        return []
    return parse(y, lb)


def verify(xs, pool, Ep, D, c, R=None):
    R = c["ver"]["R"] if R is None else R
    o = []
    for x in xs:
        x = dict(x, g=list(x["g"]), rep=0, log=[])
        for r in range(R + 1):
            a, bad, good = check(x["text"], [pool[i] for i in x["g"]], D, c)
            if a and not bad:
                o.append(x)
                break
            if r == R:
                break
            x["rep"] = r + 1
            ng = reretrieve(x["text"], bad or [x["text"]], good, x["g"], pool, Ep, D, c)
            if ng is not None and set(ng) != set(x["g"]):
                x["log"].append(["cite", list(x["g"]), ng])
                x["g"] = ng
                continue
            ys = regen(x, bad or [x["text"]], pool, c)
            if not ys:
                break
            x["log"].append(["regen", x["text"], ys[0]["text"]])
            x["text"], x["g"] = ys[0]["text"], ys[0]["g"]
    return o
