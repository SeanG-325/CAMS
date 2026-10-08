from .. import models as M
from ..features import span_text


def cite_eval(o, D, c, w="eval"):
    f = M.sup(c, w)
    tau = c["eval"]["tau_nli"]
    rc, pc, pn, tk, ns = [], 0, 0, [], []
    for y in o["sents"]:
        sp = [tuple(x) for x in y["spans"]]
        t = y["text"]
        ns.append(len(sp))
        tk.append(sum(x[2] - x[1] for x in sp))
        if not sp:
            rc.append(0)
            continue
        j = f.score([span_text(D, sp)], [t])[0] >= tau
        rc.append(int(j))
        pn += len(sp)
        if not j:
            continue
        if len(sp) == 1:
            pc += 1
            continue
        a = f.score([span_text(D, [x]) for x in sp], [t] * len(sp))
        for i in range(len(sp)):
            if a[i] >= tau or f.score([span_text(D, sp[:i] + sp[i + 1:])], [t])[0] < tau:
                pc += 1
    return {"id": o["id"], "rec": [sum(rc), len(rc)], "prec": [pc, pn], "tok": [sum(tk), len(tk)], "span": [sum(ns), len(ns)], "uncited": [sum(1 for x in ns if not x), len(ns)]}


def agg(rows, k, macro=True):
    if macro:
        v = [r[k][0] / r[k][1] for r in rows if r[k][1]]
        return sum(v) / len(v) if v else 0.0
    a = sum(r[k][0] for r in rows)
    b = sum(r[k][1] for r in rows)
    return a / b if b else 0.0
