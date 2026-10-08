import json


def export(states, p):
    n = 0
    with open(p, "w") as f:
        for st in states:
            for x in st["C"]:
                f.write(json.dumps({"claim": x["claim"], "doc_id": "%s/d%d" % (st["id"], x["doc"] + 1), "char_start": x["cs"], "char_end": x["ce"], "tok_start": x["ts"], "tok_end": x["te"], "cluster_id": "%s/g%d" % (st["id"], st["lab"][x["id"]] + 1), "quote": ""}) + "\n")
                n += 1
    return n


def fill(p, corpus, q):
    D = {str(x["id"]): x["docs"] for x in corpus}
    m = 0
    with open(p) as f, open(q, "w") as g:
        for l in f:
            if not l.strip():
                continue
            r = json.loads(l)
            a, b = r["doc_id"].rsplit("/", 1)
            if a in D:
                r["quote"] = D[a][int(b[1:]) - 1][r["char_start"]:r["char_end"]]
            else:
                m += 1
            g.write(json.dumps(r) + "\n")
    return m
