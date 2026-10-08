import json
import re


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(p, xs):
    with open(p, "w") as f:
        for x in xs:
            f.write(json.dumps(x) + "\n")


def _mn(x, i, sp):
    ds = [re.sub(r"[ \t]+", " ", d.replace("NEWLINE_CHAR", "\n")).strip() for d in x["document"].split("|||||")]
    s = re.sub(r"^[–\-\s]+", "", x["summary"]).strip()
    return {"id": x.get("id", "%s-%d" % (sp, i)), "docs": [d for d in ds if d], "summary": s}


def load(p, n=None, sp="x"):
    if p.endswith(".jsonl"):
        xs = read_jsonl(p)
    else:
        import datasets
        a, b = p.split(":", 1) if ":" in p else (p, "test")
        sp = b
        xs = list(datasets.load_dataset(a, split=b))
    r = []
    for i, x in enumerate(xs[:n] if n else xs):
        if "document" in x and "docs" not in x:
            r.append(_mn(x, i, sp))
        else:
            x = dict(x)
            x.setdefault("id", "%s-%d" % (sp, i))
            x.setdefault("summary", "")
            r.append(x)
    return r
