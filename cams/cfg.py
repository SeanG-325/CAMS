import os
import yaml


def _up(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _up(a[k], v)
        else:
            a[k] = v
    return a


def load(p, ov=()):
    with open(p) as f:
        c = yaml.safe_load(f) or {}
    if "base" in c:
        b = load(os.path.join(os.path.dirname(p), c.pop("base")))
        c = _up(b, c)
    for s in ov or ():
        k, v = s.split("=", 1)
        a = c
        ks = k.split(".")
        for x in ks[:-1]:
            a = a.setdefault(x, {})
        a[ks[-1]] = yaml.safe_load(v)
    return c
