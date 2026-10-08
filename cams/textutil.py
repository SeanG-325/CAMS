import bisect
import re

W = re.compile(r"\w+|[^\w\s]")
SP = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"')\]]))\s+(?=[A-Z0-9\"'(\[])")
MON = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
YR = re.compile(r"\b(1[89]\d\d|20\d\d)\b")
MD = re.compile(r"\b(" + "|".join(MON) + r")\.?(?:\s+(\d{1,2})\b)?")


def toks(t):
    return [(m.start(), m.end()) for m in W.finditer(t)]


def ntok(t):
    return len(W.findall(t))


def c2t(o, a, b):
    s = [x for x, _ in o]
    e = [y for _, y in o]
    i = bisect.bisect_right(e, a)
    j = bisect.bisect_left(s, b)
    if i >= j:
        return None
    return i, j


def sents(t):
    return [x.strip() for x in SP.split((t or "").strip()) if x.strip()]


def norm(t):
    s, m = [], []
    for i, ch in enumerate(t):
        if ch.isalnum():
            l = ch.lower()
            s.append(l if len(l) == 1 else ch)
            m.append(i)
        elif s and s[-1] != " ":
            s.append(" ")
            m.append(i)
    if s and s[-1] == " ":
        s.pop()
        m.pop()
    return "".join(s), m


def chunks(t, n=512, ov=64):
    ps = []
    for m in re.finditer(r"[^\n]+", t):
        if not m.group().strip():
            continue
        k = ntok(m.group())
        if k <= n:
            ps.append((m.start(), m.end(), k))
            continue
        o = [(m.start() + x, m.start() + y) for x, y in toks(m.group())]
        i = 0
        while i < len(o):
            j = min(i + n, len(o))
            ps.append((o[i][0], o[j - 1][1], j - i))
            if j == len(o):
                break
            i = max(j - ov, i + 1)
    r, cur, c = [], [], 0
    for p in ps:
        if cur and c + p[2] > n:
            r.append((cur[0][0], cur[-1][1]))
            kp, s = [], 0
            for q in reversed(cur):
                if s + q[2] > ov:
                    break
                kp.insert(0, q)
                s += q[2]
            cur, c = kp, s
        cur.append(p)
        c += p[2]
    if cur:
        r.append((cur[0][0], cur[-1][1]))
    return r


def wins(p, n):
    if len(p.split()) <= n:
        return [p]
    ss = sents(p)
    r, i = [], 0
    while i < len(ss):
        j, k = i, 0
        while j < len(ss) and (k == 0 or k + len(ss[j].split()) <= n):
            k += len(ss[j].split())
            j += 1
        r.append(" ".join(ss[i:j]))
        if j >= len(ss):
            break
        i = max(i + 1, (i + j) // 2)
    return r


def lcs_f(a, b):
    x, y = W.findall(a.lower()), W.findall(b.lower())
    if not x or not y:
        return 0.0
    p = [0] * (len(y) + 1)
    for u in x:
        q = [0]
        for j, v in enumerate(y):
            q.append(p[j] + 1 if u == v else max(p[j + 1], q[-1]))
        p = q
    l = p[-1]
    if not l:
        return 0.0
    a1, a2 = l / len(x), l / len(y)
    return 2 * a1 * a2 / (a1 + a2)


def date(t):
    y = YR.search(t)
    m = MD.search(t)
    if not y and not m:
        return None
    return (int(y.group(1)) if y else None, MON.index(m.group(1)) + 1 if m else None, int(m.group(2)) if m and m.group(2) else None)
