import os
import time

import numpy as np

from .textutil import wins

R = {}


def dev():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class LLM:
    def __init__(s, c):
        s.c = c
        if c["backend"] == "anthropic":
            import anthropic
            s.k = anthropic.Anthropic()
        else:
            import openai
            s.k = openai.OpenAI(base_url=c.get("base_url"), api_key=os.environ.get("OPENAI_API_KEY", "none"))

    def __call__(s, p, n=1024):
        c = s.c
        nr = c.get("retries", 3)
        for i in range(nr):
            try:
                if c["backend"] == "anthropic":
                    a = dict(model=c["model"], max_tokens=n, temperature=c["temperature"], messages=[{"role": "user", "content": p}])
                    if c.get("top_p", 1.0) != 1.0:
                        a["top_p"] = c["top_p"]
                    r = s.k.messages.create(**a)
                    return "".join(getattr(b, "text", "") for b in r.content)
                r = s.k.chat.completions.create(model=c["model"], messages=[{"role": "user", "content": p}], max_tokens=n, temperature=c["temperature"], top_p=c.get("top_p", 1.0))
                return r.choices[0].message.content or ""
            except Exception:
                if i == nr - 1:
                    raise
                time.sleep(5 * 2 ** i)


class Emb:
    def __init__(s, name, bs=64):
        from sentence_transformers import SentenceTransformer
        s.m = SentenceTransformer(name, device=dev())
        s.bs = bs

    def __call__(s, xs):
        xs = list(xs)
        if not xs:
            return np.zeros((0, s.m.get_sentence_embedding_dimension()))
        return s.m.encode(xs, batch_size=s.bs, normalize_embeddings=True, convert_to_numpy=True)


class NLI3:
    def __init__(s, name, bs=32, ml=512, lab=None):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        s.t = AutoTokenizer.from_pretrained(name)
        s.m = AutoModelForSequenceClassification.from_pretrained(name).to(dev()).eval()
        L = {int(k): str(v).lower() for k, v in s.m.config.id2label.items()}
        lab = lab or {}
        s.e = lab.get("ent", next((k for k, v in L.items() if v.startswith(("entail", "support"))), 0))
        s.x = lab.get("con", next((k for k, v in L.items() if v.startswith(("contra", "refut"))), len(L) - 1))
        s.bs, s.ml = bs, ml

    def probs(s, ps, hs):
        import torch
        o = []
        for i in range(0, len(ps), s.bs):
            z = s.t(list(ps[i:i + s.bs]), list(hs[i:i + s.bs]), truncation=True, max_length=s.ml, padding=True, return_tensors="pt").to(s.m.device)
            with torch.no_grad():
                o.append(torch.softmax(s.m(**z).logits.float(), -1).cpu().numpy())
        return np.concatenate(o) if o else np.zeros((0, max(s.e, s.x) + 1))

    def ent(s, ps, hs):
        return s.probs(ps, hs)[:, s.e]

    def con(s, ps, hs):
        return s.probs(ps, hs)[:, s.x]


class Sup:
    def __init__(s, name, bs=8, ml=512, win=300):
        from transformers import AutoConfig
        s.ed = AutoConfig.from_pretrained(name).is_encoder_decoder
        s.bs, s.ml, s.win = bs, ml, win
        if s.ed:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            s.t = AutoTokenizer.from_pretrained(name)
            s.m = AutoModelForSeq2SeqLM.from_pretrained(name, torch_dtype="auto").to(dev()).eval()
            s.i1 = s.t("1", add_special_tokens=False).input_ids[-1]
            s.i0 = s.t("0", add_special_tokens=False).input_ids[-1]
        else:
            s.n = NLI3(name, bs, ml)

    def raw(s, ps, hs):
        if not len(ps):
            return np.zeros(0)
        if not s.ed:
            return s.n.ent(ps, hs)
        import torch
        o = []
        for i in range(0, len(ps), s.bs):
            x = ["premise: " + p + " hypothesis: " + h for p, h in zip(ps[i:i + s.bs], hs[i:i + s.bs])]
            z = s.t(x, truncation=True, max_length=s.ml, padding=True, return_tensors="pt").to(s.m.device)
            d = torch.full((len(x), 1), s.m.config.decoder_start_token_id, dtype=torch.long, device=s.m.device)
            with torch.no_grad():
                l = s.m(**z, decoder_input_ids=d).logits[:, 0, [s.i1, s.i0]].float()
            o.append(torch.softmax(l, -1)[:, 0].cpu().numpy())
        return np.concatenate(o)

    def score(s, ps, hs):
        P, H, ix = [], [], []
        for j, (p, h) in enumerate(zip(ps, hs)):
            for w in wins(p, s.win):
                P.append(w)
                H.append(h)
                ix.append(j)
        z = s.raw(P, H)
        r = np.zeros(len(ps))
        for j, v in zip(ix, z):
            r[j] = max(r[j], float(v))
        return r


def llm(c):
    if "llm" not in R:
        R["llm"] = LLM(c["llm"])
    return R["llm"]


def emb(c):
    if "emb" not in R:
        R["emb"] = Emb(c["emb"]["model"], c["emb"].get("batch", 64))
    return R["emb"]


def nli3(c):
    if "nli3" not in R:
        R["nli3"] = NLI3(c["nli"]["three"], c["nli"].get("bs", 32), lab=c["nli"].get("labels"))
    return R["nli3"]


def sup(c, w):
    k = "sup_" + w
    if k not in R:
        R[k] = Sup(c["nli"][w], c["nli"].get("bs_sup", 8), win=c["nli"].get("win", 300))
    return R[k]
