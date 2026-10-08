from .. import models as M
from .. import prompts as P
from ..extract import pjson
from ..textutil import sents
from ..verify import facts


def rouge(refs, hyps):
    from rouge_score import rouge_scorer
    s = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True)
    r = []
    for a, b in zip(refs, hyps):
        x = s.score("\n".join(sents(a)), "\n".join(sents(b)))
        r.append({k: v.fmeasure for k, v in x.items()})
    return r


def bertscore(refs, hyps):
    from bert_score import score
    _, _, f = score([h or "." for h in hyps], refs, lang="en", verbose=False)
    return f.tolist()


def coverage(ref, hyp, c, fs=None):
    fs = fs or [x for t in sents(ref) for x in facts(t, c)]
    if not fs:
        return 0.0, fs
    x = pjson(M.llm(c)(P.cover(fs, hyp or ""), c["llm"]["max_tokens"]["judge"]))
    k = {int(i) for i in x if str(i).strip().isdigit() and 1 <= int(i) <= len(fs)} if isinstance(x, list) else set()
    return len(k) / len(fs), fs
