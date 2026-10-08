import numpy as np

from .. import models as M
from ..textutil import sents
from ..verify import facts


def alignscore(docs, sums, c):
    from alignscore import AlignScore
    m = AlignScore(model="roberta-large", batch_size=16, device=M.dev(), ckpt_path=c["faith"]["align_ckpt"], evaluation_mode="nli_sp")
    return list(m.score(contexts=["\n".join(d) for d in docs], claims=[s or "." for s in sums]))


def summac(docs, sums, c):
    from summac.model_summac import SummaCConv
    m = SummaCConv(models=["vitc"], bins="percentile", granularity="sentence", nli_labels="e", device=M.dev(), start_file="default", agg="mean")
    return list(m.score(["\n".join(d) for d in docs], [s or "." for s in sums])["scores"])


def factscore(docs, sums, c):
    f = M.sup(c, c["faith"]["fs_judge"])
    r = []
    for D, s in zip(docs, sums):
        fs = [x for t in sents(s) for x in facts(t, c)]
        if not fs:
            r.append(0.0)
            continue
        z = f.score(["\n".join(D)] * len(fs), fs)
        r.append(float(np.mean(z >= c["eval"]["tau_nli"])))
    return r
