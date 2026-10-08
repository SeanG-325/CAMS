import json

EXT_HEAD = """You decompose a news document into atomic claims.

Rules:
- Each claim states exactly one fact.
- Each claim must stand alone. Resolve pronouns and bridging references, and restore elided subjects, times and locations from the document.
- For each claim, copy a quotation from the document that licenses it. The quotation must be an exact contiguous substring of the text, as short as possible while still supporting the claim. Never paraphrase or shorten words inside the quotation.
- Never report character or token positions.
- Cover every factual statement in the text. Skip bylines, photo credits, links and other boilerplate.
- Output a JSON array only. Each element has the keys "claim", "quote" and "doc_id".
"""

EXT_SHOTS = [
    ("", "d1", "Officials in Lagos said Tuesday that the flood had displaced about 4,000 residents. They expect the water to recede by the weekend.",
     [{"claim": "Officials in Lagos said on Tuesday that a flood had displaced about 4,000 residents.", "quote": "Officials in Lagos said Tuesday that the flood had displaced about 4,000 residents", "doc_id": "d1"},
      {"claim": "Officials in Lagos expect the flood water to recede by the weekend.", "quote": "They expect the water to recede by the weekend", "doc_id": "d1"}]),
    ("", "d2", "Brightline Inc. reported quarterly revenue of $2.1 billion, up 8 percent. The company also named Ana Ruiz as chief financial officer, replacing Tom Hale, who retired in March.",
     [{"claim": "Brightline Inc. reported quarterly revenue of $2.1 billion.", "quote": "Brightline Inc. reported quarterly revenue of $2.1 billion", "doc_id": "d2"},
      {"claim": "Brightline Inc.'s quarterly revenue rose 8 percent.", "quote": "up 8 percent", "doc_id": "d2"},
      {"claim": "Brightline Inc. named Ana Ruiz as chief financial officer.", "quote": "The company also named Ana Ruiz as chief financial officer", "doc_id": "d2"},
      {"claim": "Ana Ruiz replaced Tom Hale as chief financial officer of Brightline Inc.", "quote": "replacing Tom Hale", "doc_id": "d2"},
      {"claim": "Tom Hale retired from Brightline Inc. in March.", "quote": "who retired in March", "doc_id": "d2"}]),
    ("A magnitude 6.2 earthquake struck central Chile on Sunday.", "d3", "Rescuers pulled two people alive from a collapsed hotel. The quake was felt in Santiago, 200 kilometers away.",
     [{"claim": "Rescuers pulled two people alive from a hotel that collapsed in the earthquake in central Chile on Sunday.", "quote": "Rescuers pulled two people alive from a collapsed hotel", "doc_id": "d3"},
      {"claim": "The earthquake that struck central Chile on Sunday was felt in Santiago.", "quote": "The quake was felt in Santiago", "doc_id": "d3"},
      {"claim": "Santiago is 200 kilometers from where the earthquake struck in central Chile.", "quote": "Santiago, 200 kilometers away", "doc_id": "d3"}]),
]

FACT_HEAD = """Break the sentence into atomic facts. Each fact is one short standalone statement. Do not add anything that the sentence does not say. Output a JSON array of strings only.
"""

FACT_SHOTS = [
    ("Rescuers pulled two people alive from the hotel, which collapsed during Sunday's earthquake.",
     ["Rescuers pulled two people alive from the hotel.", "The hotel collapsed during the earthquake.", "The earthquake happened on Sunday."]),
    ("Brightline's revenue rose 8 percent to $2.1 billion.",
     ["Brightline's revenue rose 8 percent.", "Brightline's revenue was $2.1 billion."]),
    ("The city council approved the budget.",
     ["The city council approved the budget."]),
]

RW_HEAD = """You write a multi-document news summary from a list of labeled claims.

Rules:
- Use only the information in the claims. Do not add facts, numbers, names, causes or judgments that are not in them.
- Follow the order of the list where you can. A sentence may combine several claims.
- End every sentence with a marker listing the labels of all claims the sentence uses, for example [g3] or [g3, g7]. Every sentence needs a marker. Do not list a label whose claim the sentence does not use.
- Some pairs of claims disagree. For each such pair, state both versions, say who holds each, and cite both labels.
- Write at most {n} words. Output the summary only.
"""

RW_SHOTS = [
    (["A magnitude 6.2 earthquake struck central Chile on Sunday.", "The earthquake was felt in Santiago.", "Rescuers pulled two people alive from a collapsed hotel."], [],
     ["A magnitude 6.2 earthquake struck central Chile on Sunday and was felt in Santiago.", [0, 1]], ["Rescuers pulled two people alive from a collapsed hotel.", [2]]),
    (["Officials in Lagos said a flood displaced about 4,000 residents.", "The Red Cross said the flood displaced more than 10,000 residents.", "Officials expect the water to recede by the weekend."], [(0, 1)],
     ["Estimates differ: officials in Lagos said about 4,000 residents were displaced, while the Red Cross said more than 10,000.", [0, 1]], ["Officials expect the water to recede by the weekend.", [2]]),
    (["Brightline Inc. reported quarterly revenue of $2.1 billion.", "Brightline Inc.'s quarterly revenue rose 8 percent.", "Brightline Inc. named Ana Ruiz as chief financial officer."], [],
     ["Brightline Inc. reported quarterly revenue of $2.1 billion, up 8 percent.", [0, 1]], ["The company named Ana Ruiz as chief financial officer.", [2]]),
]

PL_HEAD = """You write a multi-document news summary from a list of claims.

Rules:
- Use only the information in the claims. Do not add facts, numbers, names, causes or judgments that are not in them.
- Follow the order of the list where you can. A sentence may combine several claims.
- Some pairs of claims disagree. For each such pair, state both versions and say who holds each.
- Write at most {n} words. Output the summary only.
"""

RG_HEAD = """The sentence below contains content that the claims do not support. Rewrite it so that it states only what the claims support. End it with the marker {mk}. If nothing supported remains, output NONE.
"""

E2E_HEAD = """Write a summary of the news documents below in at most {n} words. After each sentence, cite the passages that support it by number in square brackets, for example [1][3]. Cite at least one passage in every sentence, and only passages that support it. Output the summary only.
"""

COV_HEAD = """Below is a list of numbered facts from a reference summary, and a candidate summary. For each fact, decide whether the candidate states it, allowing paraphrase. Output a JSON array with the numbers of the facts the candidate covers, and nothing else.
"""


def _blk(ctx, k, x):
    a = ("Context, for resolving references only. Do not extract claims from it:\n" + ctx + "\n") if ctx else ""
    return a + "Document id: " + k + "\nText:\n" + x + "\n"


def extract(x, k, ctx=""):
    s = "\n".join(_blk(a, b, t) + "JSON:\n" + json.dumps(y) + "\n" for a, b, t, y in EXT_SHOTS)
    return EXT_HEAD + "\n" + s + "\n" + _blk(ctx, k, x) + "JSON:\n"


def facts(t):
    s = "\n".join("Sentence: " + a + "\nJSON:\n" + json.dumps(b) + "\n" for a, b in FACT_SHOTS)
    return FACT_HEAD + "\n" + s + "\nSentence: " + t + "\nJSON:\n"


def _cl(xs, cf, lab):
    a = "Claims:\n" + "\n".join((("g%d: " % (i + 1)) if lab else "- ") + x for i, x in enumerate(xs))
    if cf:
        if lab:
            a += "\nConflicting pairs: " + "; ".join("g%d and g%d" % (i + 1, j + 1) for i, j in cf)
        else:
            a += "\nConflicting pairs:\n" + "\n".join("- \"" + xs[i] + "\" vs \"" + xs[j] + "\"" for i, j in cf)
    return a


def _sh(lab):
    r = []
    for xs, cf, *ys in RW_SHOTS:
        o = " ".join(t + ((" [" + ", ".join("g%d" % (i + 1) for i in g) + "]") if lab else "") for t, g in ys)
        r.append(_cl(xs, cf, lab) + "\nSummary:\n" + o + "\n")
    return "\n".join(r)


def rewrite(cl, cf, n):
    lb = {l: i for i, (l, _) in enumerate(cl)}
    xs = [t for _, t in cl]
    cf = [(lb[a], lb[b]) for a, b in cf]
    return RW_HEAD.format(n=n) + "\n" + _sh(True) + "\n" + _cl(xs, cf, True) + "\nSummary:\n"


def rewrite_plain(xs, cf, n):
    return PL_HEAD.format(n=n) + "\n" + _sh(False) + "\n" + _cl(xs, cf, False) + "\nSummary:\n"


def regen(cl, t, bad, mk):
    return RG_HEAD.format(mk=mk) + "\nClaims:\n" + "\n".join("- " + x for x in cl) + "\n\nSentence: " + t + "\nUnsupported parts:\n" + "\n".join("- " + b for b in bad) + "\n\nRewritten sentence:\n"


def e2e(ps, n):
    return E2E_HEAD.format(n=n) + "\n" + "\n\n".join("[%d] %s" % (i + 1, p) for i, p in enumerate(ps)) + "\n\nSummary:\n"


def cover(fs, h):
    return COV_HEAD + "\nFacts:\n" + "\n".join("%d. %s" % (i + 1, f) for i, f in enumerate(fs)) + "\n\nCandidate:\n" + h + "\n\nJSON:\n"
