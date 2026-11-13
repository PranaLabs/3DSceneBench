"""Geometry axis, G1 object inventory (doc Geometry, G1): inventory F1 against the required inventory, and count
constraints. Measured on the submitted arrangement, before any settling."""

from collections import Counter


# ---- G1 --------------------------------------------------------------------------------------------------------------
def inventory(spec, scene, corr, slots):
    """Inventory F1 = 2M / (N_pred + N_req) over the object pairing (sbv/common/case.py correspond()): M reference
    objects paired with a copy object, by word or by place, so a word slip at the right place is not counted again here
    (the word is reported apart and not scored, words() below). A copy object of a permitted-extra word counts in N_pred
    only when paired. Plus counts per word: required, generated, and matched (reference objects of that word that were
    paired)."""
    req, permitted = spec["required"], set(spec.get("permitted_extras", []))
    word = {o["slot"]: o["category"] for o in slots}
    paired = set(corr.values())
    got = Counter(word[s] for s in corr if s in word)
    gen = Counter(scene.cat[n] for n in scene.objects)
    M = sum(min(got[c], n) for c, n in req.items())
    Nr = sum(req.values())
    Np = sum(1 for n in scene.objects if n in paired or scene.cat[n] not in permitted)
    per = {c: dict(required=req.get(c, 0), generated=gen[c], matched=got[c]) for c in sorted(set(req) | set(gen))}
    f1 = 1.0 if Np + Nr == 0 else (0.0 if M == 0 else 2 * M / (Np + Nr))
    return dict(f1=f1, precision=M / Np if Np else None, recall=M / Nr if Nr else None, per_category=per)


def words(slots, scene, corr):
    """Word accuracy, reported and not scored: the share of paired reference objects whose copy object has the reference
    object's category word, with each pair made under another word listed. None when nothing is paired."""
    word = {o["slot"]: o["category"] for o in slots}
    wrong = [
        dict(slot=s, word=word.get(s), copy=g, copy_word=scene.cat[g])
        for s, g in sorted(corr.items())
        if scene.cat[g] != word.get(s)
    ]
    return dict(accuracy=1 - len(wrong) / len(corr) if corr else None, paired=len(corr), wrong=wrong)


def counts(constraints, scene):
    """Count-constraint satisfaction: exactly / at_least / at_most / none, pass rate; None when no constraint
    applies."""
    if not constraints:
        return dict(score=None, results=[])
    gen = Counter(scene.cat[n] for n in scene.objects)
    res = []
    for c in constraints:
        k = gen[c["category"]]
        ok = {"exactly": k == c["n"], "at_least": k >= c["n"], "at_most": k <= c["n"], "none": k == 0}[c["op"]]
        res.append(dict(c, generated=k, passed=bool(ok)))
    return dict(score=sum(r["passed"] for r in res) / len(res), results=res)
