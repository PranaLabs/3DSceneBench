"""The frozen case (doc Appendix C.2): every target, setting and expected outcome, fixed before submissions are scored,
plus the one object correspondence that every axis reuses."""

import hashlib, json, os
import numpy as np
from scipy.optimize import linear_sum_assignment
from sbv.common.scene import main_word, topic

SCHEMA = "sbv-case/0.2"  # 0.2: two pass bars (cases from 0.1 must be calibrated again)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(folder):
    """One hash over every file in a folder (a room with its meshes, textures and tasks.json)."""
    h = hashlib.sha256()
    for p in sorted(os.path.join(r, f) for r, _, fs in os.walk(folder) for f in fs):
        h.update(os.path.relpath(p, folder).encode())
        with open(p, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:16]


def scene_xml(path):
    """A scene package is a directory with scene.xml at its root; a bare MJCF file is accepted too."""
    return os.path.join(path, "scene.xml") if os.path.isdir(path) else path


def save(case, path):
    body = {k: v for k, v in case.items() if k != "case_hash"}
    if "objects" in body:
        body["objects"] = [
            {k: v for k, v in o.items() if k not in ("_matcher", "_task_reach")} for o in body["objects"]
        ]  # load()'s
    case["case_hash"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]
    with open(path, "w") as f:
        json.dump(dict(body, case_hash=case["case_hash"]), f, indent=1, default=float)


def load(path):
    """The case, with case_hash recomputed from its content, so a case edited by hand never carries a stale hash. Each
    reference object's word, and the inventory's, is read as its main word (sbv/common/categories.txt), so a case
    calibrated before a word became an alias pairs and counts the same as a new one: a true trinket is an ornament."""
    with open(path) as f:
        case = json.load(f)
    if case.get("schema") != SCHEMA:
        raise ValueError(f"{path}: expected schema {SCHEMA}, found {case.get('schema')}")
    body = {k: v for k, v in case.items() if k != "case_hash"}
    case["case_hash"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]
    case["_dir"] = os.path.dirname(os.path.abspath(path))
    for o in case.get("objects", []) + case.get("geometry", {}).get("count_constraints", []):
        o["category"] = main_word(o["category"])
    if case.get("pairing", {}).get("matcher"):  # correspond() sees only the slots: each carries the case's matcher
        for o in case["objects"]:
            o["_matcher"] = case["pairing"]["matcher"]
    task_reach(case)
    inv = case.get("geometry", {}).get("inventory")
    if inv:
        req = {}
        for c, n in inv["required"].items():
            req[main_word(c)] = req.get(main_word(c), 0) + n
        inv["required"] = req
        inv["permitted_extras"] = sorted({main_word(c) for c in inv.get("permitted_extras", [])})
    return case


def reference_path(case):
    """The reference room the case was calibrated on; refuses a file that changed since (calibrate again). A case
    calibrated before calibrate recorded reference_files is checked by its scene file's hash alone, as before, so it is
    still graded; bind_reference gives it the manifest when its retained input hashes are known."""
    p = case["reference"] if os.path.isabs(case["reference"]) else os.path.join(case["_dir"], case["reference"])
    if case.get("reference_sha256") and file_sha256(p) != case["reference_sha256"]:
        raise ValueError(f"{p} changed since calibration: calibrate again")
    from .dependencies import model_files

    if case.get("reference_files") is not None and model_files(p) != case["reference_files"]:
        raise ValueError(f"{p} dependencies changed since calibration; reuse only unaffected calibration records")
    return p


def bind_reference(case, retained_files):
    """Bind an old case to hashes retained at calibration, never to an unverified current snapshot.

    Callers supply the original input manifest (paths relative to scene.xml), then save
    the case. This preserves measurements without pretending missing hashes were known.
    """
    from .dependencies import model_files

    p = case["reference"] if os.path.isabs(case["reference"]) else os.path.join(case["_dir"], case["reference"])
    if not retained_files or model_files(p) != retained_files:
        raise ValueError("retained reference dependency hashes do not match the current inputs")
    if case.get("reference_sha256") and file_sha256(p) != case["reference_sha256"]:
        raise ValueError("main reference changed since calibration")
    case["reference_files"] = dict(retained_files)


SAME_SIZE = (
    2.0  # the word step pairs a reference object only with a copy object whose box diagonal is within 2x of its own
)
# copy object of another size still pairs by place.
PLACE_MIN, PLACE_SIZE = (
    0.15,
    3.0,
)  # the place step of correspond(): centres within max(true box diagonal, 0.15 m), at most
# cross_gate(), and, seen from above, within max(true footprint diagonal, 0.15 m); sizes within 3x


def nearest(R, G, scene, cap=None, fit=None):
    """One-to-one pairing of reference slots R with generated object names G that has the smallest total distance
    between bounding-box centres (Hungarian); exact ties go to name order. With cap (one distance per slot), farther
    pairs are dropped; with fit(slot, name), so are the pairs it refuses."""
    cg = [scene.center(g) for g in G]
    cost = np.array([[np.linalg.norm(np.asarray(o["center"]) - c) for c in cg] for o in R])
    far = cost > np.asarray(cap)[:, None] if cap is not None else np.zeros(cost.shape, bool)
    if fit is not None:
        far |= ~np.array([[fit(o, g) for g in G] for o in R], bool)
    rows, cols = linear_sum_assignment(np.where(far, 1e6, cost) + 1e-9 * np.arange(len(G))[None, :])
    return {R[i]["slot"]: G[j] for i, j in zip(rows, cols) if not far[i, j]}


def correspond(slots, scene, names=None, matcher=None):
    """Reference slot -> generated object name. First within each category, by nearest centres. Then a reference object
    with hinge or slide joints that is still unpaired takes an unpaired generated object of any category that has such
    joints too, within the reference object's own size (its box diagonal): a chest of drawers named cabinet instead of
    drawer is still found, and its drawers pass or fail on their physics. Then every reference object still unpaired
    takes an unpaired generated object of any category by its place: centres within max(its box diagonal, 0.15 m) and,
    seen from above, within max(its footprint diagonal, 0.15 m), so a tall floor lamp's reach stays its own footprint
    wide and a table lamp a metre beside it is not taken for it; box diagonals within 3 times of each other. The word is
    then reported apart (g1_inventory.words(), not scored), not by losing the object's verifiers. Only words of one
    topic in the word list pair this way (scene.topic(): a figurine can be the trinket, a pillow cannot be the plant,
    nor a book the trinket); a word that is not on the list may pair with any. Unpaired slots are missing. Every slot
    pairs by its word only with a copy object whose box diagonal is within SAME_SIZE of its own, and only within its
    gate() of its centre always: the "gate" sbv/pair_gate.py froze in the case, else the default max(0.3 m, its box
    diagonal); a robot task's object at least its true support's diagonal); farther, only the place step can find it,
    under its own limits. Both steps across words reach at most cross_gate(). A case whose pairing says matcher
    "geometric" (sbv/pair_gate.py --matcher; load() puts it on every slot) pairs by correspond_geometric() instead.
    names: the copy objects it may pair (default every one; sbv/groups.py keeps a group's copy objects for the
    group)."""
    if (matcher or next((o.get("_matcher") for o in slots if o.get("_matcher")), None)) == "geometric":
        return correspond_geometric(slots, scene, names)
    out = {}
    pool = sorted(scene.objects if names is None else names)
    for c in sorted({o["category"] for o in slots}):
        R = [o for o in slots if o["category"] == c]
        G = [n for n in pool if scene.cat[n] == c]
        if G:  # position gate on every slot, frozen in the case or default; and
            size = {}  # sizes within SAME_SIZE, when the slot has a size

            def alike(o, g):
                if o.get("dims") is None:
                    return True
                if g not in size:
                    size[g] = float(np.linalg.norm(np.subtract(*scene.aabb(g)[::-1])))
                return 1 / SAME_SIZE <= size[g] / max(float(np.linalg.norm(o["dims"])), 1e-9) <= SAME_SIZE

            out.update(nearest(R, G, scene, [gate(o) for o in R], alike))
    R = [o for o in slots if o.get("role", "").endswith("+joints") and o["slot"] not in out]
    used = set(out.values())
    G = [n for n in pool if scene.joints[n] and n not in used]
    if R and G:
        out.update(nearest(R, G, scene, [min(float(np.linalg.norm(o["dims"])), cross_gate(o)) for o in R]))
    R = [o for o in slots if o["slot"] not in out]
    used = set(out.values())
    G = [n for n in pool if n not in used]
    if R and G:
        box = {g: scene.aabb(g) for g in G}
        diag = lambda v: float(np.linalg.norm(v))
        kin = lambda a, b: None in (topic(a), topic(b)) or topic(a) == topic(b)  # a word not on the list pairs with any
        fit = lambda o, g: (
            1 / PLACE_SIZE <= diag(box[g][1] - box[g][0]) / max(diag(o["dims"]), 1e-9) <= PLACE_SIZE
            and diag(np.subtract(o["center"][:2], (box[g][0][:2] + box[g][1][:2]) / 2))
            <= max(diag(o["dims"][:2]), PLACE_MIN)
            and kin(o["category"], scene.cat[g])
        )
        out.update(nearest(R, G, scene, [min(max(diag(o["dims"]), PLACE_MIN), cross_gate(o)) for o in R], fit))
    return out


WORD_COST, TOPIC_COST, JOINT_COST = (
    0.05,
    0.02,
    0.03,
)  # metres added to a pair's centre distance: tie-breaks, never vetoes
SIZE_COST = (
    0.2  # metres per unit of |ln(box diagonal ratio)|: of two objects in one place (a cup in a box), size decides,
)
# not the word (a reference room renamed to words of other topics: 629/633 identity without it)
CROSS_FACTOR, CROSS_CAP = 0.5, 1.0  # words that differ: centres within min(1 m, max(0.3 m, half the true diagonal))
AXIS_FLOOR = 0.05  # metres: the per-axis size bar compares a 1 cm and a 4 cm thick board as equal


TASK_KINDS = ("robot", "robot_push", "robot_shelve", "robot_activity")


def task_reach(case):
    """A robot task's own objects may stand anywhere on their true support: each slot a robot task names gets
    "_task_reach", the box diagonal of the object it rests on in the reference room, so a cup paired anywhere on its
    console still lets the task run; gate() takes the larger of that and the object's own gate. Never saved (save())."""
    under = {w: s for s, w in case.get("support_pairs", [])}
    dims = {o.get("slot"): o["dims"] for o in case.get("objects", []) if o.get("dims") is not None}
    named = {
        s
        for u in case.get("functionality", {}).get("units", [])
        if u.get("kind") in TASK_KINDS
        for s in u.get("slots", [])
    }
    for o in case.get("objects", []):
        s = under.get(o.get("slot"))
        if o.get("slot") in named and s in dims:
            o["_task_reach"] = round(float(np.linalg.norm(dims[s])), 3)


def gate(o):
    """A reference object's reach: its frozen gate, else sbv/pair_gate.py's default max(0.3 m, its box diagonal); a
    robot task's object at least the diagonal of its true support (task_reach)."""
    from sbv.pair_gate import FACTOR, FLOOR  # (here: pair_gate imports this module)

    if o.get("gate") is not None:
        g = o["gate"]
    elif o.get("dims") is not None:
        g = max(FLOOR, FACTOR * float(np.linalg.norm(o["dims"])))
    else:
        g = np.inf  # a slot without a size (made by hand in a test) has no gate
    return max(g, o.get("_task_reach", 0.0))


def cross_gate(o):
    """The reach of a pair whose words differ, and of a split's pieces: min(the gate, CROSS_CAP, max(0.3 m, CROSS_FACTOR
    x the box diagonal)), so a 2 m counter takes a fryer of another word only within 1 m."""
    from sbv.pair_gate import FLOOR

    return min(gate(o), CROSS_CAP, max(FLOOR, CROSS_FACTOR * float(np.linalg.norm(o["dims"]))))


def _boxes(scene, names=None):
    """Copy objects: names, world boxes (lo, hi), centres, box diagonals, and length, width, height as the case measures
    a reference object (smallest footprint rectangle of the visible geoms, longer side first); a stand-in scene: its
    box."""
    G = sorted(scene.objects if names is None else names)
    B = [scene.aabb(g) for g in G]  # names: the copy objects it may use
    lo, hi = np.array([b[0] for b in B]).reshape(-1, 3), np.array([b[1] for b in B]).reshape(-1, 3)
    e = hi - lo
    if hasattr(scene, "geom_points"):
        from sbv.geometry.g2_shape import oriented_dims

        e = np.array(
            [oriented_dims(np.vstack([scene.geom_points(k) for k in scene.visible(scene.ogeoms[g])])) for g in G]
        ).reshape(-1, 3)
    return (
        G,
        lo,
        hi,
        (lo + hi) / 2,
        np.linalg.norm(hi - lo, axis=1),
        np.c_[e[:, :2].max(1), e[:, :2].min(1), e[:, 2]],
        [scene.cat[g] for g in G],
    )


def allowed(slots, boxes, same):
    """Centre distances, size ratios (copy / true diagonal of length, width, height) and the pairs allowed: length,
    width and height each within 3x (AXIS_FLOOR) and the diagonals too; centres within the reference object's gate when
    the words agree (same), within min(CROSS_CAP, max(0.3 m, CROSS_FACTOR x its diagonal)) when they differ: the word is
    evidence for a pair, never a veto."""
    ext = boxes[5]
    d = np.array([o["dims"] for o in slots], float).reshape(-1, 3)
    te = np.c_[d[:, :2].max(1), d[:, :2].min(1), d[:, 2]]
    td = np.linalg.norm(te, axis=1)
    dist = np.linalg.norm(np.array([o["center"] for o in slots], float)[:, None] - boxes[3][None], axis=2)
    ratio = np.linalg.norm(ext, axis=1)[None] / np.maximum(td[:, None], 1e-9)
    axes = np.maximum(ext[None], AXIS_FLOOR) / np.maximum(te[:, None], AXIS_FLOOR)
    fits = (
        np.all((axes >= 1 / PLACE_SIZE) & (axes <= PLACE_SIZE), axis=2)
        & (ratio >= 1 / PLACE_SIZE)
        & (ratio <= PLACE_SIZE)
    )
    reach = np.where(
        same, np.array([gate(o) for o in slots])[:, None], np.array([cross_gate(o) for o in slots])[:, None]
    )
    return dist, ratio, fits & (dist <= reach)


def correspond_geometric(slots, scene, names=None):
    """Reference slot -> generated object name by 3D content, not by label. One Hungarian over all true and copy objects
    on centre distance plus SIZE_COST x |ln(diagonal ratio)|, over the pairs allowed() (size within 3x on each axis; the
    reference object's gate, tighter when the words differ). The word adds WORD_COST, a topic of another kind
    TOPIC_COST, and a jointed reference object a copy without joints JOINT_COST: tie-breaks, never vetoes. Then a
    reference object built as pieces (splits()) takes its largest piece. A copy box that swallows small reference
    objects (merges()) does not pair them: they stay missing."""
    boxes = _boxes(scene, names)
    G = boxes[0]
    out = {}
    if slots and G:
        tw, gw = np.array([o["category"] for o in slots], object), np.array([scene.cat[g] for g in G], object)
        dist, ratio, ok = allowed(slots, boxes, tw[:, None] == gw[None])
        tt, gt = np.array([topic(w) for w in tw], object), np.array([topic(w) for w in gw], object)
        kin = (tt[:, None] == gt[None]) | (tt == None)[:, None] | (gt == None)[None]  # noqa: E711 (elementwise)
        jt = np.array([o.get("role", "").endswith("+joints") for o in slots])
        jg = np.array([bool(scene.joints[g]) for g in G])
        cost = (
            dist
            + SIZE_COST * np.abs(np.log(np.maximum(ratio, 1e-9)))
            + WORD_COST * (tw[:, None] != gw[None])
            + TOPIC_COST * ~kin
            + JOINT_COST * (jt[:, None] & ~jg[None])
        )
        rows, cols = linear_sum_assignment(np.where(ok, cost, 1e6) + 1e-9 * np.arange(len(G))[None, :])
        out = {slots[i]["slot"]: G[j] for i, j in zip(rows, cols) if ok[i, j]}
    for s, pieces in splits(slots, out, boxes).items():
        out[s] = pieces[0]
    return out


def splits(slots, out, boxes):
    """True slot -> copy pieces, largest first: two or more copy objects free of other reference objects (its own pair
    or unpaired, each smaller than it), centres within its cross_gate() (the full gate took in a counter's bottles and
    monitors 3 m away), parts and not contents (each at least a third of its diagonal, the largest at least half; one
    word, the copy's own grouping: counter__0 and counter__0_door_0), whose union box fits it (diagonal within 3x,
    centre within the cross gate) better than the largest piece alone does: a counter built as a carcass plus loose
    doors, not a shelf's 90 food items or a box of fries. Largest reference objects claim their pieces first."""
    G, lo, hi, ctr, size = boxes[:5]
    cat = boxes[6]
    owner, idx, taken, res = {g: s for s, g in out.items()}, {g: j for j, g in enumerate(G)}, set(), {}
    for o in sorted(slots, key=lambda o: -float(np.linalg.norm(o["dims"]))):
        s, d, c, r = o["slot"], float(np.linalg.norm(o["dims"])), np.asarray(o["center"], float), cross_gate(o)
        near = np.flatnonzero(np.linalg.norm(ctr - c, axis=1) <= r)
        P = sorted(
            (
                j
                for j in near
                if G[j] not in taken
                and size[j] >= d / PLACE_SIZE
                and (owner.get(G[j]) == s or (G[j] not in owner and size[j] < d))
            ),
            key=lambda j: (-size[j], G[j]),
        )
        P = [j for j in P if cat[j] == cat[P[0]]]
        if len(P) < 2 or size[P[0]] < d / 2 or (s in out and idx[out[s]] not in P):
            continue
        ulo, uhi = lo[P].min(0), hi[P].max(0)
        err = lambda x: abs(np.log(max(x, 1e-9) / max(d, 1e-9)))
        if (
            err(np.linalg.norm(uhi - ulo)) <= np.log(PLACE_SIZE)
            and np.linalg.norm((ulo + uhi) / 2 - c) <= r
            and err(np.linalg.norm(uhi - ulo)) < err(size[P[0]])
        ):
            res[s] = [G[j] for j in P]
            taken |= set(res[s])
    return res


def merges(slots, out, boxes):
    """Copy object -> true slots it swallows: unpaired reference objects smaller than a third of its box diagonal whose
    centres lie inside its box (each counted once, in the smallest such box), two or more per copy object. A copy that
    built a fryer as one box and none of its food: the food is missing, a 3D error and not a naming one."""
    G, lo, hi, ctr, size = boxes[:5]
    res = {}
    for o in slots:
        if o["slot"] in out:
            continue
        c = np.asarray(o["center"], float)
        d = float(np.linalg.norm(o["dims"]))
        hold = np.flatnonzero((size > 3 * d) & np.all(lo <= c, axis=1) & np.all(c <= hi, axis=1))
        if len(hold):
            res.setdefault(G[hold[np.argmin(size[hold])]], []).append(o["slot"])
    return {g: v for g, v in sorted(res.items()) if len(v) >= 2}


def match_report(slots, scene, out, matcher=None):
    """What a pairing did, for the report (never scored): which matcher, pairs, pairs whose words differ, merges,
    splits."""
    b = _boxes(scene)
    slot = {o["slot"]: o for o in slots}
    m = matcher or next((o.get("_matcher") for o in slots if o.get("_matcher")), None) or "word"
    wrong = sorted([s, g] for s, g in out.items() if slot[s]["category"] != scene.cat[g])
    mg = merges(slots, out, b)
    return dict(
        matcher=m,
        pairs=len(out),
        word_mismatch=wrong,
        merges=mg,
        merged_slots=sum(map(len, mg.values())),
        splits=splits(slots, out, b),
    )
