"""The "package" family of a reference room's tasks.json, beside pick_place, push, shelve and activity:
tasks whose action and success check the room brings as code. Each entry:
  {"id": "cover_microscope",                  lower case, digits and _, unique in the room
   "words": "Take the dust cover off ...",    the task in plain words for the instruction, without object ids
   "entry": "package_tasks/cover.py",         a Python file inside the room that defines run(ctx) (below)
   "objects": ["cloth__0", "microscope__1"],  the reference room's objects it uses: top-level bodies of scene.xml or
                                              objects of nonrigid.json; a copy is run on its matched objects
   "controller": "ideal_grasp",               what acts: "rby1" (the verifiers' robot), "ideal_grasp" (a weld or kinematic
                                              grasp, no arm) or "scripted" (forces or motions applied directly)
   "level": "interaction",                    the reward level it counts in (sbv/reward.py), default interaction
   "engine": "mujoco",                        "mujoco" runs in the verifiers' Python, "genesis" in SBV_NONRIGID_PYTHON
   "params": {...},                           frozen into the case and passed to run(): thresholds belong here
   "files": ["controller_config.json"],        optional extra runtime inputs, relative to the room
   "jitter": 2,                               how many jittered runs the reference room must pass besides the nominal one
   "timeout_s": 600,                          one run's limit, seconds
   "negatives": [                             wrong copies the check must fail; at least one
     {"name": "heavy_cover", "why": "a cover ten times too heavy slides off",
      "perturb": ["--mass", "cloth_frame__0", "10"],             sbv.perturb edits of scene.xml, and/or
      "nonrigid_set": {"cloth__0": {"material": {"fabric": "denim"}}}}]}   fields merged into nonrigid.json objects
      (that name is still read)

The entry file defines run(ctx) -> dict. ctx: scene_dir, scene_xml, nonrigid (a path or None), objects (true name -> the
name to act on in this scene), params, seed (0 nominal; above 0 a jittered run: move every free object by up to 1 mm, as
api.jitter does, and say so), work_dir (a fresh folder for any output) and threads. It returns {"stages": [[name, bool],
...], "finite": bool, "jitter_applied": bool, "metrics": {name: number}, "why": str}: the task cut into its stages in
order, as the robot tasks are (docs/reference/VERIFIERS.md, "How a task scores"). It scores the share of stages passed
before the first one that fails, succeeds when all pass, and scores 0 in a non-finite state; a task with one stage
scores 0 or 1. The stage names are frozen at calibration. The run gets no hint whether it acts on the reference room, a
wrong copy or an entrant's copy, and must never read files outside ctx. Its result must not depend on appearance
(colours, textures, lights, cameras) or on files it neither loads through scene.xml nor lists in the task's optional
`files`: calibration reuse keys a trial without them."""

import json
import os
import re

FILE, KEY = "tasks.json", "package"
CONTROLLERS = ("rby1", "ideal_grasp", "scripted")
LEVELS = ("balance", "response", "affordance", "interaction")  # sbv/reward.py LEVEL_ORDER
ENGINES = ("mujoco", "genesis")
ID = re.compile(r"^[a-z][a-z0-9_]*$")
# A wrong copy keeps the task's objects: removing one, losing a mesh or rescaling the room fails every check trivially.
PERTURB_ALLOWED = (
    "--mass",
    "--com",
    "--fix",
    "--weld",
    "--flip",
    "--damping",
    "--range",
    "--solid",
    "--friction",
    "--move",
)
DEFAULTS = dict(level="interaction", engine="mujoco", params={}, jitter=2, timeout_s=600, files=[])


def room_objects(room):
    """(rigid names: the top-level bodies of scene.xml, non-rigid names: the objects of nonrigid.json)."""
    import mujoco

    m = mujoco.MjModel.from_xml_path(os.path.join(room, "scene.xml"))
    rigid = {
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(1, m.nbody) if m.body_parentid[b] == 0
    } - {None}
    from sbv.nonrigid.schema import side_file

    side = side_file(room)
    nonrigid = {o.get("name") for o in json.load(open(side)).get("objects", [])} if os.path.exists(side) else set()
    return rigid, nonrigid


def inside(room, rel):
    """The file rel names inside room, or None: relative, no '..', no symlink on the way, a regular file."""
    if not isinstance(rel, str) or not rel or os.path.isabs(rel) or ".." in rel.replace("\\", "/").split("/"):
        return None
    p = room
    for part in rel.replace("\\", "/").split("/"):
        p = os.path.join(p, part)
        if os.path.islink(p):
            return None
    return p if os.path.isfile(p) and os.path.realpath(p).startswith(os.path.realpath(room) + os.sep) else None


def has(room):
    """Whether the room's tasks.json declares package tasks."""
    try:
        with open(os.path.join(room, FILE)) as f:
            d = json.load(f)
    except (OSError, ValueError):
        return False
    return isinstance(d, dict) and KEY in d


def load(room):
    """(package tasks with defaults filled in, problems). An empty problems list means they can be calibrated."""
    try:
        with open(os.path.join(room, FILE)) as f:
            d = json.load(f)
    except (OSError, ValueError) as e:
        return [], [f"{FILE}: {e}"]
    tasks = d.get(KEY) if isinstance(d, dict) else None
    if not isinstance(tasks, list) or not tasks:
        return [], [f'{FILE}: "{KEY}" must be a nonempty list of tasks']
    try:
        rigid, nonrigid = room_objects(room)
    except Exception as e:
        return [], [f"scene.xml does not load: {type(e).__name__}: {str(e)[:200]}"]
    problems, ids, out = [], set(), []
    for i, t in enumerate(tasks):
        if not isinstance(t, dict):
            problems.append(f"task {i}: must be an object")
            continue
        unknown = set(t) - {"id", "words", "entry", "objects", "controller", "negatives"} - set(DEFAULTS)
        tid = t.get("id")
        who = f"task {tid!r}" if isinstance(tid, str) else f"task {i}"
        if unknown:
            problems.append(f"{who}: unknown fields {sorted(unknown)}")
        if not isinstance(tid, str) or not ID.fullmatch(tid) or tid in ids:
            problems.append(f"{who}: id must be lower case, digits and _, unique")
        ids.add(tid)
        w = t.get("words")
        if not isinstance(w, str) or not w.strip() or "__" in w:
            problems.append(f"{who}: words must say the task in plain words, without object ids")
        e = t.get("entry")
        if not isinstance(e, str) or not e.endswith(".py") or inside(room, e) is None:
            problems.append(f"{who}: entry must be a .py file inside the room")
        objs = t.get("objects")
        if not isinstance(objs, list) or not objs or len(set(map(str, objs))) != len(objs):
            problems.append(f"{who}: objects must be a nonempty list of distinct names")
        else:
            missing = [o for o in objs if o not in rigid and o not in nonrigid]
            if missing:
                problems.append(f"{who}: no top-level body or non-rigid object named {missing}")
        if t.get("controller") not in CONTROLLERS:
            problems.append(f"{who}: controller must be one of {CONTROLLERS}")
        t = {**DEFAULTS, **t}
        if not isinstance(t["files"], list) or not all(inside(room, p) for p in t["files"]):
            problems.append(f"{who}: files must list regular input files inside the room")
        if t["level"] not in LEVELS:
            problems.append(f"{who}: level must be one of {LEVELS}")
        if t["engine"] not in ENGINES:
            problems.append(f"{who}: engine must be one of {ENGINES}")
        if not isinstance(t["params"], dict):
            problems.append(f"{who}: params must be an object")
        else:
            try:
                json.dumps(t["params"], allow_nan=False)
            except ValueError:
                problems.append(f"{who}: params must be finite JSON")
        if type(t["jitter"]) is not int or not 2 <= t["jitter"] <= 10:
            problems.append(f"{who}: jitter must be an integer from 2 to 10")
        if (
            isinstance(t["timeout_s"], bool)
            or not isinstance(t["timeout_s"], (int, float))
            or not 1 <= t["timeout_s"] <= 3600
        ):
            problems.append(f"{who}: timeout_s must be 1 to 3600 seconds")
        problems += [f"{who}: {p}" for p in negative_problems(t.get("negatives"), nonrigid)]
        out.append(t)
    return out, problems


def negative_problems(negs, nonrigid):
    if not isinstance(negs, list) or not negs:
        return [
            "negatives must list at least one wrong copy the check has to fail (a check no wrong copy fails adds nothing)"
        ]
    problems, names = [], set()
    for n in negs:
        if not isinstance(n, dict) or set(n) - {
            "name",
            "why",
            "perturb",
            "nonrigid_set",
            "soft_set",
        }:  # soft_set.
            problems.append("a negative has name, why, perturb and/or nonrigid_set")
            continue
        name = n.get("name")
        if not isinstance(name, str) or not ID.fullmatch(name) or name in names:
            problems.append(f"negative {name!r}: name must be lower case, digits and _, unique")
        names.add(name)
        if not isinstance(n.get("why"), str) or not n["why"].strip():
            problems.append(f"negative {name!r}: why must say what is wrong in this copy")
        p, s = n.get("perturb"), n.get("nonrigid_set", n.get("soft_set"))
        if not p and not s:
            problems.append(f"negative {name!r}: give perturb edits, nonrigid_set changes, or both")
        if p is not None:
            if not isinstance(p, list) or not all(isinstance(x, str) for x in p):
                problems.append(f"negative {name!r}: perturb is a list of sbv.perturb arguments")
            else:
                bad = [x for x in p if x.startswith("--") and x not in PERTURB_ALLOWED]
                if bad or (p and p[0] not in PERTURB_ALLOWED):
                    problems.append(f"negative {name!r}: perturb edits must be among {PERTURB_ALLOWED}")
        if s is not None:
            if not isinstance(s, dict) or not all(isinstance(v, dict) for v in s.values()):
                problems.append(f"negative {name!r}: nonrigid_set maps a non-rigid object to fields")
            elif set(s) - nonrigid:
                problems.append(
                    f"negative {name!r}: nonrigid_set names no non-rigid object {sorted(set(s) - nonrigid)}"
                )
    return problems
