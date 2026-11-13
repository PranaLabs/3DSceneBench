"""profile ROOM CASE: where a room's robot and function verification time goes, cheaply: a sampled whole-room settle
(instrumented), each robot task's frozen zone settle (the settle of a changed trial), one nominal trial per task under
layer 1, and the room's collision geometry. Writes profile.json and profile.md.

Numbers that are measured: build, settle and task wall times, contacts, solver iterations, awake trees, restless
bodies, mesh sizes. Numbers that are modelled (labelled so): per-body time shares (from contacts and constraint rows)
and the calibration totals (trials per task times the measured trial parts)."""

import collections, json, math, os, time
import numpy as np

from sbv.speed import run as R

SAMPLE_S = 0.5  # seconds of whole-room settle sampled (the plain settle is settle_s, 5 s)
ZONE_EVERY = 25  # zone settle: one sample of awake trees every 25 steps
EVERY = 5  # sampled settle: contacts and island loads read every EVERY steps
STILL_V = 1e-3  # m/s: a body faster than this at the end of a sample is still moving
GEOM_NAMES = {
    0: "plane",
    1: "hfield",
    2: "sphere",
    3: "capsule",
    4: "ellipsoid",
    5: "cylinder",
    6: "box",
    7: "mesh",
    8: "sdf",
}


def _owner_names(w):
    return np.array([w.name(w.owner[b]) for b in range(w.m.nbody)], dtype=object)


def collision_complexity(w):
    """Per object: its colliding geoms by type, and for mesh geoms the vertex and face counts of the mesh and of the
    convex hull MuJoCo collides with."""
    m = w.m
    names = _owner_names(w)
    out = collections.defaultdict(
        lambda: dict(geoms=0, types=collections.Counter(), mesh_verts=0, mesh_faces=0, hull_verts=0, hull_faces=0)
    )
    for g in range(m.ngeom):
        if not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        b = m.geom_bodyid[g]
        if b in w.robot:
            continue
        n = names[b]
        o = out[n]
        o["geoms"] += 1
        t = int(m.geom_type[g])
        o["types"][GEOM_NAMES.get(t, str(t))] += 1
        if t == 7 and m.geom_dataid[g] >= 0:
            k = m.geom_dataid[g]
            o["mesh_verts"] += int(m.mesh_vertnum[k])
            o["mesh_faces"] += int(m.mesh_facenum[k])
            ga = m.mesh_graphadr[k]
            if ga >= 0:
                o["hull_verts"] += int(m.mesh_graph[ga])
                o["hull_faces"] += int(m.mesh_graph[ga + 1])
    return {n: dict(o, types=dict(o["types"])) for n, o in out.items()}


def sampled_settle(w, seconds=SAMPLE_S):
    """Step the room plus parked robot from the file's start, all awake (the plain whole-room settle), for `seconds`;
    per step its wall time, MuJoCo's stage timers, contacts by object and geom-type pair, and the solver iterations of
    the island each object is in."""
    import mujoco

    m, d = w.m, w.d
    names = _owner_names(w)
    n = int(round(seconds / m.opt.timestep))
    x0 = d.xipos.copy()
    contacts, pairs, cons = collections.Counter(), collections.Counter(), collections.Counter()
    step_s = []
    tm = mujoco.mjtTimer
    keys = ("mjTIMER_POS_COLLISION", "mjTIMER_CONSTRAINT", "mjTIMER_STEP")
    for k in keys:
        d.timer[getattr(tm, k)].duration = 0.0
    gtype = np.array([GEOM_NAMES.get(int(t), str(t)) for t in m.geom_type], dtype=object)
    gname = names[m.geom_bodyid]
    for k in range(n):
        t = time.perf_counter()
        mujoco.mj_step(m, d)
        step_s.append(time.perf_counter() - t)
        if k % EVERY:
            continue  # contacts and islands every EVERY steps (counts scaled), numpy only
        g1, g2 = d.contact.geom1[: d.ncon], d.contact.geom2[: d.ncon]
        for side in (g1, g2):
            u, c = np.unique(gname[side], return_counts=True)
            contacts.update({a: EVERY * int(b) for a, b in zip(u, c)})
        if d.ncon:
            ta, tb = gtype[g1].astype(str), gtype[g2].astype(str)
            lo, hi = np.where(ta < tb, ta, tb), np.where(ta < tb, tb, ta)
            u, c = np.unique(np.char.add(np.char.add(lo, "-"), hi), return_counts=True)
            pairs.update({tuple(str(a).split("-")): EVERY * int(b) for a, b in zip(u, c)})
        if d.nisland > 0:  # constraint rows times iterations, shared by the island's trees
            q = min(d.nisland, len(d.solver_niter))
            it = np.full(d.nisland, float(np.mean(d.solver_niter[:q])) if q else 0.0)
            it[:q] = d.solver_niter[:q]  # MuJoCo keeps solver statistics for its first mjNISLAND islands only
            load = it * d.island_nefc[: d.nisland] / np.maximum(1, d.island_ntree[: d.nisland])
            isl = d.tree_island[: m.ntree]
            tr = np.nonzero(isl >= 0)[0]
            for a, b in zip(names[m.tree_bodyadr[tr]], load[isl[tr]]):
                cons[a] += EVERY * float(b)
    v = np.linalg.norm(d.cvel[:, 3:], axis=1)
    moved = np.linalg.norm(d.xipos - x0, axis=1)
    still_moving = sorted(
        {names[b] for b in range(1, m.nbody) if b not in w.robot and m.body_treeid[b] >= 0 and (v[b] > STILL_V)}
    )
    timers = {k[8:].lower(): round(d.timer[getattr(tm, k)].duration, 3) for k in keys}
    return dict(
        steps=n,
        sim_s=seconds,
        wall_s=round(sum(step_s), 3),
        ms_per_step=round(1000 * float(np.mean(step_s)), 3),
        ms_per_step_last50=round(1000 * float(np.mean(step_s[-50:])), 3),
        timers_s=timers,
        ncon_end=int(d.ncon),
        ntree=int(m.ntree),
        contacts=dict(contacts),
        pairs={"-".join(k): v for k, v in pairs.items()},
        constraint_load=dict(cons),
        still_moving=still_moving,
        moved_cm={
            names[b]: round(100 * float(moved[b]), 2) for b in range(1, m.nbody) if moved[b] > 0.01 and b not in w.robot
        },
    )


def zone_settle(w, unit, restless, seed=1):
    """The settle of a changed frozen trial of this task (pick_place.setup with a jitter seed), on w, a fresh world at
    the task's start: its awake set, and the awake trees, contacts and wall time over the 5 s, sampled every ZONE_EVERY
    steps; the bodies that woke on contact (awake at the end, outside the starting set) and those of the zone that
    never moved."""
    import mujoco
    from sbv.functionality.f4_robot import plan as RP

    p = unit["params"]
    build = 0.0
    named = [n for n in p.get("near", []) + p.get("needed", []) if n in w.objects]
    keep = [s for s in unit["slots"] if s in w.objects]
    w.jitter(seed)
    zone = RP.around(w, keep)
    awake = set(restless) | set(named) | zone
    saved = w.freeze(awake)
    m, d = w.m, w.d
    x0 = d.xipos.copy()
    n = int(w.phys["settle_s"] / m.opt.timestep)
    samples = []
    t1 = time.time()
    start_awake = set()
    for k in range(n):
        mujoco.mj_step(m, d)
        if k == 0:
            start_awake = {
                w.name(m.tree_bodyadr[t]) for t in range(m.ntree) if d.tree_asleep[t] < 0
            }  # after the first step
        if k % ZONE_EVERY == 0:
            samples.append((round(float(d.time), 3), int(d.ntree_awake), int(d.ncon)))
    wall = time.time() - t1
    m.tree_sleep_policy[:] = saved
    end_awake = {w.name(m.tree_bodyadr[t]) for t in range(m.ntree) if d.tree_asleep[t] < 0}
    moved = np.linalg.norm(d.xipos - x0, axis=1)
    objs = {nm: b for nm, b in w.objects.items()}
    return dict(
        id=unit["id"],
        restless=sorted(restless),
        settle_wall_s=round(wall, 2),
        ms_per_step=round(1000 * wall / max(1, n), 3),
        zone_m=RP.AWAKE_M,
        zone=sorted(zone),
        awake_names=sorted(awake),
        awake_trees_start=len(start_awake),
        awake_trees_max=max(s[1] for s in samples),
        awake_trees_end=int(d.ntree_awake),
        samples=samples,
        woke_on_contact=sorted(x for x in end_awake - start_awake if x in objs),
        zone_never_moved=sorted(nm for nm in zone if nm in objs and moved[objs[nm]] < 1e-4 and nm not in keep),
    )


def _shares(contacts, cons):
    tc, tk = sum(contacts.values()) or 1, sum(cons.values()) or 1
    return {n: round(0.5 * contacts.get(n, 0) / tc + 0.5 * cons.get(n, 0) / tk, 5) for n in set(contacts) | set(cons)}


def _category(n):
    return n.split("__")[0] if "__" in n else n


def profile(room, case_path, out, procs=4, sample_s=SAMPLE_S, cache=None, log=print):
    from sbv.common import case as K
    from sbv.functionality.f4_robot import plan as RP

    t_all = time.time()
    os.makedirs(out, exist_ok=True)
    case = json.load(open(case_path))
    xml = K.scene_xml(os.path.abspath(room))
    cache = cache or os.path.join(out, "cache")
    env = R.mode_env("l1", cache)
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    rep = dict(room=os.path.abspath(room), case=os.path.abspath(case_path), xml=xml, measured={}, modelled={})
    log("profile: building the room plus robot")
    t0 = time.time()
    start = tuple(case["geometry"]["navigation"]["start"]) + (0.0,)
    w = RP.World(xml, start, True, case["physics"])
    build = time.time() - t0
    m = w.m
    rep["model"] = dict(
        nbody=int(m.nbody),
        ngeom=int(m.ngeom),
        ntree=int(m.ntree),
        nv=int(m.nv),
        objects=len(w.objects),
        free_objects=sum(1 for b in w.objects.values() if m.body_jntadr[b] >= 0 and m.jnt_type[m.body_jntadr[b]] == 0),
        timestep=float(m.opt.timestep),
        solver=int(m.opt.solver),
        integrator=int(m.opt.integrator),
        build_s=round(build, 2),
    )
    rep["collision"] = collision_complexity(w)
    from sbv.common.scene import Scene

    sc = Scene(xml, case["physics"])
    rep["free"] = sorted(n for n in sc.objects if sc.free_dof(n) is not None)
    del sc
    log(f"profile: sampled settle {sample_s} s ({m.nbody} bodies)")
    ss = sampled_settle(w, sample_s)
    rep["sampled_settle"] = ss
    settle_steps = int(case["physics"].get("settle_s", 5.0) / m.opt.timestep)
    rep["modelled"]["full_settle_s"] = round(ss["ms_per_step"] * settle_steps / 1000, 1)
    del w
    units, bad = R.named_units(xml, case)
    units = [R.as_mode(u, "l1") for u in units]
    rep["tasks"] = [u["id"] for u in units]
    rep["tasks_refused_at_build"] = bad
    log(f"profile: one nominal trial per task, layer 1 ({len(units)} tasks, {procs} procs)")
    res, wall = R.pool(xml, case, [("trials", u, [(0, None)]) for u in units], procs, "l1", cache, log=log, probe=True)
    rep["nominal"] = {r["id"]: r for r in res}
    rep["measured"]["nominal_wall_s"] = wall
    full = [t["stages"].get("restore_s", 0) for r in res if not r.get("error") for t in r["trials"]]
    rep["measured"]["full_settle_s"] = (
        round(max(full), 1) if full else None
    )  # the worker that settled a start (the others waited, then read it)
    kept = os.path.join(cache, "robot_settle")
    rep["settle_starts"] = len([f for f in os.listdir(kept) if f.endswith(".json")]) if os.path.isdir(kept) else None
    log("profile: the zone settle of a changed trial per task (and the restless bodies of its start)")
    res, wall = R.pool(xml, case, [("zone", u, 1) for u in units], procs, "l1", cache, log=log)
    rep["zones"] = {r["id"]: (r["zone"] if "zone" in r else dict(error=r.get("error"))) for r in res}
    rep["restless"] = sorted({n for z in rep["zones"].values() for n in z.get("restless", [])})
    rep["measured"]["zones_wall_s"] = wall
    rep["top"] = top_contributors(rep)
    rep["cost"] = cost_model(rep, case)
    rep["measured"]["profile_wall_s"] = round(time.time() - t_all, 1)
    json.dump(R.jsonable(rep), open(os.path.join(out, "profile.json"), "w"), indent=1)
    open(os.path.join(out, "profile.md"), "w").write(markdown(rep))
    log(f"profile: wrote {out}/profile.json and profile.md ({rep['measured']['profile_wall_s']} s)")
    return rep


def recost(profile_json, case_path):
    """Recompute a profile's cost model and top contributors after a rule change, from its own
    measurements (and the room's free objects when the profile predates that field)."""
    from sbv.common.scene import Scene

    rep = json.load(open(profile_json))
    case = json.load(open(case_path))
    if "free" not in rep:
        sc = Scene(rep["xml"], case["physics"])
        rep["free"] = sorted(n for n in sc.objects if sc.free_dof(n) is not None)
    if "settle_starts" not in rep:
        kept = os.path.join(os.path.dirname(profile_json), "cache", "robot_settle")
        rep["settle_starts"] = (
            len([f for f in os.listdir(kept) if f.endswith(".json")]) if os.path.isdir(kept) else None
        )
    rep["top"] = top_contributors(rep)
    rep["cost"] = cost_model(rep, case)
    json.dump(R.jsonable(rep), open(profile_json, "w"), indent=1)
    open(os.path.join(os.path.dirname(profile_json), "profile.md"), "w").write(markdown(rep))
    return rep


def top_contributors(rep):
    """The largest costs, in words with their numbers: trial stages summed over the nominal trials, objects of the
    whole-room settle (modelled share), categories, and the zones with the most awake trees."""
    from sbv.functionality.f4_robot.rby1 import P as ROBOT  # the robot's own bodies are not room objects

    ss = rep["sampled_settle"]
    sh = _shares(ss["contacts"], ss["constraint_load"])
    robotish = lambda n: n.startswith(ROBOT) or n in ("world", "floor")
    objs = sorted(((v, n) for n, v in sh.items() if not robotish(n)), reverse=True)[:15]
    cats = collections.Counter()
    for n, v in sh.items():
        if not robotish(n):
            cats[_category(n)] += v
    stage = collections.Counter()
    for r in rep["nominal"].values():
        for t in r.get("trials", []):
            for k, v in t["stages"].items():
                stage[k] += v
    zones = sorted(
        ((z.get("awake_trees_max", 0), i, z.get("settle_wall_s")) for i, z in rep["zones"].items() if "error" not in z),
        reverse=True,
    )
    coll = sorted(
        ((c["hull_verts"], n, c["mesh_verts"], c["geoms"]) for n, c in rep["collision"].items()), reverse=True
    )[:10]
    return dict(
        settle_objects=[dict(object=n, share=round(v, 4), contacts=ss["contacts"].get(n, 0)) for v, n in objs],
        settle_categories=[
            dict(category=c, share=round(v, 4), n=sum(1 for x in sh if _category(x) == c))
            for c, v in cats.most_common(8)
        ],
        nominal_stages={k: round(v, 1) for k, v in stage.most_common()},
        zones=[dict(id=i, awake_trees_max=a, settle_wall_s=s) for a, i, s in zones],
        hulls=[dict(object=n, hull_verts=h, mesh_verts=mv, geoms=g) for h, n, mv, g in coll],
        pairs=dict(sorted(ss["pairs"].items(), key=lambda x: -x[1])[:8]),
    )


def cost_model(rep, case):
    """Modelled calibration seconds of the room's robot tasks (calibrate._robot_job: reference, jitter seeds, two
    edits, ten nudges when the task needs a free object), per task, plain and layer 1, from the measured parts:
    plain trial = build + floor + whole-room settle + task; plain edit = the same with the task run with the room awake
    (task steps times the whole-room step time); layer-1 changed trial = build + floor + its zone settle + task."""
    J = case.get("calibration", {}).get("jitter_seeds", 3)
    full = rep["measured"].get("full_settle_s") or rep["modelled"]["full_settle_s"]
    step_full = rep["sampled_settle"]["ms_per_step"] / 1000
    out = {}
    unit_slots = {u["id"]: set(u["slots"]) for u in case["functionality"]["units"]}
    unit_slots.update({i: set(r.get("slots") or []) for i, r in rep.get("nominal", {}).items() if r.get("slots")})
    for i, r in rep["nominal"].items():
        if r.get("error") or not r.get("trials"):
            continue
        t = r["trials"][0]
        st = t["stages"]
        base = st.get("build_s", 0) + st.get("floor_s", 0)
        task = st.get("task_s", 0)
        z = rep["zones"].get(i, {})
        zone = z.get("settle_wall_s", 0.0) or 0.0
        steps = (t.get("sim_s") or 0) / rep["model"]["timestep"] - case["physics"].get("settle_s", 5.0) / rep["model"][
            "timestep"
        ]
        free = set(rep.get("free") or [])
        needed = [n for n in (t.get("passed_by") or []) if n not in unit_slots.get(i, ())]
        n_nudge = 10 if any(n in free for n in needed) else 0  # calibrate nudges only the free objects a task needs
        ok = t.get("ok")
        if not ok:  # refused at the reference run: one trial
            plain, l1 = base + full + task, base + task
            counts = dict(nominal=1)
        else:
            plain = (1 + J + n_nudge) * (base + full + task) + 2 * (base + full + max(task, steps * step_full))
            l1 = (1 + J) * (base + task) + (n_nudge + 2) * (
                base + zone + task
            )  # jitter seeds restore their own kept settle
            counts = dict(nominal=1, jitter=J, edits=2, nudges=n_nudge)
        out[i] = dict(plain_s=round(plain, 1), l1_s=round(l1, 1), trials=counts)
    J1 = 1 + case.get("calibration", {}).get("jitter_seeds", 3)  # one kept whole-room settle per start and seed
    tot_p = sum(v["plain_s"] for v in out.values())
    tot_l = sum(v["l1_s"] for v in out.values()) + full * (rep.get("settle_starts") or 1) * J1
    return dict(
        per_task=out,
        plain_s=round(tot_p, 1),
        l1_s=round(tot_l, 1),
        note="modelled from the measured parts; layer 1 adds one whole-room settle per start and jitter seed",
    )


def markdown(rep):
    md = rep["model"]
    ss = rep["sampled_settle"]
    top = rep["top"]
    c = rep["cost"]
    L = [
        f"# Speed profile: {os.path.basename(rep['room'].rstrip('/'))}",
        "",
        f"- model: {md['nbody']} bodies, {md['ngeom']} geoms, {md['ntree']} trees, {md['objects']} objects ({md['free_objects']} free); room+robot build {md['build_s']} s",
        f"- whole-room settle, all awake (sampled {ss['sim_s']} s): {ss['ms_per_step']} ms/step (last 50: {ss['ms_per_step_last50']}), "
        f"timers {ss['timers_s']}, {ss['ncon_end']} contacts at the end",
        f"- full whole-room settle: measured {rep['measured'].get('full_settle_s')} s, modelled {rep['modelled']['full_settle_s']} s; "
        f"{rep.get('settle_starts')} distinct task start(s), one settle each (a nominal trial's 'restore' includes waiting for it)",
        f"- restless after the full settle (kept awake in every changed trial): {len(rep['restless'])} {rep['restless'][:12]}",
        f"- still moving at the end of the sample (v > {STILL_V} m/s): {len(ss['still_moving'])} {ss['still_moving'][:12]}",
        "",
        "## Nominal trial per task (layer 1)",
        "",
        "| task | ok | s | build | floor | settle/restore | task | awake end / trees |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, r in rep["nominal"].items():
        if r.get("error"):
            L.append(f"| {i} | ERROR {r['error'][:60]} |||||||")
            continue
        t = r["trials"][0]
        s = t["stages"]
        L.append(
            f"| {i} | {t['ok']} | {t['seconds']} | {s.get('build_s', 0):.1f} | {s.get('floor_s', 0):.1f} | {s.get('settle_s', 0) + s.get('restore_s', 0):.1f} | {s.get('task_s', 0):.1f} | {t.get('awake_end')} / {t.get('ntree')} |"
        )
    L += [
        "",
        "## Zone settle of a changed trial per task (layer 1: an edited or nudged trial's; measured with a 1 mm jitter)",
        "",
        "| task | zone m | awake start / max / end | settle s | woke on contact | zone never moved |",
        "|---|---|---|---|---|---|",
    ]
    for i, z in rep["zones"].items():
        if "error" in z:
            L.append(f"| {i} | ERROR {z['error'][:80]} |||||")
            continue
        L.append(
            f"| {i} | {z['zone_m']} | {z['awake_trees_start']} / {z['awake_trees_max']} / {z['awake_trees_end']} | {z['settle_wall_s']} | {len(z['woke_on_contact'])} {z['woke_on_contact'][:5]} | {len(z['zone_never_moved'])} |"
        )
    L += [
        "",
        "## Top cost contributors",
        "",
        f"- nominal-trial stage seconds (summed): {top['nominal_stages']}",
        "- whole-room settle, by category (modelled share from contacts and constraint rows): "
        + ", ".join(f"{x['category']} {x['share']:.1%} (n={x['n']})" for x in top["settle_categories"]),
        "- objects: " + ", ".join(f"{x['object']} {x['share']:.1%}" for x in top["settle_objects"][:8]),
        "- contact pairs by geom type: " + ", ".join(f"{k} {v}" for k, v in top["pairs"].items()),
        "- largest collision hulls: "
        + ", ".join(f"{x['object']} {x['hull_verts']}v/{x['geoms']}g" for x in top["hulls"][:6]),
        "",
        "## Calibration of the robot tasks (modelled)",
        "",
        f"- plain {c['plain_s']} s, layer 1 {c['l1_s']} s ({c['note']})",
        "",
    ]
    return "\n".join(L) + "\n"
