"""Calibrate F3 units on the reference room: actuation forces for every door and drawer, the objects that actually
hold a dropped ball (containers), grasp brackets for small objects, and a clear spot for the load test on tops."""

import numpy as np
from sbv.functionality import f3_functions as Fn, f2_calibrate as C
from sbv.common.scene import Unstable


def cal_joint(scene, n, j):
    """Returns (unit, None) or (None, reason). The unit needs every component to pass on the reference."""
    rec = Fn.joint_record(scene, n, j)
    if not np.isfinite(rec["travel"]) or rec["travel"] < 1e-3:
        return None, f"joint {rec['name']} has no finite travel"
    clearance = Fn.sweep_report(scene, j, rec["sgn"], rec["travel"], None)
    if clearance["status"] != "clear":
        return None, f"reference joint {rec['name']} clearance: {clearance}"
    owner = scene.objects[n]
    reach = lambda F, T: Fn.drive(scene, j, rec["sgn"], F, T, None, owner)[0]
    try:
        F = C.bisect(lambda F: reach(F, Fn.ACT_T) >= Fn.ACT_FRAC * rec["travel"], 0.01, 1000.0, 14, log=True)
        Ff = C.bisect(lambda F: reach(F, Fn.FULL_T) >= Fn.FULL_FRAC * rec["travel"], 0.01, 1000.0, 14, log=True)
    except Unstable:
        return None, f"joint {rec['name']}: unstable when driven"
    if F is None or Ff is None:
        return None, f"joint {rec['name']} does not open between 0.01 and 1000 (N or N m)"
    unit = dict(
        id=f"joint|{n}|{rec['name']}",
        kind="joint",
        slots=[n],
        hint_components=Fn.HINT_COMPONENTS,
        params=dict(ref=rec, F=round(F, 5), F_full=round(Ff, 5), bars=list(C.P.BARS)),
        expect=dict(rest=None),
    )
    obs, mons = Fn.joint_trial(scene, unit)
    bad = [k for k in Fn.COMPONENTS if k not in ("rest", "actuation") and not obs[k]]
    if not obs["actuation"] or any(a != [False, True] for a in obs["actuation"]):
        bad.append("actuation")  # at every bar
    if bad or obs["rest"] is None:
        return None, f"reference joint {rec['name']} fails {bad or ['rest']}"
    if any(mo.crossed(mo.owners) for mo in mons):
        return None, f"reference joint {rec['name']} crosses a barrier when driven"
    if any(mo.report(mo.owners).get("sweep_unresolved") for mo in mons):
        return None, "reference joint motion check is unresolved"
    unit["expect"]["rest"] = obs["rest"]
    unit["tol"] = round(max(0.01, 1.5 * max(max(mo.pen.values()) for mo in mons) + 0.002), 4)
    rec["handle"] = Fn.has_handle(scene, n, j)  # a copy's part must have one when the true part does
    return unit, None


def cal_holds(scene, n, items):
    """A unit for what the reference room declares container n holds (functions.json); (unit, None) or (None,
    reason)."""
    unit = dict(id=f"holds|{n}|{'+'.join(items)}", kind="holds", slots=[n] + list(items), params={}, named=True)
    why = C.finalize(scene, unit, [True] * len(items), run=Fn.trial)
    return (None, f"declared, but the reference room fails it: {why}") if why else (unit, None)


def cal_opening(scene, n, k, op):
    """A unit for the k-th opening the reference room declares on container n (functions.json); (unit, None) or (None,
    reason). The opening, given in the container's own frame, is frozen in room axes beside a sample of the container's
    collision surface, so a copy can be lined up with it whatever body frame the copy uses."""
    unit = dict(id=f"opening|{n}|{k}", kind="opening", slots=[n], params=frozen_opening(scene, n, op), named=True)
    why = C.finalize(scene, unit, [True], run=Fn.trial)
    return (None, f"declared, but the reference room fails it: {why}") if why else (unit, None)


def frozen_opening(scene, n, op):
    """An opening given in the container's own frame, frozen in room axes beside a sample of its collision surface."""
    from sbv.common.align import collision_points

    scene.reset()
    b = scene.objects[n]
    R, o = scene.d.xmat[b].reshape(3, 3), scene.d.xpos[b]
    opening = dict(pos=(R @ op["pos"]).round(4).tolist(), dir=(R @ op["dir"]).round(4).tolist(), radius=op["radius"])
    return dict(opening=opening, key=(collision_points(scene, n, 300) - o).round(4).tolist())


TURN_KG, TURN_M = 5.0, 0.6  # F3.6 turns over what one hand turns over: at most 5 kg and 60 cm across


def turnable(scene, n):
    scene.reset()
    lo, hi = scene.aabb(n)
    return (
        scene.free_dof(n) is not None
        and max((hi - lo)[:2]) <= TURN_M
        and scene.m.body_subtreemass[scene.objects[n]] <= TURN_KG
    )


def cal_turnover(scene, n, items=(), k=None, op=None):
    """F3.6 on the reference room: n turned upside down, with the free objects resting on it (items) or the ball put in
    through its k-th declared opening (op). The reference room's outcome, item by item, is the expected one; (unit,
    None) or (None, reason)."""
    if not turnable(scene, n):
        return None, f"{n} is fixed, wider than {TURN_M} m or heavier than {TURN_KG} kg"
    params = {} if op is None else frozen_opening(scene, n, op)
    uid = f"turnover|{n}|" + (f"opening{k}" if op is not None else "+".join(items))
    unit = dict(id=uid, kind="turnover", slots=[n] + list(items), params=params, named=op is not None)
    why = C.finalize(scene, unit, None, run=Fn.trial)
    if why is None and not isinstance(unit.get("expect"), list):
        why = f"reference outcome: {unit.get('expect')}"
    return (None, f"the reference room fails it: {why}") if why else (unit, None)


def cal_grasp(scene, n):
    scene.reset()
    lo, hi = scene.aabb(n)
    ext = hi - lo
    if (
        scene.free_dof(n) is None
        or min(ext[:2]) > 0.12
        or ext[2] < 0.03
        or scene.m.body_subtreemass[scene.objects[n]] > 5
    ):
        return None
    if Fn.lift_once(scene, n, 60.0, None, check_clear=True)[0] is not True:
        return None
    F = C.bisect(lambda F: Fn.lift_once(scene, n, F, None)[0], 0.3, 60.0, 7, log=True)
    if F is None:
        return None
    unit = dict(id=f"grasp|{n}", kind="grasp", slots=[n], params=dict(F=round(F, 4), bars=list(C.P.BARS)))
    return None if C.finalize(scene, unit, [False, True] * len(C.P.BARS), run=Fn.trial) else unit


def cal_load(scene, n):
    scene.reset()
    lo, hi = scene.aabb(n)
    c = (lo + hi) / 2
    ext = hi - lo
    top = float(hi[2])
    if not 0.2 <= top <= 2.0 or min(ext[:2]) < 0.25:
        return None
    for dx, dy in [
        (0, 0),
        (0.1, 0),
        (-0.1, 0),
        (0, 0.1),
        (0, -0.1),
        (0.15, 0.15),
        (-0.15, 0.15),
        (0.15, -0.15),
        (-0.15, -0.15),
    ]:
        spot = c[:2] + [dx, dy]
        if all(C.on_top(scene, n, spot + [sx * 0.07, sy * 0.07], top) for sx in (-1, 0, 1) for sy in (-1, 0, 1)):
            unit = dict(id=f"load|{n}", kind="load", slots=[n], params=dict(rel=[dx, dy]))
            return None if C.finalize(scene, unit, [True], run=Fn.trial) else unit
    return None
