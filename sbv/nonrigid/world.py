"""One non-rigid object in Genesis (CPU), with just the part of the room around it, under one hidden action.

The room part comes unchanged from the copy's scene.xml (room.py): the bodies near the object, the room geometry, and
the contact excludes between them. Every body is held still except a fluid's container, which the action moves.
"""

import os
import time

import numpy as np

from .actions import BALL_R, _fluid, _lift, _nudge, _press
from .room import _aabbs, cut, neighbourhood, prune
from .schema import FABRIC, FIXED, HELD_SDF_MAX_RES, SETTLE_MU, VISCOSITY, bounds, center

BALL_KG = 1.0
# cloth must start at rest (SX.7). It first settles into contact for SETTLE_S, its speed set to zero after every step:
# Genesis's cloth particles are 15 mm across, so cloth lying on a bench or hanging on a hanger, as the room rules allow
# (overlaps up to 2 mm), starts up to half a particle inside it, and Genesis throws it clear (a towel lying on a bench
# jumped 12 cm at 18 m/s). Then no particle may pass START_SPEED in the next START_S. Falling alone reaches g * START_S
# (0.98 m/s), and a hanging garment let go after settling a little more: at 1.0 m/s it was refused in 3 runs of 8), a
# towel 0.49, a wool arch 0.43. A mesh crossing itself is still being pushed apart: three crossing sheets 3.1 m/s, a
# crumpled wipe 5.3 m/s. Cloth actions start at 1 s or later (verify.RANGES), after both.
SETTLE_S, START_S, START_SPEED = 0.05, 0.1, 1.5


class Unsettled(Exception):
    """A cloth moved faster than falling allows in the START_S after settling."""


def cloth_domain(rail, blo, bhi, draw=None, axis=None):
    """Include the actual garment, even when its authored rail point is wrong.

    Genesis enforces its domain as a physical wall. Clipping a garment against
    an undersized box can otherwise look like a stable, motionless cloth.
    """
    rail = np.asarray(rail, dtype=float)
    lo = np.minimum(rail - [0.6, 0.4, rail[2] + 0.1], np.asarray(blo) - 0.1)
    hi = np.maximum(rail + [0.6, 0.6, 0.3], np.asarray(bhi) + 0.1)
    if draw:
        axis = np.asarray(axis, float)
        axis /= np.linalg.norm(axis)
        shifts = np.r_[0.0, np.asarray(draw["offsets"], float)][:, None] * axis
        lo = np.minimum(lo, np.asarray(blo) + shifts.min(0) - 0.1)
        hi = np.maximum(hi, np.asarray(bhi) + shifts.max(0) + 0.1)
    return lo, hi


def touches_domain(points, lower, upper):
    points = np.asarray(points)
    return bool(
        len(points) and (np.any(points <= np.asarray(lower) + 1e-6) or np.any(points >= np.asarray(upper) - 1e-6))
    )


def run(scene_xml, obj, action, work, video=None, scale=None, look=False):
    """Run one hidden action on one object. `scale` multiplies one material number (calibration's sensitivity runs)."""
    import genesis as gs

    if (
        obj["kind"] == "thin_shell"
    ):  # Genesis's cloth gets no friction from rigid bodies without this (see genesis_patch)
        from .genesis_patch import cloth_rigid_friction

        cloth_rigid_friction()

    os.makedirs(work, exist_ok=True)
    kind, s, mat, fx = (
        obj["kind"],
        obj["shape"],
        dict(obj["material"]),
        {**FIXED[obj["kind"]], **action.get("fixed", {})},
    )  # a unit may coarsen its particles (fixed)
    draw = action.get("ring_draw")
    if obj.get("attachments"):
        from .attachments import problems as attachment_problems

        why = attachment_problems(obj)
        if why:
            return dict(error="; ".join(why))
    if draw:
        if not obj.get("attachments"):
            return dict(error="ring_draw requires declared clip attachments")
        try:
            offsets = np.asarray(draw["offsets"], float)
            valid = (
                offsets.shape == (obj["attachments"]["count"],)
                and np.isfinite(offsets).all()
                and float(draw.get("ramp_s", 0)) > 0
                and float(draw.get("start_s", 0)) >= 0
            )
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            return dict(error="ring_draw requires finite offsets per clip, nonnegative start_s and positive ramp_s")
    if obj["on"] != "floor" and obj["on"] not in _aabbs(scene_xml):
        return dict(error=f"'{obj['on']}' is not a rigid body in scene.xml")
    keep, boxes = neighbourhood(scene_xml, obj)
    actor = obj["on"] if kind == "fluid" else None
    files = cut(
        scene_xml, keep, actor, work, looks=look, hide=(obj["name"] + "_look", obj["name"] + "_contents_look")
    )  # look: the room's own meshes and colours in the video
    folder = os.path.dirname(os.path.abspath(scene_xml))
    blo, bhi = (np.asarray(v, float) for v in bounds(obj, folder))
    c0 = np.array(s["rail"], float) if kind == "thin_shell" else center(obj, folder)
    if kind == "fluid":  # room for the spill around the holder (the size no longer costs: see hash_grid)
        base = c0[2] - s["height"] / 2
        lo, hi = (
            np.array([c0[0] - 0.45, c0[1] - 0.45, base - 0.12]),
            np.array([c0[0] + 0.45, c0[1] + 0.45, base + s["height"] + 0.25]),
        )
    elif kind == "soft_body":  # the object, the ball's fall, a little around: MPM pays for every grid cell
        lo, hi = blo - [0.12, 0.12, 0.06], bhi + [0.12, 0.12, action["height"] + 2 * BALL_R + 0.05]
    else:
        lo, hi = cloth_domain(c0, blo, bhi, action.get("ring_draw"), obj.get("attachments", {}).get("axis"))
    if kind == "fluid" and not look:
        prune(files["still"], lo, hi)
    pad = 3 / 64 if fx["solver"] == "MPM" else 0.0
    opt = dict(lower_bound=tuple(lo - pad), upper_bound=tuple(hi + pad), particle_size=fx["particle_size"])
    kw = dict(
        sim_options=gs.options.SimOptions(dt=fx["dt"], substeps=fx["substeps"]),
        show_viewer=False,
        rigid_options=gs.options.RigidOptions(
            max_collision_pairs=10000, enable_collision=kind != "fluid"
        ),  # a busy desk overflowed the default buffer (B1, v2); a fluid run moves or holds every rigid body itself
        vis_options=gs.options.VisOptions(show_world_frame=False, ambient_light=(0.4, 0.4, 0.4)),
    )
    kw[{"SPH": "sph_options", "MPM": "mpm_options", "PBD": "pbd_options"}[fx["solver"]]] = (
        gs.options.SPHOptions(**opt, hash_grid_res=(fx["hash_grid"],) * 3)
        if fx["solver"] == "SPH"
        else gs.options.MPMOptions(**opt, grid_density=fx["grid_density"])
        if fx["solver"] == "MPM"
        else gs.options.PBDOptions(**opt)
    )
    sc = gs.Scene(**kw)
    held = dict(vis_mode="visual" if look else "collision", material=gs.materials.Rigid(sdf_max_res=HELD_SDF_MAX_RES))
    sc.add_entity(gs.morphs.MJCF(file=files["still"]), **held)  # videos show what the physics uses
    cont = sc.add_entity(gs.morphs.MJCF(file=files["actor"]), **held) if actor else None
    k = scale or {}
    if kind == "fluid":
        mu = VISCOSITY[mat["viscosity"]] * k.get("viscosity", 1.0)
        ent = sc.add_entity(
            material=gs.materials.SPH.Liquid(
                mu=max(mu, SETTLE_MU), rho=mat.get("density", 1000.0), sampler=fx["sampler"]
            ),
            morph=gs.morphs.Cylinder(
                pos=(s["pos"][0], s["pos"][1], s["pos"][2] + s["height"] * (k.get("amount", 1.0) - 1) / 2),
                radius=s["radius"],
                height=s["height"] * k.get("amount", 1.0),
            ),  # more or less fluid from the same base
            surface=gs.surfaces.Default(color=(0.35, 0.2, 0.1, 1.0)),
        )
    elif kind == "soft_body":
        ent = sc.add_entity(
            material=gs.materials.MPM.Elastic(
                E=mat["youngs_kpa"] * 1e3 * k.get("youngs_kpa", 1.0),
                nu=mat.get("poisson", 0.3),
                rho=mat["density"],
                sampler=fx["sampler"],
            ),
            morph=gs.morphs.Box(pos=tuple(s["pos"]), size=tuple(s["size"]), euler=(0, 0, s.get("yaw_deg", 0)))
            if s["type"] == "box"
            else gs.morphs.Mesh(
                file=os.path.join(folder, s["file"]), pos=tuple(s["pos"]), euler=(0, 0, s.get("yaw_deg", 0))
            ),
            surface=gs.surfaces.Default(
                color=(0.55, 0.65, 0.5, 1.0), vis_mode="particle" if s["type"] == "mesh" else "visual"
            ),
        )
        top = bhi[2]
        spot = c0[:2] + np.array(action["offset"]) * (bhi - blo)[:2] / 2
        if action.get("tool", "ball") == "ball":  # a 1 kg ball dropped on the spot
            half_z = BALL_R
            ball = sc.add_entity(
                gs.morphs.Sphere(pos=(spot[0], spot[1], top + action["height"] + BALL_R), radius=BALL_R),
                material=gs.materials.Rigid(rho=BALL_KG / (4 / 3 * np.pi * BALL_R**3)),
                surface=gs.surfaces.Default(color=(0.85, 0.3, 0.2, 1.0)),
            )
        else:  # a flat block laid on top (cannot roll off a narrow object)
            bs, kg = action["block"], action["block_kg"]
            half_z = bs[2] / 2
            ball = sc.add_entity(
                gs.morphs.Box(
                    pos=(spot[0], spot[1], top + action["height"] + half_z),
                    size=tuple(bs),
                    euler=(0, 0, s.get("yaw_deg", 0)),
                ),
                material=gs.materials.Rigid(rho=kg / (bs[0] * bs[1] * bs[2])),
                surface=gs.surfaces.Default(color=(0.85, 0.3, 0.2, 1.0)),
            )
    else:
        from .schema import sheet

        mesh = os.path.join(work, "sheet.obj")
        cloth = sheet(obj, folder)
        cloth.export(mesh)  # a draped sheet from its numbers, or the package's garment mesh as placed
        rho, bend = FABRIC[k.get("fabric", mat["fabric"])]  # calibration swaps the fabric
        rho *= k.get("areal_density", 1.0)
        ent = sc.add_entity(
            material=gs.materials.PBD.Cloth(
                rho=rho, bending_compliance=bend, static_friction=fx["friction"], kinetic_friction=fx["friction"] - 0.1
            ),
            morph=gs.morphs.Mesh(file=mesh),
            surface=gs.surfaces.Default(color=(0.25, 0.3, 0.45, 1.0)),
        )
    cam = None
    if video:
        cam = sc.add_camera(
            res=(480, 480),
            pos=tuple(c0 + np.array(action.get("camera", (0.9, -0.9, 0.6)))),
            lookat=tuple(c0),
            fov=45,
            GUI=False,
        )
    t_build = time.time()
    sc.build()
    if cam:
        cam.start_recording(save_to_filename=video, fps=25)
    P = lambda: (
        ent.get_particles_pos().cpu().numpy()
        if hasattr(ent.get_particles_pos(), "cpu")
        else np.asarray(ent.get_particles_pos())
    ).reshape(-1, 3)
    initial = P()
    if not len(initial) or not np.isfinite(initial).all():
        return dict(error="non-rigid import has empty or non-finite particles")
    track = None
    if obj.get("attachments"):
        from .attachments import RingTrack

        try:
            track = RingTrack(ent, initial, scene_xml, obj)
        except ValueError as e:
            return dict(error=str(e))
    if touches_domain(initial, opt["lower_bound"], opt["upper_bound"]):
        return dict(error="initial non-rigid geometry reaches the artificial solver boundary")
    observed = dict(boundary_contact=False, finite=True, steps=0, start_speed=0.0, last=initial)
    trace = [initial.copy()] if os.environ.get("NONRIGID_PARTICLE_TRACE") else None
    trace_times = [0.0]
    native_step = sc.step
    n_settle, n_start = int(round(SETTLE_S / fx["dt"])), int(round(START_S / fx["dt"]))

    def step():
        if track:
            velocity = ent.get_particles_vel()
            velocity = velocity.cpu().numpy() if hasattr(velocity, "cpu") else np.asarray(velocity)
            offsets = None
            if draw:
                u = np.clip((observed["steps"] * fx["dt"] - draw.get("start_s", 0)) / draw["ramp_s"], 0, 1)
                offsets = np.asarray(draw["offsets"], float) * (u * u * (3 - 2 * u))
            track.apply(P(), velocity.reshape(-1, 3), offsets, dt=fx["dt"])
        native_step()
        points = P()
        observed["steps"] += 1
        observed["finite"] &= bool(np.isfinite(points).all())
        observed["boundary_contact"] |= touches_domain(points, opt["lower_bound"], opt["upper_bound"])
        if (
            kind == "thin_shell" and observed["steps"] <= n_settle
        ):  # settling: keep where contact puts it, not its speed
            ent.set_particles_vel(np.zeros_like(points))
        elif kind == "thin_shell" and observed["steps"] <= n_settle + n_start:
            observed["start_speed"] = max(
                observed["start_speed"], float(np.linalg.norm(points - observed["last"], axis=1).max() / fx["dt"])
            )
            if observed["start_speed"] > START_SPEED:
                raise Unsettled()
        observed["last"] = points
        if trace is not None and observed["steps"] % max(1, round(0.1 / fx["dt"])) == 0:
            trace.append(points.copy())
            trace_times.append(observed["steps"] * fx["dt"])

    sc.step = step
    steps = int(round(action["duration"] / fx["dt"]))
    out, wall0 = dict(kind=kind, build_s=round(time.time() - t_build, 1)), time.time()
    if kind == "fluid":
        from .interior import held_test, inside  # held means inside the container's real inside, not its outline

        within = held_test(*inside(scene_xml, obj["on"], np.array(s["pos"]) - [0, 0, s["height"] / 2 - 0.004]))
        info = sc.sim.sph_solver.particles_info  # this run's only fluid: its own viscosity from the action's start

        def thin():
            m = info.mu.to_numpy()
            m[:] = mu
            info.mu.from_numpy(m)

        out.update(_fluid(sc, cont, P, action, boxes[obj["on"]], steps, fx["dt"], within, thin=thin))
    elif kind == "soft_body":
        out.update(
            _press(
                sc,
                ball,
                P,
                action,
                spot,
                top,
                boxes.get(obj["on"], [np.zeros(3), np.zeros(3)]),
                steps,
                fx["dt"],
                half_z,
            )
        )
    else:
        try:
            if "force_n" in action:  # the lift; a case's older swing reads its own action
                out.update(_lift(sc, ent, P, action, s, steps, fx["dt"], rho * cloth.area / len(initial)))
            else:
                out.update(_nudge(sc, ent, P, action, s, rho, steps, fx["dt"]))
        except Unsettled:
            return dict(
                error=f"cloth is not at rest: after settling {SETTLE_S} s, a particle reached {observed['start_speed']:.2f} m/s "
                f"within {START_S} s, faster than falling allows ({START_SPEED} m/s); its mesh crosses itself, "
                f"which the solver keeps pushing apart",
                start_speed_mps=round(observed["start_speed"], 2),
            )
    if cam:
        cam.stop_recording()
    if os.environ.get("NONRIGID_SAVE_END"):  # for checks: where every particle ended
        np.save(os.environ["NONRIGID_SAVE_END"], P())
    if trace is not None:
        np.savez_compressed(
            os.environ["NONRIGID_PARTICLE_TRACE"], times=np.asarray(trace_times), points=np.asarray(trace)
        )
    out["wall_s"] = round(time.time() - wall0, 1)
    if track:
        out["attachments"] = track.report(P())
    out.update(
        solver_boundary_contact=observed["boundary_contact"],
        finite=bool(out.get("finite", True) and observed["finite"]),
        solver_domain=dict(lower=list(opt["lower_bound"]), upper=list(opt["upper_bound"])),
        domain_check_steps=observed["steps"],
        **({"start_speed_mps": round(observed["start_speed"], 2)} if kind == "thin_shell" else {}),
        domain_check_scope="every physics step; not continuous substep certification",
    )
    return out
