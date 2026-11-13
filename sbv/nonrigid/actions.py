"""The hidden actions and what each one reads out: tilt a container, drop a ball on a non-rigid object, pull a cloth up
(nudging it, the retired swing, stays for older cases)."""

import numpy as np

BALL_R = 0.05
REST_WINDOW = 0.2  # stillness is the fluid's net movement over the last 0.2 s before the action: over one


def _fluid(sc, cont, P, a, box, steps, dt, within, thin=None):
    """`within(points)`: True for points inside the container's real inside, in its starting frame (interior.py).
    `thin()` gives the fluid its own viscosity once it has settled, at a["t_start"]."""
    from scipy.spatial.transform import Rotation as R

    q0 = cont.get_qpos().cpu().numpy().reshape(-1).copy()  # x y z, qw qx qy qz
    lo, hi = box
    c = (lo + hi) / 2
    yaw = np.radians(a["yaw_deg"])
    side = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    pivot = (
        np.array([c[0], c[1], lo[2]]) + side * (abs(side[0]) * (hi[0] - lo[0]) + abs(side[1]) * (hi[1] - lo[1])) / 2
    )  # the base edge it tips toward
    axis = np.cross([0, 0, 1.0], side)
    held, rest_h, rest_v, base_q = None, None, None, R.from_quat([q0[4], q0[5], q0[6], q0[3]])
    early, t_early = None, None
    for i in range(steps):
        t = (i + 1) * dt
        if (
            early is None and t - dt >= a["t_start"] - REST_WINDOW - 1e-9
        ):  # the positions REST_WINDOW before the rest read
            early, t_early = P(), t - dt
        ang = np.radians(a["angle_deg"]) * min(1.0, max(0.0, (t - a["t_start"]) / a["t_ramp"]))
        rot = R.from_rotvec(axis * ang)
        pos = pivot + rot.apply(q0[:3] - pivot)
        qx = (rot * base_q).as_quat()
        w = np.radians(a["angle_deg"]) / a["t_ramp"] if a["t_start"] <= t < a["t_start"] + a["t_ramp"] else 0.0
        vel = np.concatenate([np.cross(axis * w, pos - pivot), axis * w])
        cont.set_qpos(np.concatenate([pos, [qx[3], qx[0], qx[1], qx[2]]]))
        cont.set_dofs_velocity(vel)
        sc.step()
        if held is None and t >= a["t_start"] - 1e-9:
            if thin:
                thin()
            p = P()
            held = float(within(p).mean())
            rest_h = float(
                np.percentile(p[:, 2], 99) - np.percentile(p[:, 2], 1)
            )  # the fluid's height just before the tilt
            rest_v = float(
                np.linalg.norm(p - early, axis=1).mean() / max(t - t_early, dt)
            )  # net movement over the window, m/s
    p = P()
    local = rot.inv().apply(p - pivot) + pivot  # particles in the container's starting frame
    inside = within(local)
    return dict(
        readout=float(1 - inside.mean()),
        held_at_rest=held,
        rest_speed=rest_v,
        rest_height=rest_h,
        lowest_z=float(p[:, 2].min()),
        finite=bool(np.isfinite(p).all()),
        base_z=float(lo[2]),
    )


def _press(sc, ball, P, a, spot, top0, support, steps, dt, half_z=BALL_R):
    near = lambda p: p[np.linalg.norm(p[:, :2] - spot, axis=1) < 0.03]
    top, low, lowest = None, 1e9, 1e9
    for i in range(steps):
        sc.step()
        z = float(ball.get_pos().cpu().numpy().reshape(-1)[2]) - half_z
        p = P()
        if top is None and z < top0 + 0.08:  # just before contact: the settled top under the drop spot
            q = near(p)
            top = float(q[:, 2].max()) if len(q) else top0
        low, lowest = min(low, z), min(lowest, float(p[:, 2].min()))
    p = P()
    return dict(
        readout=(top if top is not None else top0) - low,
        ball_lowest=low,
        object_lowest=lowest,
        support_top=float(support[1][2]),
        finite=bool(np.isfinite(p).all()),
    )


def _nudge(sc, ent, P, a, s, rho, steps, dt):
    rail = np.array(s["rail"])
    settle = int(round(a["t_start"] / dt))
    rest, hem, disp = None, None, 0.0
    for i in range(steps):
        if i == settle:
            p = P()
            front = np.where(p[:, 1] > rail[1])[0]
            rest = float(p[front, 2].min())
            hem = front[p[front, 2] < rest + 0.2]
            v = (
                ent.get_particles_vel().cpu().numpy()
                if hasattr(ent.get_particles_vel(), "cpu")
                else np.asarray(ent.get_particles_vel())
            ).reshape(-1, 3)
            v[hem] += (
                np.array([np.sin(np.radians(a["yaw_deg"])), np.cos(np.radians(a["yaw_deg"])), 0]) * a["impulse"] / rho
            )
            ent.set_particles_vel(v)
            h0 = p[hem, :2].mean(0)
        sc.step()
        if hem is not None:
            disp = max(disp, float(np.linalg.norm(P()[hem, :2].mean(0) - h0)))
    p = P()
    return dict(
        readout=disp,
        rest_hem_z=rest,
        top_z=float(p[:, 2].max()),
        rail_z=float(rail[2]),
        finite=bool(np.isfinite(p).all()),
    )


G = 9.81
EDGE = 0.03  # the strip that is pulled: the lowest 3 cm of a hung cloth, or a lying cloth's 3 cm edge toward yaw_deg


def _lift(sc, ent, P, a, s, steps, dt, particle_kg):
    """Pull one strip of a settled cloth straight up with a constant force, a["force_n"] newtons spread over the strip,
    from a["t_start"]: the strip rises until the cloth it lifts weighs as much, so the rise follows the cloth's weight
    per square metre. A cloth more than 15 cm tall hangs: its hem is pulled; a lower one lies: its edge across the axis
    of its own nearest yaw_deg is."""
    start = int(round(a["t_start"] / dt))
    strip, z0, rise = None, None, 0.0
    for i in range(steps):
        if i == start:
            p = P()
            if np.ptp(p[:, 2]) > 0.15:
                strip = np.where(p[:, 2] < p[:, 2].min() + EDGE)[0]
            else:
                xy = (
                    p[:, :2] - p[:, :2].mean(0)
                )  # the edge across its own axis nearest yaw_deg: a turn of a few degrees takes the same edge, not a corner
                axes = np.linalg.eigh(np.cov(xy.T))[1].T
                d = max(
                    [*axes, *-axes],
                    key=lambda v: v @ [np.cos(np.radians(a["yaw_deg"])), np.sin(np.radians(a["yaw_deg"]))],
                )
                along = xy @ d
                strip = np.where(along > along.max() - EDGE)[0]
            z0, dv = float(p[strip, 2].mean()), a["force_n"] / len(strip) / particle_kg * dt
        if strip is not None:
            v = ent.get_particles_vel()
            v = (v.cpu().numpy() if hasattr(v, "cpu") else np.asarray(v)).reshape(-1, 3)
            v[strip, 2] += dv
            ent.set_particles_vel(v)
        sc.step()
        if strip is not None:
            rise = max(rise, float(P()[strip, 2].mean()) - z0)
    p = P()
    return dict(
        readout=rise,
        strip=len(strip),
        top_z=float(p[:, 2].max()),
        rail_z=float(s["rail"][2]),
        finite=bool(np.isfinite(p).all()),
    )
