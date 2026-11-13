"""Ideal sliding ring clips for hanging cloth, shared by native rest runs and package actions.

The declared rigid support must supply a collidable capsule/cylinder rod. Only small header patches are
kinematic; the rest remains PBD cloth. Uncommanded clips hold their position (ideal track stiction).
This is a prescribed clip boundary, not passive ring dynamics or a robot grasp. No particle is teleported.
"""

import numpy as np


def problems(obj):
    a = obj.get("attachments")
    if a is None:
        return []
    if not isinstance(a, dict):
        return ["attachments must be an object"]
    out = []
    if obj.get("kind") != "thin_shell" or a.get("type") != "sliding_ring_clips":
        out.append("attachments supports only cloth sliding_ring_clips")
    if type(a.get("count")) is not int or not 2 <= a["count"] <= 64:
        out.append("sliding_ring_clips count must be 2..64")
    try:
        axis = np.asarray(a.get("axis", []), float)
    except (ValueError, TypeError):
        axis = np.array([])
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-8:
        out.append("sliding_ring_clips axis must be a finite nonzero 3-vector")
    return out


def rod(scene_xml, obj):
    """(origin, axis, half length) of the longest collidable capsule or cylinder on the support obj["on"] (its body or
    below); raises ValueError when there is none. RingTrack binds to it; the non-rigid verifiers' buildability check
    asks it first."""
    import mujoco as mj

    m = mj.MjModel.from_xml_path(str(scene_xml))
    d = mj.MjData(m)
    mj.mj_forward(m, d)
    b = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, obj["on"])
    if b < 0:
        raise ValueError("ring clips have no mapped rigid support")
    bodies = {b}
    for i in range(b + 1, m.nbody):
        if m.body_parentid[i] in bodies:
            bodies.add(i)
    rods = [
        g
        for g in range(m.ngeom)
        if m.geom_bodyid[g] in bodies
        and (m.geom_contype[g] or m.geom_conaffinity[g])
        and int(m.geom_type[g]) in (int(mj.mjtGeom.mjGEOM_CAPSULE), int(mj.mjtGeom.mjGEOM_CYLINDER))
    ]
    if not rods:
        raise ValueError("ring clips require a collidable capsule/cylinder rod")
    g = max(rods, key=lambda g: m.geom_size[g, 1])
    return d.geom_xpos[g].copy(), d.geom_xmat[g].reshape(3, 3)[:, 2].copy(), float(m.geom_size[g, 1])


def header(points, origin, axis, half, count):
    """(patches, starts, gaps) of a cloth's header (its points within 2.5 cm of its top), one patch per clip spread
    along the rod; raises ValueError when a patch is over 8 cm from the rod axis or beyond its half-length + 2 cm.
    RingTrack applies it to the cloth's particles; the non-rigid verifiers' buildability check to the shape's own
    vertices first."""
    q = np.asarray(points)
    top = np.flatnonzero(q[:, 2] > q[:, 2].max() - 0.025)
    along = (q[top] - origin) @ axis
    patches = []
    for u in np.linspace(along.min(), along.max(), count):
        near = top[np.abs(along - u) <= np.abs(along - u).min() + 0.001]
        centre = q[near].mean(0)
        distances = np.linalg.norm(q[top] - centre, axis=1)
        patches.append(top[distances <= distances.min() + 0.004])
    starts = np.array([q[ix].mean(0) for ix in patches])
    rel = starts - origin
    along = rel @ axis
    gap = np.linalg.norm(rel - along[:, None] * axis, axis=1)
    if gap.max() > 0.08 or np.max(np.abs(along)) > half + 0.02:
        raise ValueError("cloth header is not within 8 cm of its rod/clip support")
    return patches, starts, gap


class RingTrack:
    """Bind header patches to an actual mapped rod; reject unsupported or badly placed declarations."""

    def __init__(self, entity, points, scene_xml, obj):
        import mujoco as mj

        why = problems(obj)
        if why:
            raise ValueError("; ".join(why))
        self.origin, self.axis, half = rod(scene_xml, obj)
        declared = np.asarray(obj["attachments"]["axis"], float)
        declared /= np.linalg.norm(declared)
        if abs(self.axis @ declared) < 0.99:
            raise ValueError("clip axis is not parallel to the rigid rod")
        if self.axis @ declared < 0:
            self.axis *= -1
        self.half = half
        self.entity = entity
        self.patches, self.starts, gap = header(points, self.origin, self.axis, half, obj["attachments"]["count"])
        self.ids = np.unique(np.concatenate(self.patches))
        entity.fix_particles(self.ids)
        self.initial_gap = float(gap.max())

    def apply(self, points, velocities, offsets=None, dt=0.002, max_speed=0.35, max_accel=5.0):
        """Before each physics step. Offsets are one scalar along the rod per clip; None holds clips still."""
        v = np.asarray(velocities).copy()
        if offsets is None:
            v[self.ids] = 0
        else:
            offsets = np.asarray(offsets, float)
            if offsets.shape != (len(self.patches),) or not np.isfinite(offsets).all():
                raise ValueError("one finite track offset is required per clip")
            targets = self.starts + offsets[:, None] * self.axis
            if np.max(np.abs((targets - self.origin) @ self.axis)) > self.half + 0.02:
                raise ValueError("clip target passes beyond its actual rod")
            for ix, target in zip(self.patches, targets):
                want = (target - points[ix].mean(0)) * 15
                want *= min(1.0, max_speed / max(np.linalg.norm(want), 1e-9))
                dv = want - v[ix]
                v[ix] += dv * np.minimum(
                    1.0, max_accel * dt / np.maximum(np.linalg.norm(dv, axis=1, keepdims=True), 1e-9)
                )
        self.entity.set_particles_vel(v[self.ids], particles_idx_local=self.ids)

    def report(self, points):
        centres = np.array([points[ix].mean(0) for ix in self.patches])
        rel = centres - self.origin
        along = rel @ self.axis
        return dict(
            type="sliding_ring_clips",
            clips=len(self.patches),
            particles=len(self.ids),
            header_shift_along_m=((centres - self.starts) @ self.axis).tolist(),
            max_support_gap_m=float(np.linalg.norm(rel - along[:, None] * self.axis, axis=1).max()),
            max_header_normal_drift_m=float(
                np.linalg.norm(
                    (centres - self.starts) - ((centres - self.starts) @ self.axis)[:, None] * self.axis, axis=1
                ).max()
            ),
            initial_support_gap_m=self.initial_gap,
        )
