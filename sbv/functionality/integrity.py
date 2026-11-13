"""Dynamic integrity checks (doc Appendix C.3) that run inside every F2, F3 and F4 rollout: penetration depth per
object, and barrier crossing (tunnelling)."""

import numpy as np, mujoco

# Run rule R1: nothing may stay sunk more than SINK_M into anything for HOLD_S or longer. Contacts with the verifiers'
# own probes, and with objects a test pins in place, do not count.
SINK_M, HOLD_S = 0.005, 0.3


class Monitor:
    """Dynamic integrity during one rollout, per monitored object (root body id): the deepest penetration into any other
    object or the architecture (an overlap between two monitored objects counts against the one that moved more), and
    barrier crossing: a link's centre passing through another object's collision geometry between two checks. An object
    can tunnel through a thin wall and end with zero overlap. Contacts are checked every physics step; larger
    motions against fixed external colliders are checked on the joint-space integration path. The legacy centre
    crossing diagnostic (including moving obstacles) retains its `every` interval. No change to solver timestep."""

    def __init__(self, scene, owners, every=5, ignore=(), pinned=()):
        """ignore: owners whose contacts are allowed (the robot's hand on the object it handles, doc C.3). pinned:
        owners the test holds in place (tilt), whose contacts do not count toward R1's sinking."""
        self.s, self.every, self.k = scene, every, 0
        self.owners = set(int(o) for o in owners)
        self.ignore = set(int(o) for o in ignore)
        self.skip = set(int(o) for o in pinned) | {int(b) for b in getattr(scene, "probe", {}).values()}
        self.sunk = {o: 0 for o in self.owners}
        self.held = {o: 0.0 for o in self.owners}  # steps deeper than SINK_M in a row; longest, s
        self.b = [k for k in range(1, scene.m.nbody) if scene.owner[k] in self.owners and scene.m.body_mass[k] > 0]
        self.prev = {b: scene.d.xipos[b].copy() for b in self.b}
        self.pen = {o: 0.0 for o in self.owners}
        self.cand = {}  # (body, hit) -> [point before crossing, still beyond]
        self.start = {o: scene.d.xipos[o].copy() for o in self.owners}
        self.pose = (scene.d.xpos.copy(), scene.d.xmat.copy())  # every body at the last crossing check
        from sbv.functionality.trajectory import Trajectory

        self.trajectory = Trajectory(scene, self.owners, self.ignore)
        self.contact_events = []
        go = self.gowner = self.trajectory.gowners
        ids = np.arange(scene.m.nbody)  # each geom's owner; masks over owners
        self.watch, self.ign, self.skipm = (np.isin(ids, list(x)) for x in (self.owners, self.ignore, self.skip))
        self.owner_geoms = {
            o: [g for g in map(int, np.flatnonzero(go == o)) if self.trajectory.policy.active(g)] for o in self.owners
        }

    def before_step(self):
        self.trajectory.before_step()

    def tick(self):
        self.k += 1
        native = self.update_contacts()
        self.trajectory.update(native)
        for o, depth in self.trajectory.pen.items():
            self.pen[o] = max(self.pen[o], depth)
        if self.k % self.every == 0:
            self.update_crossings()

    def update_contacts(self):
        """Every contact at once, as arrays, in MuJoCo's order: the same maxima and the same first 12 events as a
        loop."""
        s = self.s
        m, d = s.m, s.d
        con = d.contact[: d.ncon]
        depth = -con.dist
        o1, o2 = self.gowner[con.geom1], self.gowner[con.geom2]
        w1, w2 = self.watch[o1], self.watch[o2]
        ok = (o1 != o2) & ~self.ign[o1] & ~self.ign[o2] & (w1 | w2)
        both = ok & w1 & w2
        moved = np.zeros(m.nbody)  # both watched: the overlap is charged to the one that moved more
        for o in np.unique(np.r_[o1[both], o2[both]]):
            moved[o] = float(np.linalg.norm(d.xipos[o] - self.start[o]))
        mine = np.where(w1 & ~(both & (moved[o2] > moved[o1])), o1, o2)  # the first one on a tie, as max() does
        native, deep = np.zeros(m.nbody), np.zeros(m.nbody)
        np.maximum.at(native, mine[ok], depth[ok])
        keep = ok & ~self.skipm[o1] & ~self.skipm[o2]
        np.maximum.at(deep, mine[keep], depth[keep])
        for i in np.flatnonzero(ok & (depth > 0.002))[: max(0, 12 - len(self.contact_events))]:
            c = con[int(i)]
            force = np.zeros(6)
            if c.efc_address >= 0:
                mujoco.mj_contactForce(m, d, int(i), force)
            self.contact_events.append(
                dict(
                    owners=[int(mine[i])],
                    time=self.trajectory.t0,
                    geoms=[int(c.geom1), int(c.geom2)],
                    position=c.pos.tolist(),
                    penetration=-float(c.dist),
                    force=force.tolist(),
                )
            )
        dt = float(m.opt.timestep)
        out = {}
        for o in self.owners:
            out[o] = float(native[o])
            self.pen[o] = max(self.pen[o], out[o])
            self.sunk[o] = self.sunk[o] + 1 if deep[o] > SINK_M else 0
            self.held[o] = max(self.held[o], self.sunk[o] * dt)
        return out

    def update(self):
        """Legacy explicit tick; report()/ok() never inspect a later rollout."""
        self.tick()

    def update_crossings(self):
        """A link's centre passing through another object's collision geometry since the last check, judged in that
        object's own frame: the lower of two stacked books landing while the upper one falls into its old place is no
        crossing, as the upper book moved too. A fixed obstacle's frame is the world's, so nothing changes for it."""
        s = self.s
        m, d = s.m, s.d
        for b in self.b:
            p0, p1 = self.prev[b], d.xipos[b].copy()
            v = p1 - p0
            L = float(np.linalg.norm(v))
            if L > 1e-4:
                hit = self.ray_hit(p0, v / L, L, int(s.owner[b]))
                if hit is not None and (b, hit) not in self.cand:
                    local = self.local(hit, p0)
                    q0 = self.world(hit, local)
                    w = p1 - q0
                    Lw = float(np.linalg.norm(w))
                    if Lw > 1e-4 and self.ray_hit(q0, w / Lw, Lw, int(s.owner[b])) == hit:
                        self.cand[(b, hit)] = [local, True]
            self.prev[b] = p1
        for (b, hit), c in self.cand.items():  # a crossing counts while the centre stays on the far side
            v = self.world(hit, c[0]) - d.xipos[b]
            L = float(np.linalg.norm(v))
            c[1] = L > 1e-4 and self.ray_hit(d.xipos[b].copy(), v / L, L, int(s.owner[b])) == hit
        self.pose = (d.xpos.copy(), d.xmat.copy())

    def local(self, o, p):
        """Point p in owner o's frame at the last check (the world's own for o = 0)."""
        return p.copy() if not o else self.pose[1][o].reshape(3, 3).T @ (p - self.pose[0][o])

    def world(self, o, q):
        """A point of owner o's frame where o is now."""
        d = self.s.d
        return q.copy() if not o else d.xpos[o] + d.xmat[o].reshape(3, 3) @ q

    def ray_hit(self, p, u, L, owner):
        s = self.s
        m, d = s.m, s.d
        gid = np.zeros(1, np.int32)
        travelled = 0.0
        for _ in range(8):
            x = mujoco.mj_ray(m, d, p, u, None, 1, -1, gid)
            if x < 0 or travelled + x > L:
                return None
            g = int(gid[0])
            o = int(s.owner[m.geom_bodyid[g]])
            if (
                o != owner
                and o not in self.ignore
                and self.trajectory.policy.active(g)
                and not s.name(o).startswith("probe_")
                and any(self.trajectory.policy.allows(a, g) for a in self.owner_geoms.get(owner, ()))
            ):
                return o  # the owner hit (0: the world)
            step = x + 1e-4
            p = p + u * step
            travelled += step
        return None

    def crossed(self, owners):
        owners = {int(o) for o in owners}
        return self.trajectory.crossed(owners) + [
            f"{self.s.name(b)} through {self.s.name(hit) if hit else 'world'}"
            for (b, hit), c in self.cand.items()
            if c[1] and int(self.s.owner[b]) in owners
        ]

    def ok(self, owners, tol):
        return all(
            max(self.pen.get(int(o), 0.0), self.trajectory.upper.get(int(o), 0.0)) <= tol
            and int(o) not in self.trajectory.unresolved
            for o in owners
        ) and not self.crossed(owners)

    def ok_rules(self, owners, sink=True):
        """Run rules R1 (no object stays sunk more than SINK_M for HOLD_S; off in tests where things fall, sink=False)
        and R2 (no tunnelling, every motion check resolved). One limit for every room: no tolerance from the reference
        room."""
        return (
            not self.crossed(owners)
            and not (set(int(o) for o in owners) & self.trajectory.unresolved)
            and (not sink or all(self.held.get(int(o), 0.0) < HOLD_S for o in owners))
        )

    def report(self, owners):
        return dict(
            max_penetration=round(max([self.pen.get(int(o), 0.0) for o in owners] or [0.0]), 5),
            crossed=self.crossed(owners)[:3],
            sunk_s=round(max([self.held.get(int(o), 0.0) for o in owners] or [0.0]), 3),
            penetration_upper=round(
                max([max(self.pen.get(int(o), 0.0), self.trajectory.upper.get(int(o), 0.0)) for o in owners] or [0.0]),
                6,
            ),
            sweep_unresolved=bool(set(owners) & self.trajectory.unresolved),
            sweep_reasons=sorted({r for o in owners for r in self.trajectory.reasons.get(o, ())}),
            sweep_accuracy=0.0005,
            sweep_scope="fixed_external_colliders",
            sweep_evaluations=self.trajectory.evaluations,
            sweep_events=[e for e in self.trajectory.events if e["owner"] in owners],
            contact_events=[e for e in self.contact_events if set(e["owners"]) & set(owners)],
        )
