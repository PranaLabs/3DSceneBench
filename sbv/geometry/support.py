"""Local contact support and paths to architecture, separate from F2 stability.

Contacts may occur anywhere on an object (rims, shoulders, hooks), not just at
its AABB minimum. A path is necessary, not a force/torque equilibrium proof.
"""

import copy, heapq
from collections import deque

import mujoco
import numpy as np

UP = 0.1  # reject horizontal/underside normals; tilted supports are allowed
EPS = (
    1e-6  # float slack on every tolerance compare: a bench whose lowest point is 0.010000000000000009 m up meets 0.01 m
)
SINK = np.cos(np.radians(30))  # sunk_from_above lifts by depth / SINK: a sink leaning up to 30 degrees still rests


def box_separation(m, d, a, b):
    """OBB separating axis and normal from b to a.

    Native box distance can select a lateral witness for exactly coplanar faces.
    SAT recovers the actual touching face; its positive distance is a lower bound,
    so this is used only for the local contact tolerance, not motion certification.
    """
    A, B = d.geom_xmat[a].reshape(3, 3), d.geom_xmat[b].reshape(3, 3)
    axes = np.vstack((A.T, B.T, np.cross(A.T[:, None], B.T[None, :]).reshape(-1, 3)))
    norms = np.linalg.norm(axes, axis=1)
    axes = axes[norms > 1e-9] / norms[norms > 1e-9, None]
    projection = axes @ (d.geom_xpos[a] - d.geom_xpos[b])
    separation = np.abs(projection) - np.abs(axes @ A) @ m.geom_size[a] - np.abs(axes @ B) @ m.geom_size[b]
    i = int(np.argmax(separation))
    return float(separation[i]), axes[i] * (1 if projection[i] >= 0 else -1)


def contact_edges(scene, tol=0.01):
    """Yield (supporter geom, supported geom) from actual engine contacts.

    The contact frame normal points from geom1 to geom2. Frictional contacts
    also count when the solver reports upward load; mere sideways touch does not.
    """
    m, d = scene.m, scene.d
    friction_ready = set()
    for n, owner in scene.objects.items():
        free = scene.free_dof(n)
        if free is not None:
            v = free[1]
            if (
                np.linalg.norm(d.qacc[v : v + 3]) < 0.1
                and np.linalg.norm(d.qacc[v + 3 : v + 6]) < 0.1
                and np.linalg.norm(d.qvel[v : v + 6]) < 0.001
            ):
                friction_ready.add(owner)
    for i, c in enumerate(d.contact[: d.ncon]):
        if c.dist > tol or min(c.geom1, c.geom2) < 0:
            continue
        a, b = int(c.geom1), int(c.geom2)
        if not scene.collides(a) or not scene.collides(b):
            continue
        force = np.zeros(6)
        if c.efc_address >= 0:
            mujoco.mj_contactForce(m, d, i, force)
        up_force = float((c.frame.reshape(3, 3).T @ force[:3])[2])
        for s, w, sign in ((a, b, 1), (b, a, -1)):
            owner = int(scene.owner[m.geom_bodyid[w]])
            threshold = max(1e-5, float(m.body_subtreemass[owner]) * 9.81 * 1e-4)
            if sign * c.frame[2] > UP or (owner in friction_ready and sign * up_force > threshold):
                yield s, w


def rooted_paths(parents):
    """Reachability to world, including cycles that have an external support.

    Two unsupported objects cannot become roots by pointing to each other.
    """
    paths = {"world": ["world"]}
    children = {}
    for child, supports in parents.items():
        for parent in supports:
            children.setdefault(parent, set()).add(child)
    todo = deque(["world"])
    while todo:
        parent = todo.popleft()
        for child in sorted(children.get(parent, ())):
            if child not in paths:
                paths[child] = [child] + paths[parent]
                todo.append(child)
    return paths


class SupportIndex:
    """One configuration's support queries; never retained across scene.step/reset. pen: the case's penetration_tol;
    a geom pair deeper than it supports only from above (sunk_from_above). None: depth is not read."""

    def __init__(self, scene, tol, pen=None):
        self.scene, self.tol, self.pen = scene, tol, pen
        m = scene.m
        names = {b: n for n, b in scene.objects.items()}
        self.label = {}
        for g in range(m.ngeom):
            owner = int(scene.owner[m.geom_bodyid[g]])
            if not scene.collides(g) or scene.name(owner).startswith("probe_"):
                continue
            # Architecture is the construction contract's fixed room structure.
            if owner not in names and m.body_weldid[m.geom_bodyid[g]] != 0:
                continue
            self.label[g] = names.get(owner, "world")
        self.geoms = np.array(sorted(self.label), dtype=int)
        self.lo, self.hi = np.empty((m.ngeom, 3)), np.empty((m.ngeom, 3))
        for g in self.geoms:
            points = scene.geom_points(g)
            self.lo[g], self.hi[g] = points.min(0), points.max(0)
        self.planes = m.geom_type[self.geoms] == mujoco.mjtGeom.mjGEOM_PLANE
        self.native = set(contact_edges(scene, tol + EPS))
        self.pairs = {tuple(sorted((int(a), int(b)))) for a, b in zip(m.pair_geom1, m.pair_geom2)}
        self.excludes = set(map(int, m.exclude_signature))
        self.cache = {}
        self.scratch = None
        self.sunk = {}

    def allowed(self, a, b):
        m = self.scene.m
        if tuple(sorted((a, b))) in self.pairs:
            return True
        x, y = sorted((int(m.geom_bodyid[a]), int(m.geom_bodyid[b])))
        if x == y or (x << 16) + y in self.excludes:
            return False
        return bool((m.geom_contype[a] & m.geom_conaffinity[b]) or (m.geom_contype[b] & m.geom_conaffinity[a]))

    def nearby(self, a):
        ids = self.geoms
        close = self.planes | (
            np.all(self.lo[ids] <= self.hi[a] + self.tol + EPS, axis=1)
            & np.all(self.hi[ids] >= self.lo[a] - self.tol - EPS, axis=1)
        )
        return ids[close]

    def supports(self, a, b):
        """Does geom b provide local upward support to a within tolerance? Through a penetration deeper than pen, only
        when a's object rests in b from above: a drawer whose top cap was pushed 2 cm up into a ledge of the building
        held up 17 objects, and a coat stand's pole through a shelf held the shelf up. The penetrating object still
        fails placement's non_penetration."""
        if not self.upward(a, b):
            return False
        if self.pen is None:
            return True
        from sbv.geometry.g3_layout import pair_gap

        return pair_gap(self.scene, a, b, self.tol + 0.001) >= -self.pen or self.sunk_from_above(a, b)

    def sunk_from_above(self, a, b):
        """Geom a is deeper than pen in geom b: does a's object rest in b from above? b's top must be below the object's
        top, and lifting the whole object straight up by D / SINK, D its deepest penetration into b, must free it: every
        geom of it that was deeper than pen clears b, and none ends deeper than pen. A counter base sunk 4 cm into the
        floor rests, so the 51 objects on it keep their support; a box pushed sideways into a wall, an object pushed
        into b from below or sunk through it, and a pole through a shelf do not. The lift needs no contact normal, so a
        thin part inside b (the drawer's top cap, inside the ledge) cannot pass for the whole object."""
        from sbv.geometry.g3_layout import pair_gap

        n = self.label[a]
        if (n, b) in self.sunk:
            return self.sunk[n, b]
        own = [g for g in self.scene.ogeoms[n] if g in self.label]
        d = self.scene.d
        ok = False
        if self.hi[b, 2] < self.hi[own, 2].max():  # a plane's top is its height; a wall plane's is far up
            plane = self.scene.m.geom_type[b] == mujoco.mjtGeom.mjGEOM_PLANE
            near = lambda g, up: g == a or (
                self.allowed(g, b)
                and (
                    plane
                    or (
                        (self.lo[g] + up - self.tol <= self.hi[b]).all()
                        and (self.hi[g] + up + self.tol >= self.lo[b]).all()
                    )
                )
            )
            depth = {g: pair_gap(self.scene, g, b, self.tol + 0.001) for g in own if near(g, 0)}
            up = np.array([0, 0, -min(depth.values()) / SINK])
            x = d.geom_xpos[own].copy()
            d.geom_xpos[own] = x + up
            try:
                ok = all(
                    pair_gap(self.scene, g, b, self.tol + 0.001)
                    >= (-1e-6 if depth.get(g, 0) < -self.pen else -self.pen)
                    for g in own
                    if g in depth or near(g, up)
                )
            finally:
                d.geom_xpos[own] = x
        self.sunk[n, b] = ok
        return ok

    def upward(self, a, b):
        """supports() without the depth rule."""
        key = (a, b)
        if key in self.cache:
            return self.cache[key]
        result = False
        if self.allowed(a, b):
            if (b, a) in self.native:
                result = True
            else:
                m, d = self.scene.m, self.scene.d
                points = np.zeros(6)
                if m.geom_type[a] == m.geom_type[b] == mujoco.mjtGeom.mjGEOM_BOX:
                    distance, normal = box_separation(m, d, a, b)
                    # Without an engine contact, b must also lie under a: their footprints overlap.
                    under = (self.lo[b, :2] < self.hi[a, :2]).all() and (self.hi[b, :2] > self.lo[a, :2]).all()
                    result = distance <= self.tol + EPS and normal[2] > UP and under
                    self.cache[key] = bool(result)
                    return bool(result)
                distance = mujoco.mj_geomDistance(m, d, a, b, self.tol + 0.001, points)
                if distance < 0 or (distance < self.tol + 0.001 and not points.any()):
                    from sbv.geometry.g3_layout import STEP, pair_gap

                    if pair_gap(self.scene, a, b, self.tol + 0.001) >= -1e-6:
                        # MuJoCo misreads shapes within nanometres of touching: a lamp 1 nm above its table reads 56 mm
                        # deep, with witnesses through the table; a table's mesh leg flush on a box floor reads 0 with
                        # no witness at all (all six 0), so the footprint test below would take the floor's point at the
                        # world origin. They touch, so read the direction with a lifted 1 mm.
                        x = d.geom_xpos[a].copy()
                        d.geom_xpos[a] = x + (0, 0, STEP)
                        try:
                            lifted = mujoco.mj_geomDistance(m, d, a, b, self.tol + STEP + 0.001, points)
                        finally:
                            d.geom_xpos[a] = x
                        under = (self.lo[a, :2] - 1e-6 <= points[3:5]).all() and (
                            points[3:5] <= self.hi[a, :2] + 1e-6
                        ).all()
                        result = 0 < lifted <= self.tol + STEP + EPS and under and (points[2] - points[5]) / lifted > UP
                        self.cache[key] = bool(result)
                        return bool(result)
                # Without an engine contact, b's closest point must lie under a (inside a's footprint):
                # a cup hovering just past a table edge is near the edge but not over it.
                under = (self.lo[a, :2] - 1e-6 <= points[3:5]).all() and (points[3:5] <= self.hi[a, :2] + 1e-6).all()
                if distance <= self.tol + EPS and under:
                    if abs(distance) > 1e-7 and np.linalg.norm(points[:3] - points[3:]) > 1e-7:
                        # Signed witness separation reverses for penetrating pairs.
                        result = (points[2] - points[5]) / distance > UP
                    else:
                        # At exact contact the two witnesses coincide. Query a tiny
                        # vertical displacement in scratch data to recover the normal.
                        # This also handles fixed/fixed pairs omitted by mj_collision.
                        if self.scratch is None:
                            self.scratch = mujoco.MjData(m)
                            self.scratch.geom_xpos[:] = d.geom_xpos
                            self.scratch.geom_xmat[:] = d.geom_xmat
                        eps = 1e-5
                        sd = self.scratch
                        sd.geom_xpos[a, 2] += eps
                        above = mujoco.mj_geomDistance(m, sd, a, b, self.tol + 0.001, None)
                        sd.geom_xpos[a, 2] -= 2 * eps
                        below = mujoco.mj_geomDistance(m, sd, a, b, self.tol + 0.001, None)
                        sd.geom_xpos[a] = d.geom_xpos[a]
                        result = (above - below) / (2 * eps) > UP
        self.cache[key] = bool(result)
        return bool(result)

    def on(self, name, geoms):
        targets = set(geoms)
        for a in self.scene.ogeoms[name]:
            if a not in self.label:
                continue
            for b in self.nearby(a):
                if b in targets and self.label[a] != self.label[b] and self.supports(a, int(b)):
                    return True
        return False

    def graph(self, mounts=(), corr=None):
        parents = {n: set() for n in self.scene.objects}
        for a in self.geoms:
            name = self.label[a]
            if name == "world":
                continue
            for b in self.nearby(a):
                other = self.label[b]
                if other != name and other not in parents[name] and self.supports(int(a), int(b)):
                    parents[name].add(other)
        # Only the frozen case may authorize a mount; fixing a submitted body
        # in midair cannot create one. The declared interface must still touch.
        corr = corr if corr is not None else {n: n for n in self.scene.objects}
        for mount in mounts:
            a, b = corr.get(mount["a"]), ("world" if mount["b"] == "world" else corr.get(mount["b"]))
            if a in parents and b is not None and self.mounted(a, b):
                parents[a].add(b)
        return parents, rooted_paths(parents)

    def mounted(self, a, b):
        """Can fixed object a hang on b (world or a fixed object)? A fixed geom of a touches a fixed geom of b within
        tol and the two may collide."""
        m, d = self.scene.m, self.scene.d
        if m.body_weldid[self.scene.objects[a]] != 0:
            return False
        if b != "world" and m.body_weldid[self.scene.objects[b]] != 0:
            return False
        for ga in self.scene.ogeoms[a]:
            if ga not in self.label or m.body_weldid[m.geom_bodyid[ga]] != 0:
                continue
            if any(
                self.label[gb] == b
                and m.body_weldid[m.geom_bodyid[gb]] == 0
                and self.allowed(ga, int(gb))
                and mujoco.mj_geomDistance(m, d, ga, int(gb), self.tol + 0.001, None) <= self.tol + EPS
                for gb in self.nearby(ga)
            ):
                return True
        return False

    def gaps(self, parents, paths, cap=0.10):
        """{object without a path: the smallest tolerance at which it reaches support, None beyond cap}. Support only
        grows with the tolerance, so their support edges are found once, at cap, each read at its own distance (engine
        contact depth, the box separating gap, else MuJoCo's distance; never below 0); the parents graph() found read 0,
        mounts included. An object's gap is the bottleneck: over its paths to architecture or to an object with a path,
        the smallest largest edge."""
        bad = {n for n in parents if n not in paths}
        if not bad:
            return {}
        m, d = self.scene.m, self.scene.d
        ix = copy.copy(self)
        ix.tol, ix.cache, ix.scratch, ix.sunk = cap, {}, None, {}
        ix.native = set(contact_edges(self.scene, cap))
        depth = {}
        for c in d.contact[: d.ncon]:
            k = tuple(sorted((int(c.geom1), int(c.geom2))))
            depth[k] = min(depth.get(k, np.inf), float(c.dist))

        def reading(a, b):
            if (b, a) in ix.native:
                return max(0.0, depth.get(tuple(sorted((a, b))), 0.0))
            if m.geom_type[a] == m.geom_type[b] == mujoco.mjtGeom.mjGEOM_BOX:
                return max(0.0, box_separation(m, d, a, b)[0])
            return max(0.0, mujoco.mj_geomDistance(m, d, a, b, cap + 0.001, None))

        edge = {(n, p): 0.0 for n in bad for p in parents[n]}
        for a in ix.geoms:
            if ix.label[a] not in bad:
                continue
            for b in ix.nearby(a):
                k = (ix.label[a], ix.label[b])
                if k[0] != k[1] and ix.supports(int(a), int(b)):
                    edge[k] = min(edge.get(k, np.inf), reading(int(a), int(b)))
        kids = {}
        for (c, p), v in edge.items():
            kids.setdefault(p, []).append((c, v))
        g = dict.fromkeys(bad, np.inf)
        heap = [(0.0, n) for n in sorted(paths)]
        done = set()
        while heap:
            v, u = heapq.heappop(heap)
            if u in done:
                continue
            done.add(u)
            for c, w in kids.get(u, ()):
                if c in g and max(v, w) < g[c]:
                    g[c] = max(v, w)
                    heapq.heappush(heap, (g[c], c))
        return {n: float(g[n]) if g[n] <= cap else None for n in sorted(bad)}
