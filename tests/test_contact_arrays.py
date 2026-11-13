"""The Monitor, the door and drawer trace and the robot push force handle contacts as arrays: the same state, bit for
bit, as the loops over every contact they replace (kept here), on real states with about 1,500 contacts."""

import pickle, tempfile, unittest, mujoco, numpy as np
from types import SimpleNamespace
from unittest.mock import patch
from sbv.common.scene import Scene, PHYSICS
from sbv.functionality import f3_functions as Fn
from sbv.functionality.integrity import Monitor, SINK_M
from sbv.functionality.joint_trace import DriveTrace
from sbv.functionality.trajectory import Trajectory, ACCURACY
from sbv.functionality.f4_robot.push import hand_force
from tests import rooms


def loop_contacts(self):  # Monitor.update_contacts before the change
    s = self.s
    m, d = s.m, s.d
    native = {o: 0.0 for o in self.owners}
    deep = {o: 0.0 for o in self.owners}
    for i, c in enumerate(d.contact[: d.ncon]):
        o1, o2 = int(s.owner[m.geom_bodyid[c.geom1]]), int(s.owner[m.geom_bodyid[c.geom2]])
        if o1 == o2 or o1 in self.ignore or o2 in self.ignore:
            continue
        mine = [o for o in (o1, o2) if o in self.pen]
        if len(mine) == 2:
            mine = [max(mine, key=lambda o: float(np.linalg.norm(d.xipos[o] - self.start[o])))]
        for o in mine:
            native[o] = max(native[o], -float(c.dist))
            self.pen[o] = max(self.pen[o], -float(c.dist))
            if not ({o1, o2} & self.skip):
                deep[o] = max(deep[o], -float(c.dist))
        if mine and -c.dist > 0.002 and len(self.contact_events) < 12:
            force = np.zeros(6)
            if c.efc_address >= 0:
                mujoco.mj_contactForce(m, d, i, force)
            self.contact_events.append(
                dict(
                    owners=mine,
                    time=self.trajectory.t0,
                    geoms=[int(c.geom1), int(c.geom2)],
                    position=c.pos.tolist(),
                    penetration=-float(c.dist),
                    force=force.tolist(),
                )
            )
    for o in self.owners:
        self.sunk[o] = self.sunk[o] + 1 if deep[o] > SINK_M else 0
        self.held[o] = max(self.held[o], self.sunk[o] * float(m.opt.timestep))
    return native


def loop_charge(self, native, travel):  # in Trajectory.update before the change
    for o in self.owners:
        mask = self.gowners[self.watched] == o
        bound = float(travel[mask].max()) if mask.any() else 0.0
        self.upper[o] = max(self.upper[o], native.get(o, 0.0) + min(bound, ACCURACY))


def loop_tick(self, k, final=False):  # DriveTrace.tick before the change
    m, d = self.s.m, self.s.d
    self.peak_constraint = max(self.peak_constraint, abs(float(d.qfrc_constraint[self.v])))
    if k % self.stride and not final:
        return
    contacts = []
    for i, c in enumerate(d.contact[: d.ncon]):
        if (m.geom_bodyid[c.geom1] in self.sub) == (m.geom_bodyid[c.geom2] in self.sub):
            continue
        force = np.zeros(6)
        if c.efc_address >= 0:
            mujoco.mj_contactForce(m, d, i, force)
        contacts.append(
            dict(geoms=[int(c.geom1), int(c.geom2)], depth=max(0.0, -float(c.dist)), normal_force=float(force[0]))
        )
    self.samples.append(
        dict(
            time=float(d.time),
            q=float(d.qpos[self.a]),
            velocity=float(d.qvel[self.v]),
            force_time=float(d.time - m.opt.timestep),
            constraint_force=float(d.qfrc_constraint[self.v]),
            passive_force=float(d.qfrc_passive[self.v]),
            contacts=sorted(contacts, key=lambda c: abs(c["normal_force"]), reverse=True)[:4],
        )
    )


def loop_hand_force(w, tb, u):  # push.hand_force before the change
    m, d = w.m, w.d
    f, tot = np.zeros(6), 0.0
    for i, cn in enumerate(d.contact[: d.ncon]):
        b1, b2 = m.geom_bodyid[cn.geom1], m.geom_bodyid[cn.geom2]
        if (w.owner[b1] == tb and b2 in w.robot) or (w.owner[b2] == tb and b1 in w.robot):
            mujoco.mj_contactForce(m, d, i, f)
            tot += abs(float((cn.frame.reshape(3, 3).T @ f[:3])[:2] @ u))
    return tot


LOOPS = (
    patch.object(Monitor, "update_contacts", loop_contacts),
    patch.object(Trajectory, "charge", loop_charge),
    patch.object(DriveTrace, "tick", loop_tick),
)


class ContactArrays(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The test room, 144 cubes in the drawer, 70 boxes on the floor (10 on nonrigid contacts: they sink 9 mm, for
        # R1) and in the drawer a weight on a block: two parts of the drawer touching each other, which its trace must
        # leave out.
        parts = (
            'frictionloss="0.3"/><body name="block" pos="-0.21 0 0.05"><joint type="slide" axis="0 1 0" range="-0.001 0.001"/>'
            '<geom type="box" size="0.01 0.01 0.01"/></body><body name="weight" pos="-0.21 0 0.07"><joint type="slide" axis="0 0 1"/>'
            '<geom type="box" size="0.01 0.01 0.01" mass="2"/></body>'
        )
        drawer = "".join(
            f'<body name="cube__{i}" pos="{3.54 + 0.04 * (i % 9)} {1.1 + 0.04 * (i // 9)} 0.475"><freejoint/>'
            f'<geom type="box" size="0.019 0.019 0.019"/></body>'
            for i in range(144)
        )
        nonrigid = ' solref="0.5 1"'
        floor = "".join(
            f'<body name="crate__{i + 1}" pos="{2.4 + 0.099 * (i % 10)} {0.2 + 0.099 * (i // 10)} 0.05">'
            f'<freejoint/><geom type="box" size="0.05 0.05 0.05"{nonrigid if i < 10 else ""}/></body>'
            for i in range(70)
        )
        d = tempfile.mkdtemp()
        rooms.write(
            d,
            rooms.room_xml()
            .replace('frictionloss="0.3"/>', parts)
            .replace("</worldbody>", drawer + floor + "</worldbody>"),
        )
        cls.S = S = Scene(d + "/scene.xml", dict(PHYSICS, settle_s=0.5), probes=True)
        S.settle()
        cls.cubes = {S.objects[f"cube__{i}"] for i in range(144)}
        cls.crates = [S.objects[f"crate__{i + 1}"] for i in range(70)]

    def test_monitor_on_the_whole_room_is_identical(
        self,
    ):  # every object watched, as in the gravity rollout; 5 sinking boxes pinned
        S = self.S

        def run():
            S.reset()
            mon = Monitor(S, S.objects.values(), pinned=self.crates[5:15], ignore=self.crates[60:])
            S.step(0.4, mon)
            t = mon.trajectory
            return mon, pickle.dumps(
                (
                    mon.pen,
                    mon.held,
                    mon.sunk,
                    mon.contact_events,
                    t.upper,
                    t.pen,
                    t.events,
                    t.unresolved,
                    mon.report(list(S.objects.values())),
                )
            )

        with LOOPS[0], LOOPS[1]:
            old = run()[1]
        mon, new = run()
        self.assertEqual(old, new)
        self.assertEqual(len(mon.contact_events), 12)
        self.assertEqual(
            sorted(o for o, h in mon.held.items() if h >= 0.3), self.crates[:5]
        )  # R1 counts the unpinned ones

    def test_door_and_drawer_traces_are_identical(self):
        S = self.S
        owner = S.objects["cabinet__0"]
        touching = 0
        for j in (S.m.joint("drawer_slide").id, S.m.joint("door_hinge").id):
            sgn = Fn.joint_record(S, "cabinet__0", j)["sgn"]
            for F in (5.0, 60.0):
                with LOOPS[0], LOOPS[1], LOOPS[2]:
                    t0, m0 = Fn.drive(S, j, sgn, F, 0.2, None, owner)
                    q0 = S.d.qpos.copy()
                t1, m1 = Fn.drive(S, j, sgn, F, 0.2, None, owner)
                self.assertEqual(
                    pickle.dumps((t0, m0.drive_report, m0.report([owner]))),
                    pickle.dumps((t1, m1.drive_report, m1.report([owner]))),
                )
                np.testing.assert_array_equal(q0, S.d.qpos)
                touching += sum(len(s["contacts"]) for s in m1.drive_report["samples"])
        self.assertGreater(touching, 100)  # the drawer carries the cubes

    def test_push_force_is_identical(
        self,
    ):  # the cubes stand in for the robot's hand, the cabinet or a crate for the object
        S, m, d = self.S, self.S.m, self.S.d
        j = S.m.joint("drawer_slide").id
        cases = [
            (S.objects["cabinet__0"], self.cubes),
            (self.crates[11], set(self.crates) - {self.crates[11]}),
            (self.crates[11], set()),
        ]
        S.reset()
        nonzero = 0
        for k in range(100):
            d.qfrc_applied[m.jnt_dofadr[j]] = 40.0
            mujoco.mj_step(m, d)
            for tb, robot in cases if k % 10 == 0 else ():
                w = SimpleNamespace(m=m, d=d, owner=S.owner, robot=robot)
                a, b = loop_hand_force(w, tb, np.array([0.6, -0.8])), hand_force(w, tb, np.array([0.6, -0.8]))
                self.assertEqual(np.float64(a).tobytes(), np.float64(b).tobytes())
                nonzero += a > 0
        self.assertGreater(nonzero, 10)


if __name__ == "__main__":
    unittest.main()
