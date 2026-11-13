"""Edge drop (F2.6) at the copy's own edge: the reference room's two mug spots give the direction and the sideways
offset; the spots are 3 cm inside and outside the copy's own top edge, so a desk a little narrower, wider or moved
passes (its size is geometry's), and a top with no collision surface fails."""

import os, tempfile, unittest
from sbv.common.scene import Scene
from sbv.functionality import f2_calibrate as C, f2_physical as P


def room(w, dx=0.0, solid=True):
    """A desk w metres long (top 5 cm thick on four legs) at x = dx, and a lamp far away; solid=False: a top that is
    only seen, never touched."""
    top = (
        f'<geom type="box" pos="0 0 0.725" size="{w / 2} 0.3 0.025"'
        + ("" if solid else ' contype="0" conaffinity="0"')
        + "/>"
    )
    legs = "".join(
        f'<geom type="box" pos="{sx * (w / 2 - 0.03)} {sy * 0.27} 0.35" size="0.02 0.02 0.35"/>'
        for sx in (-1, 1)
        for sy in (-1, 1)
    )
    x = f"""<mujoco><worldbody><geom name="floor" type="plane" size="5 5 0.1"/>
<body name="desk__0" pos="{dx} 0 0">{top}{legs}</body>
<body name="lamp__0" pos="3 3 0.1"><freejoint/><geom type="box" size="0.05 0.05 0.1" mass="1"/></body>
</worldbody></mujoco>"""
    d = tempfile.mkdtemp(prefix="sbv_edge_")
    path = os.path.join(d, "scene.xml")
    with open(path, "w") as f:
        f.write(x)
    return path


class EdgeDrop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        t = Scene(room(1.2), probes=True)
        t.settle()
        cls.unit, want = C.cal_edge(
            t, "desk__0", C.context(t, [3.0, 0.0])
        )  # the room's centre to +x: the desk's +x end
        assert want == [False, True] and C.finalize(t, cls.unit, want) is None

    def outcome(self, w, dx=0.0, solid=True):
        s = Scene(room(w, dx, solid), probes=True)
        s.settle()
        return P.trial(s, self.unit, qpos=s.start)[0]

    def test_the_true_desk_passes(self):
        self.assertEqual(self.outcome(1.2), self.unit["expect"])

    def test_a_desk_of_another_size_or_place_passes(self):
        for w, dx in (
            (1.08, 0.0),
            (1.32, 0.0),
            (1.2, 0.05),
        ):  # 10% narrower or wider, 5 cm off: before, all three failed
            self.assertEqual(self.outcome(w, dx), self.unit["expect"], (w, dx))

    def test_a_top_that_cannot_be_touched_fails(self):
        self.assertEqual(
            self.outcome(1.2, solid=False), [True, True]
        )  # the mug falls through it at the inside spot too


if __name__ == "__main__":
    unittest.main()
