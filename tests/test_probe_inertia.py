"""Probe bodies keep their mass whatever the copy's <compiler> and <default> say."""

import tempfile, unittest
from pathlib import Path
import mujoco, numpy as np
from sbv.common.scene import Scene, add_probes

MASSES = {
    "probe_ball": 0.01,
    "probe_mug": 0.3,
    "probe_load": 5.0,
    "probe_wrist": 0.5,
    "probe_finger_l": 0.2,
    "probe_finger_r": 0.2,
}
ROOM = (
    '<worldbody><geom name="floor" type="plane" size="3 3 .1"/>'
    '<body name="box__0" pos="0 0 .2"><freejoint/><inertial pos="0 0 0" mass="1" diaginertia=".01 .01 .01"/>'
    '<geom type="box" size=".1 .1 .1"/></body></worldbody>'
)


class ProbeInertia(unittest.TestCase):
    def scene(self, head):
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        p = Path(t.name) / "s.xml"
        p.write_text(f"<mujoco>{head}{ROOM}</mujoco>")
        return Scene(str(p), probes=True)

    def test_copy_without_inertia_from_geoms(self):
        s = self.scene('<compiler inertiafromgeom="false"/><default><geom mass="0"/></default>')
        for b, mass in MASSES.items():
            k = s.m.body(b).id
            self.assertAlmostEqual(s.m.body_mass[k], mass, places=9, msg=b)
            self.assertTrue((s.m.body_inertia[k] > 0).all(), b)

    def test_ordinary_room_is_unchanged(self):
        # the explicit inertials are MuJoCo's own numbers: a room with ordinary settings compiles exactly as before
        s = self.scene("")
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        p = Path(t.name) / "s.xml"
        p.write_text(f"<mujoco>{ROOM}</mujoco>")
        spec = mujoco.MjSpec.from_file(str(p))
        add_probes(spec, inertials=False)
        m = spec.compile()
        for b in MASSES:
            for f in ("body_mass", "body_inertia", "body_ipos", "body_iquat"):
                np.testing.assert_array_equal(
                    getattr(s.m, f)[s.m.body(b).id], getattr(m, f)[m.body(b).id], err_msg=f"{b} {f}"
                )


if __name__ == "__main__":
    unittest.main()
