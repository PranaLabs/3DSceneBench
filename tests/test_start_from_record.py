"""The nominal init record's state at settle_s is Scene.settle()'s start, bit for bit, so calibration's prepare job and
scoring take the settled start from it instead of settling the room again."""

import os, tempfile, unittest
from sbv.common.scene import Scene, PHYSICS
from sbv.functionality import f2_physical as P
from tests import rooms


class StartFromRecord(unittest.TestCase):
    def test_nominal_record_passes_through_the_settled_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, xml in (("room", rooms.room_xml()), ("robot_room", rooms.robot_room_xml())):
                with self.subTest(room=name):
                    path = rooms.write(os.path.join(tmp, name), xml)
                    scene = Scene(path, dict(PHYSICS), probes=True)
                    keep = {}
                    first = P.init_record(scene, keep=keep)  # record first (prepare job and scoring now) ...
                    jittered = P.init_record(scene, scene.jittered(1, base=scene.q0), keep={})
                    scene.settle()  # ... then settle the same scene (as before)
                    self.assertEqual(keep["q1"].dtype, scene.start.dtype)
                    self.assertEqual(keep["q1"].tobytes(), scene.start.tobytes())
                    fresh = Scene(path, dict(PHYSICS), probes=True)
                    fresh.settle()  # a room settled on its own
                    self.assertEqual(keep["q1"].tobytes(), fresh.start.tobytes())
                    self.assertEqual(
                        repr(first), repr(P.init_record(fresh))
                    )  # settling first does not change the record
                    self.assertEqual(repr(jittered), repr(P.init_record(fresh, fresh.jittered(1, base=fresh.q0))))


if __name__ == "__main__":
    unittest.main()
