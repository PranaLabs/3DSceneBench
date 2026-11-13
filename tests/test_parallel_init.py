"""Calibration's nominal and jittered gravity rollouts give the same records, bit for bit, one after another in the
prepare job's settled room and in parallel in fresh copies of the room."""

import os, tempfile, unittest
from sbv.calibrate import init_records
from sbv.common.scene import Scene, PHYSICS
from tests import rooms


class ParallelInit(unittest.TestCase):
    def test_parallel_records_equal_serial_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, xml in (("room", rooms.room_xml()), ("robot_room", rooms.robot_room_xml())):
                path = rooms.write(os.path.join(tmp, name), xml)
                scene = Scene(path, dict(PHYSICS), ["cup", "table"], probes=True)
                scene.settle()  # as in the prepare job
                serial = init_records(scene, 3, 1)
                for procs in (2, 4):  # 2: two workers take two seeds each
                    with self.subTest(room=name, procs=procs):
                        keep = {}
                        parallel = init_records(scene, 3, procs, keep=keep)
                        self.assertEqual(len(parallel), 4)
                        self.assertEqual(parallel, serial)
                        self.assertEqual(
                            keep["q1"].tobytes(), scene.start.tobytes()
                        )  # the worker's state at settle_s: the settled start
                        self.assertEqual(repr(parallel), repr(serial))  # every float's digits, types and key order too


if __name__ == "__main__":
    unittest.main()
