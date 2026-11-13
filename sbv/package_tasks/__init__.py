"""Package tasks (F4.7): tasks a reference room brings in its own code, for outcomes the built-in moves cannot reach,
such as taking a dust cover off a microscope. They are the "package" family of the room's tasks.json; the code lives in
a folder of the room. The room supplies the action, its stages and the success check; the verifiers own the rest: the
check runs on the reference room, nominal and jittered, every declared wrong copy must fail it, the code is frozen by
its hash in package_case.json, and a copy is scored by running the same frozen code on the copy's matched objects. See
schema.py for the entries and the entry contract, verify.py for calibration and scoring."""
