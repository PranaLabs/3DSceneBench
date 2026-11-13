"""Non-rigid verifiers (S): a copy's fluids, soft bodies and thin shells, described in its side file nonrigid.json, run
in Genesis on CPU under hidden actions and compared with the reference room. Rigid objects stay with MuJoCo (A, G, F).
    schema.py    the side file, and the solver settings every copy is run with
    room.py      the part of the copy's scene.xml around one object, as MJCF Genesis can load
    interior.py  a fluid holder's real inside, as 1.5 mm cubes (what counts as spilled)
    actions.py   the hidden actions and their readouts; drape.py the draped-sheet cloth shape
    world.py     one Genesis run: one object, one hidden action
    verify.py    calibrate on the reference room, evaluate a copy (python -m sbv.nonrigid.verify)
"""
