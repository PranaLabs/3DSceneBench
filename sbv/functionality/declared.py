"""What the reference room declares its containers are for, in functions.json next to the reference scene (or calibrate
--functions). The verifiers check these declarations only; nothing here is guessed from shapes:
    {"containers": {"pen_cup__0": {"holds": ["pencil__0", "pencil__1"]},
                    "watering_can__0": {"openings": [{"pos": [0.05, 0, 0.24], "dir": [0, 0, -1], "radius": 0.04}]}}}
holds: the objects the container carries. A pencil may stick half out of its cup: it counts as held if it comes along
when the cup is lifted 10 cm (F3.5). openings: where a ball goes in, in the container's own frame (metres): the centre
of the opening, the direction a ball travels going in, and the opening's radius. Each opening gets its own F3.2 run.

A room with an object named by a container word (CONTAINER_WORDS) must have a functions.json, even an empty one:
calibrate stops without it. `python -m sbv.declare ROOM` drafts the file from the reference room."""

import json, os
from sbv.common.scene import BALL_R, main_word, category, is_arch

# Words of things that hold what is put in them (sbv/common/categories.txt). Closed or lidded ones (a cardboard box, a
# kettle) are left out by sbv.declare when the ball cannot get in; the word only says "look at this one". No alias of a
# word that is not a container: flowerpot is a plant (a potted plant), so it is not here.
CONTAINER_WORDS = frozenset(
    main_word(w)
    for w in """
    basket bin box crate enamel_box food_box hamper jar roll_basket storage_bin utility_basket storage_tray supply_box
    blue_supply_box pot vase terrarium bowl cup glass mug pitcher wine_glass measuring_cup mortar cooking_pot colander
    utensil_holder tray pen_cup desk_organizer letter_tray bucket toolbox barrel laundry_basket trash_can
    watering_can cooler aquarium litter_box pet_bowl beaker crucible flask graduated_cylinder petri_dish sample_dish
    culture_bottle toothbrush_holder""".split()
)


def path(xml):
    return os.path.join(os.path.dirname(os.path.abspath(xml)), "functions.json")


def load(xml, given=None):
    p = given or path(xml)
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)


def container_like(names):
    """The objects among names (a Scene's objects, or names read from a file) named by a container word."""
    return sorted(n for n in names if not is_arch(n) and category(n) in CONTAINER_WORDS)


def object_names(xml):
    """The top-level body names of a scene file, read without compiling it (includes are followed)."""
    import mujoco

    return [b.name for b in mujoco.MjSpec.from_file(xml).worldbody.bodies if b.name]


def missing_file(xml, names, given=None):
    """Why calibration must stop: the room has container-like objects and no functions.json (None when it may go on)."""
    if given is not None or os.path.exists(path(xml)):
        return None
    like = container_like(names)
    if not like:
        return None
    return (
        f"the room has {len(like)} container-like objects ({', '.join(like[:8])}{', ...' if len(like) > 8 else ''}) and no "
        f"functions.json next to scene.xml. Run `python -m sbv.declare {os.path.dirname(os.path.abspath(xml))} --write`, review the "
        "file, or pass --allow-no-functions"
    )


def undeclared(spec, scene):
    """Container-like objects the file does not name: worth a look, never an error (a closed box is no container)."""
    named = set(spec.get("containers", {})) | set(spec.get("not_containers", {}))
    return [n for n in container_like(scene.objects) if n not in named]


def check(spec, scene):
    """The declarations the reference scene cannot carry out, in words (empty when every one can)."""
    bad = []
    for n, c in spec.get("containers", {}).items():
        if n not in scene.objects:
            bad.append(f"no object named {n}")
            continue
        held = c.get("holds", [])
        if held and scene.free_dof(n) is None:
            bad.append(f"{n} holds things but has no free joint, so it cannot be lifted")
        bad += [f"{n} holds {i}, but there is no object named {i}" for i in held if i not in scene.objects]
        bad += [
            f"{n} holds {i}, but {i} has no free joint, so it cannot come along"
            for i in held
            if i in scene.objects and scene.free_dof(i) is None
        ]
        for k, op in enumerate(c.get("openings", [])):
            p, u, r = op.get("pos"), op.get("dir"), op.get("radius")
            if not (
                isinstance(p, list)
                and len(p) == 3
                and isinstance(u, list)
                and len(u) == 3
                and any(u)
                and isinstance(r, (int, float))
            ):
                bad.append(f"{n} opening {k} needs pos [x, y, z], a non-zero dir [x, y, z] and a radius, in metres")
            elif r < BALL_R + 0.002:
                bad.append(f"{n} opening {k} is {200 * r:.1f} cm across, too narrow for the 2 cm probe ball")
    return bad
