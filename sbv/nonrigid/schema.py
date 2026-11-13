"""The side file `nonrigid.json` and the solver settings every copy is run with.

The kinds follow graphics simulation: rigid bodies (scene.xml, MuJoCo); deformable bodies, of which soft bodies are run
today (elastic solids such as rubber come once a test at their stiffness passes with the fixed settings); thin shells
(cloth, paper); fluids. Granular and elastoplastic matter are not run. A room's rigid objects live in scene.xml; its
non-rigid objects live here. Each object:
  {"name": "coffee__0", "kind": "fluid" | "soft_body" | "thin_shell", "on": "<rigid body in scene.xml>",
   "shape": {...}, "material": {...}}
  fluid      shape {"type": "cylinder", "pos": [x, y, z], "radius": r, "height": h}   (the fluid at rest)
              material {"viscosity": "water" | "oil" | "syrup", "density": kg/m3}
  soft_body  shape {"type": "box", "pos": [x, y, z], "size": [sx, sy, sz], "yaw_deg": a}  (full sizes)
              material {"youngs_kpa": E, "poisson": nu, "density": kg/m3}
  thin_shell  shape {"type": "draped_sheet", "rail": [x, y, z], "rail_radius": r, "width": w,
                     "front_length": f, "back_length": b}   (hung over a rail running along x)
              material {"fabric": "silk" | "cotton" | "denim" | "wool_coat"}   (scored by class: FABRIC_CLASS)
A soft body or a thin shell may instead be a mesh file in the package:
              shape {"type": "mesh", "file": "pillow__0.obj", "pos": [x, y, z], "yaw_deg": a}   (the file's own coordinates,
              turned by yaw_deg about z, then moved by pos); a soft body's mesh must be closed; a thin shell's mesh also
              gives "rail": [x, y, z], the point of the rail or hanger bar it hangs from.
"on" names the container (fluid), the support (soft body) or the rail's body (thin shell). Never written.
"""

import json
import os

import numpy as np

KINDS = ("fluid", "soft_body", "thin_shell")
DEFORMABLE = ("soft_body",)  # the deformable bodies run today
FILE, CASE, REPORT = "nonrigid.json", "nonrigid_case.json", "nonrigid_report.json"
OLD_NAMES = {FILE: "soft_liquid.json", CASE: "soft_case.json", REPORT: "soft_report.json"}
OLD_KINDS = {"liquid": "fluid", "soft_solid": "soft_body", "cloth": "thin_shell"}


def side_file(folder, name=FILE):
    """The path of a folder's nonrigid.json (or nonrigid_case.json, nonrigid_report.json): the old name's file when only
    that one exists, else the new name's path (which may not exist)."""
    new, old = os.path.join(folder, name), os.path.join(folder, OLD_NAMES[name])
    return old if not os.path.exists(new) and os.path.exists(old) else new


def kind(k):
    """A kind under its current name."""
    return OLD_KINDS.get(k, k) if isinstance(k, str) else k


def upgrade(d):
    """A side file, case or report read from disk, with every "kind" under its new name (in place; returns d)."""
    if isinstance(d, dict):
        for key, v in d.items():
            if key == "kind":
                d[key] = kind(v)
            else:
                upgrade(v)
    elif isinstance(d, list):
        for v in d:
            upgrade(v)
    return d


VISCOSITY = {
    "water": 1e-4,
    "oil": 2e-4,
    "syrup": 5e-3,
}  # Genesis SPH mu, kinematic (m2/s; its viscous term is Monaghan's).
# Was 0.005/0.02/0.08 (Genesis's default, 5000x real water: a tipped cup poured about a third of what its shape lets
# out). Water sits at 1e-4, the lowest that stays one body at 6 mm particles, oil just above (real water 1e-6 sprays as
# scattered drops; 1e-4 pours as one puddle). Real: olive oil ~1e-4, honey ~5e-3.
SETTLE_MU = 0.005  # every fluid settles at this (honey) until its action starts, then takes its own viscosity: a
# starting column collapses into a wider holder, and thin water was still sloshing high when its holder was tipped
# kg/m2, bending compliance. The compliance gives each fabric its real bending length in a Peirce cantilever (ASTM
# D1388), simulated with these settings (15 mm particles): silk 1.5 cm (the particles cannot go below about 2.4 cm, so
# the softest setting), cotton 2.5 cm, denim 4 cm, wool coat 3.5 cm. The earlier values (1e-3 to 3e-6) gave 8-10 cm:
# cloth as stiff as card.
FABRIC = {"silk": (0.08, 1e3), "cotton": (0.15, 560.0), "denim": (0.45, 4.5), "wool_coat": (0.8, 4.8)}
# The material check's two classes: in the simulation silk and cotton bend alike (2.4 and 2.5 cm) and so do denim and
# wool coat (4 and 3.5 cm), each pair within a factor of 2 in weight; light and heavy differ 3 to 10 times. Which of a
# pair a picture shows is a guess no hidden action could tell apart.
FABRIC_CLASS = {"silk": "light", "cotton": "light", "denim": "heavy", "wool_coat": "heavy"}

# Fixed for every copy and for the reference room: the copy is judged on its physics numbers, not on solver choices. All
# on CPU.
FIXED = {
    "fluid": dict(solver="SPH", particle_size=0.006, dt=2e-3, substeps=40, sampler="regular", hash_grid=0.24),
    "soft_body": dict(solver="MPM", particle_size=0.008, grid_density=64, dt=2e-3, substeps=20, sampler="regular"),
    "thin_shell": dict(solver="PBD", particle_size=0.015, dt=2e-3, substeps=10, friction=0.6),
}
# ash_grid: Genesis finds each particle's neighbours in a lookup table, by default one cell per 12 mm of the whole fluid
# box (about 300,000 cells around a tall holder), re-sorted every substep: over half of each substep. A table that
# repeats every 0.24 m finds the same neighbours and takes about 0.5 ms. The held-still room pieces: Genesis keeps a
# distance grid per collision piece; 64 cells a side instead of its default 128 took a soft body run from about 11 to 3
# GB with the same press depth to the last digit
HELD_SDF_MAX_RES = 64


def load(path):
    """The side file's objects and its problems. A file that is not a JSON object is a problem of the copy, never a
    crash."""
    try:
        with open(path) as f:
            d = json.load(f)
    except (OSError, UnicodeDecodeError, ValueError, RecursionError) as e:  # nesting deeper than the decoder allows too
        return [], [f"{os.path.basename(path)} is not valid JSON ({type(e).__name__}: {str(e)[:120]})"]
    if not isinstance(d, dict):
        return [], [f"{os.path.basename(path)} is a JSON {type(d).__name__}, not an object"]
    try:
        upgrade(d)
    except RecursionError:
        return [], [f"{os.path.basename(path)} is nested too deeply"]
    problems = [] if isinstance(d.get("objects"), list) else [f"no 'objects' list (found keys: {sorted(d)})"]
    objects = d["objects"] if isinstance(d.get("objects"), list) else []
    problems += [f"objects[{i}] is not an object" for i, o in enumerate(objects) if not isinstance(o, dict)]
    objects = [o for o in objects if isinstance(o, dict)]
    names = set()
    kept = []
    # Every entry: sane numbers, then its checks; anything an entry raises is its problem, never a crash
    for o in objects:
        n = o.get("name", "")
        if not isinstance(n, str) or not n:
            problems.append(f"objects entry {n!r}: a name is text")
            continue
        try:
            bad = insane(o)
            if bad:
                problems.append(f"{n!r}: {bad}")
                continue
            problems += entry_problems(o, names, path)
        except Exception as e:
            problems.append(f"{n!r}: cannot be read ({type(e).__name__})")
            continue  # RecursionError included
        kept.append(o)
    return kept, problems


NUMBER_MAX = (
    1e6  # no declared number is larger in magnitude: positions and sizes in metres, moduli in kPa, densities in kg/m3
)


def insane(o):
    """Why an entry's numbers cannot be used, or None: any number in its shape, material or attachments that is not a
    finite real within NUMBER_MAX in magnitude (a huge integer overflowed squares and products)."""

    def walk(v, where):
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            try:
                f = float(v)
            except OverflowError:
                return f"{where} is too large"
            return (
                None
                if np.isfinite(f) and abs(f) <= NUMBER_MAX
                else f"{where} is not a finite number within {NUMBER_MAX:g}"
            )
        if isinstance(v, dict):
            return next((w for k, x in v.items() for w in [walk(x, f"{where}.{k}")] if w), None)
        if isinstance(v, list):
            return next((w for i, x in enumerate(v) for w in [walk(x, f"{where}[{i}]")] if w), None)
        return None

    for k in ("shape", "material", "attachments"):
        if k in o and k != "attachments" and not isinstance(o[k], dict):
            return f"{k} is not an object"
        w = walk(o.get(k), k)
        if w:
            return w
    return None


def entry_problems(o, names, path):
    """One entry's problems (load() catches anything it raises)."""
    problems = []
    from .attachments import problems as attachment_problems

    problems += [f"{o.get('name', '')}: {p}" for p in attachment_problems(o)]
    n, k = o.get("name", ""), o.get("kind")
    if not isinstance(n, str):
        problems.append(f"{n!r}: a name is text")
        return problems  # a malformed entry is a problem, never a crash
    if "__" not in n or n in names:
        problems.append(f"{n!r}: names are <category>__<n> and unique")
    names.add(n)
    if k not in KINDS:
        problems.append(f"{n}: kind must be one of {KINDS}")
    if not isinstance(o.get("on"), str) or not o.get("on"):
        problems.append(f"{n}: 'on' must name a rigid body in scene.xml")
    m = o.get("material", {})
    if k == "fluid" and m.get("viscosity") not in VISCOSITY:
        problems.append(f"{n}: viscosity must be one of {list(VISCOSITY)}")
    if k == "thin_shell" and m.get("fabric") not in FABRIC:
        problems.append(f"{n}: fabric must be one of {list(FABRIC)}")
    if k == "soft_body" and not all(isinstance(m.get(x), (int, float)) and m[x] > 0 for x in ("youngs_kpa", "density")):
        problems.append(f"{n}: youngs_kpa and density must be positive numbers")
    s = o.get("shape", {})
    if not isinstance(s, dict) or not isinstance(s.get("type"), str):
        problems.append(f"{n}: shape is an object with a type")
        return problems
    if s.get("type") == "mesh":
        f = str(s.get("file", ""))
        if (
            os.path.basename(f) != f
            or not f.endswith(".obj")
            or not os.path.exists(os.path.join(os.path.dirname(path), f))
        ):
            problems.append(f"{n}: a mesh shape names an .obj file next to {os.path.basename(path)}")
        elif k == "soft_body" and not mesh_of(o, os.path.dirname(path)).is_watertight:
            problems.append(f"{n}: a soft body's mesh must be closed")
        if k == "thin_shell" and not (isinstance(s.get("rail"), list) and len(s["rail"]) == 3):
            problems.append(f"{n}: a thin shell's mesh also gives rail [x, y, z]")
    return problems


def mesh_of(o, base):
    """A mesh shape in the room's frame: the file turned by yaw_deg about z, then moved by pos."""
    import trimesh

    s = o["shape"]
    m = trimesh.load(os.path.join(base, s["file"]), force="mesh", process=False)
    m.apply_transform(trimesh.transformations.rotation_matrix(np.radians(s.get("yaw_deg", 0)), [0, 0, 1]))
    m.apply_translation(s.get("pos", [0, 0, 0]))
    return m


def sheet(o, base="."):
    """A thin shell's mesh in the room's frame: the draped sheet built from its numbers, or its mesh file as placed."""
    if o["shape"]["type"] != "draped_sheet":
        return mesh_of(o, base)
    from .drape import draped_sheet

    s = o["shape"]
    return draped_sheet(s["rail"], s["rail_radius"], s["width"], s["front_length"], s["back_length"])


def bounds(o, base="."):
    """The object's box at rest in the room's frame, as (lo, hi)."""
    s = o["shape"]
    if s["type"] == "mesh":
        return mesh_of(o, base).bounds
    if s["type"] == "draped_sheet":
        r = np.array(s["rail"], float)
        low = max(s["front_length"], s["back_length"])  # the longer panel hangs lowest
        return r - [s["width"] / 2, s["rail_radius"], low], r + [s["width"] / 2, s["rail_radius"], s["rail_radius"]]
    c = np.array(s["pos"], float)
    if s["type"] == "cylinder":
        h = np.array([s["radius"], s["radius"], s["height"] / 2])
    else:
        a = np.radians(s.get("yaw_deg", 0))
        h = np.abs(np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])) @ (
            np.array(s["size"]) / 2
        )
    return c - h, c + h


def center(o, base="."):
    lo, hi = bounds(o, base)
    return (np.asarray(lo) + np.asarray(hi)) / 2


def volume_ml(o, base="."):
    s = o["shape"]
    if s["type"] == "mesh":
        return float(abs(mesh_of(o, base).volume)) * 1e6 if o["kind"] == "soft_body" else 0.0
    if s["type"] == "cylinder":
        return 3.14159 * s["radius"] ** 2 * s["height"] * 1e6
    if s["type"] == "box":
        return s["size"][0] * s["size"][1] * s["size"][2] * 1e6
    return 0.0
