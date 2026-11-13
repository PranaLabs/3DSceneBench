#!/usr/bin/env python
"""Submission audit: signs of cheating in one copy, for a person to read; it never changes a score. The scene layer
reads only the scene folder, so it also covers entrants who send no trajectory. With --case each count is set beside
the reference room's, and only what the copy has beyond it is a sign. A scene's <option> never reaches a verifier
(scene.apply_physics): it is reported as intent. The trajectory layer reads a Harbor trajectory.jsonl: paths outside
/task, /output and /tmp, network attempts, and the files of /opt/scenebench/sbv read or imported. Paths are found in
the command text, so one built inside a script, or relative to a folder the agent cd'd into, is missed.
Usage: python -m sbv.audit SCENE --out audit.json [--case CASE.json] [--pictures DIR] [--trajectory trajectory.jsonl] [--name NAME]
       python -m sbv.audit --table audit.json ...     (the summary table, one line per copy)
"""

import argparse, collections, json, os, re, sys
import numpy as np, mujoco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the repository root
from sbv import groups as GR  # noqa: E402
from sbv.common import case as K  # noqa: E402
from sbv.common.scene import Scene  # noqa: E402
from sbv.functionality.f1_gate import package, constraints  # noqa: E402

COUNTS = dict(
    equality="neq",
    exclude="nexclude",
    pair="npair",
    mocap="nmocap",
    tendon="ntendon",
    actuator="nu",
    sensor="nsensor",
    plugin="nplugin",
    keyframe="nkey",
    flex="nflex",
)
INTENT = {
    "gravity",
    "wind",
    "magnetic",
    "density",
    "viscosity",
    "disableflags",
    "enableflags",
    "o_margin",
    "o_solref",
    "o_solimp",
    "o_friction",
}
IMAGES = (".png", ".jpg", ".jpeg", ".bmp", ".tga", ".tif", ".tiff", ".webp", ".hdr", ".exr", ".ktx")
ROOTS = {
    "bin",
    "boot",
    "dev",
    "etc",
    "home",
    "lib",
    "lib64",
    "media",
    "mnt",
    "opt",
    "proc",
    "root",
    "run",
    "sbin",
    "srv",
    "sys",
    "usr",
    "var",
    "logs",
    "tests",
    "solution",
    "agent",
    "installed-agent",
    "verifier",
    "workspace",
    "app",
}
HARMLESS = {"/dev/null", "/dev/stdout", "/dev/stderr"}  # /task, /output and /tmp are not in ROOTS: never reported
PATH = re.compile(
    r"(?<![\w.:/~$-])/[A-Za-z_][\w.+-]*(?:/[\w.+-]+)*"
)  # a word before it is code: (a+b)/np.pi is not a path
NET = re.compile(
    r"\b(?:curl|wget|git\s+clone|pip3?\s+(?:install|download)|urllib|socket|requests\.(?:get|post)|http\.client"
    r"|apt(?:-get)?\s+install|huggingface_hub|hf_hub_download"
    # the model APIs the agent step can reach (make_tasks.MODEL_HOSTS): only the agent's own calls may use them
    r"|(?:import|from)\s+(?:anthropic|openai|litellm)|api\.(?:anthropic|openai)\.com|(?:ANTHROPIC|OPENAI)_(?:API_KEY|AUTH_TOKEN|BASE_URL))\b"
)
SBV, MODULE = "/opt/scenebench/sbv", re.compile(r"\b(?:from|import|-m)\s+(sbv(?:\.\w+)+)")
# damping, friction loss, stiffness, armature, torsional and rolling friction: rounding-level values are no sign
FLOOR = 0.01
ACTIVE = dict(
    torsional=4, rolling=6
)  # these act only in contacts of condim 4 and 6 (a contact takes its geoms' higher condim)
LABEL = dict(
    timeconst="contact softness (solref time constant, s)",
    dampratio="contact damping ratio (solref)",
    torsional="torsional friction",
    rolling="rolling friction",
)
BITS = (
    lambda m, g: f"{m.geom_contype[g]}/{m.geom_conaffinity[g]}"
)  # two geoms touch if one's contype meets the other's conaffinity


def options(xml):
    """Engine options the scene file sets away from MuJoCo's defaults (MjSpec follows includes and <flag>)."""
    got, base, out = mujoco.MjSpec.from_file(xml).option, mujoco.MjSpec().option, {}
    for f in (f for f in dir(base) if not f.startswith("_")):
        a, b = getattr(got, f), getattr(base, f)
        if not np.array_equal(np.asarray(a), np.asarray(b)):
            out[f] = a.tolist() if isinstance(a, np.ndarray) else a if isinstance(a, (int, float)) else str(a)
    return out


def facts(s):
    """What can hold, move or hide an object besides its own contacts, and the range of its contact and joint
    numbers."""
    m = s.m
    coll = [g for g in range(m.ngeom) if s.collides(g)]
    own = lambda g: s.name(int(s.owner[m.geom_bodyid[g]]))
    ref = [
        g for g in coll if m.geom_solref[g, 0] > 0
    ]  # solref as (time constant, damping ratio); negative: raw stiffness, damping
    one, low = [g for g in coll if m.geom_condim[g] == 1], min((int(m.geom_priority[g]) for g in coll), default=0)
    span = lambda a: [round(float(a.min()), 4), round(float(a.max()), 4)] if len(a) else None
    return dict(
        counts={k: int(getattr(m, v)) for k, v in COUNTS.items()} | dict(gravcomp=int((m.body_gravcomp > 0).sum())),
        gravcomp=sorted({s.name(int(s.owner[b])) for b in np.flatnonzero(m.body_gravcomp > 0)}),
        mocap=sorted(s.name(int(b)) for b in np.flatnonzero(m.body_mocapid >= 0)),
        held={n: v for n, v in constraints(s).items() if any(v.values())},
        no_collision=sorted(n for n in s.objects if not any(s.collides(g) for g in s.ogeoms[n])),
        bits=dict(collections.Counter(BITS(m, g) for g in coll)),
        heavy_joints=sorted(
            {
                s.name(int(s.owner[m.dof_bodyid[v]]))
                for v in np.flatnonzero(  # armature that at least doubles a joint's inertia
                    (m.dof_armature > FLOOR) & (m.dof_armature > m.dof_M0 - m.dof_armature)
                )
            }
        ),
        condim=sorted(
            {int(m.geom_condim[g]) for g in coll}
        ),  # condim 1 is frictionless only against condim 1, or when it wins on priority
        frictionless=sorted({own(g) for g in one if len(one) > 1 or m.geom_priority[g] > low}),
        direct_solref=sorted({own(g) for g in coll if m.geom_solref[g, 0] < 0}),
        impedance=span(m.geom_solimp[coll, :2]),  # solimp's dmin and dmax: the lower, the softer the contact
        ranges=dict(
            friction=span(m.geom_friction[coll, 0]),
            damping=span(m.dof_damping),
            frictionloss=span(m.dof_frictionloss),
            stiffness=span(m.jnt_stiffness),  # a stiff spring holds a door shut
            torsional=span(m.geom_friction[coll, 1]),
            rolling=span(m.geom_friction[coll, 2]),  # rolling friction resists tipping
            timeconst=span(m.geom_solref[ref, 0]),
            dampratio=span(m.geom_solref[ref, 1]),
        ),
    )


def scene_layer(xml, case, pictures, signs):
    folder = os.path.dirname(xml)
    _, bad = package(xml)
    walk = [os.path.join(r, f) for r, ds, fs in os.walk(folder) for f in ds + fs]
    links = [
        f"{os.path.relpath(p, folder)} -> {os.readlink(p)}" for p in walk if os.path.islink(p)
    ]  # listed, never followed
    images = {
        os.path.relpath(p, folder): K.file_sha256(p)
        for p in walk
        if p.lower().endswith(IMAGES) and os.path.isfile(p) and not os.path.islink(p)
    }
    pics = (
        {
            K.file_sha256(os.path.join(r, f)): os.path.relpath(os.path.join(r, f), pictures)
            for r, _, fs in sorted(os.walk(pictures))
            for f in sorted(fs)
            if f.lower().endswith(IMAGES)
        }
        if pictures
        else {}
    )  # the extra pictures too (pictures/extra/)
    out = dict(
        package=bad,
        symlinks=links,
        images=len(images),
        input_pictures={r: pics[h] for r, h in images.items() if h in pics},
    )
    signs += [f"file outside the folder: {e}" for e in bad if "leaves" in e] + [f"symlink: {x}" for x in links]
    signs += [f"input picture in the copy: {r} = {p}" for r, p in out["input_pictures"].items()]
    try:
        out["options"] = options(xml)
    except Exception as e:
        out["options"] = {"error": str(e)[:300]}
    signs += [f"option: {k} = {v}" for k, v in out["options"].items() if k in INTENT]
    phys, vocab = (case["physics"], case["vocabulary"]) if case else (None, ())
    truth = (
        facts(Scene(K.reference_path(case), phys, vocab, textures=False))
        if case
        else dict(counts={}, no_collision=[], ranges={})
    )
    try:
        s = Scene(
            xml, phys, vocab, textures=False
        )  # after the reference room is freed: one room's compile in memory at a time
    except Exception as e:
        out["load_error"] = str(e)[:300]
        signs.append("does not compile: see load_error")
        return out
    out.update(facts(s))
    slot = {}
    if case:  # pairs as the grader does (groups, the case's matcher and gates)
        out["true_room"] = truth
        corr = GR.pairing(case["objects"], s, GR.groups(case))[0]
        slot = {corr[o["slot"]]: o for o in case["objects"] if o["slot"] in corr}
        apart = lambda c, o: round(float(np.linalg.norm(np.mean(s.aabb(c), 0) - o["center"])), 2)
        out["fixed_but_free_in_truth"] = {
            c: dict(slot=o["slot"], metres_apart=apart(c, o))
            for c, o in slot.items()
            if o["role"].startswith("free") and s.role[c].startswith("fixed")
        }
        signs += [
            f"fixed in the copy, free in the reference room: {c} ({v['slot']}, {v['metres_apart']} m apart)"
            for c, v in out["fixed_but_free_in_truth"].items()
        ]
        for c, o in slot.items():  # the object gate's comparison (evaluate.object_gate)
            more = [
                k
                for k in ("equality", "gravcomp", "exclude")
                if o.get("constraints") and out["held"].get(c, {}).get(k, 0) > o["constraints"][k] + 1e-9
            ]
            signs += [f"held more than its reference object: {c} ({', '.join(more)})"] if more else []
    signs += [
        f"{k}: {n} in the copy, {truth['counts'].get(k, 0)} in the reference room"
        for k, n in out["counts"].items()
        if n > truth["counts"].get(k, 0)
    ]
    used, odd = set(truth["bits"]) if case else {"1/1"}, collections.defaultdict(set)  # 1/1: MuJoCo's default
    for g in (g for g in range(s.m.ngeom) if s.collides(g) and BITS(s.m, g) not in used):
        odd[BITS(s.m, g)].add(s.name(int(s.owner[s.m.geom_bodyid[g]])))
    out["odd_bits"] = {k: sorted(v) for k, v in odd.items()}  # bits that never meet the room's can pass through it
    signs += [
        f"collision bits {k} {'not in the reference room' if case else 'not the default 1/1'}: {n}"
        for k, v in out["odd_bits"].items()
        for n in v
    ]
    beyond = lambda key: [
        n for n in out[key] if slot.get(n, {}).get("slot") not in set(truth.get(key, ()))
    ]  # its reference object has it too: no sign
    signs += [f"armature more than its joint's own inertia: {n}" for n in beyond("heavy_joints")] + [
        f"no colliding geom: {n}" for n in beyond("no_collision")
    ]
    signs += [f"frictionless contact (condim 1): {n}" for n in beyond("frictionless")]
    signs += [f"solref in direct mode (raw stiffness and damping): {n}" for n in beyond("direct_solref")]
    for k, r in out["ranges"].items():
        tr = truth["ranges"].get(k)
        if k in ACTIVE and max(out["condim"] or [3]) < ACTIVE[k]:
            continue  # no contact of the copy uses it
        if r and tr and r[1] > max(2 * tr[1], FLOOR) + 1e-9:
            signs.append(f"{LABEL.get(k, k)} up to {r[1]}, more than twice the reference room's {tr[1]}")
    lo, tlo = out["impedance"], truth.get("impedance")
    if lo and tlo and lo[0] < tlo[0] / 2:
        signs.append(
            f"contact impedance (solimp) down to {lo[0]}, under half the reference room's {tlo[0]}: softer contacts"
        )
    return out


def trajectory_layer(path):
    """The bash commands and viewed pictures of a Harbor trajectory: what they name outside the agent's own folders."""
    out = dict(
        outside=collections.Counter(), network=[], sbv_read=collections.Counter(), sbv_imported=collections.Counter()
    )
    with open(path) as f:
        for r in map(json.loads, f):
            if r.get("kind") not in ("bash", "view_image"):
                continue
            text = r.get("command") or r.get("path") or ""
            for p in {q for q in map(os.path.normpath, PATH.findall(text)) if q.split("/")[1] in ROOTS} - HARMLESS:
                out["outside"][p] += 1
                if p == SBV or p.startswith(SBV + "/"):
                    out["sbv_read"][os.path.relpath(p, os.path.dirname(SBV)) + ("/" if p == SBV else "")] += 1
            for mod in set(MODULE.findall(text)):
                out["sbv_imported"][mod.replace(".", "/") + ".py"] += 1
            for mt in NET.finditer(text):
                out["network"].append(
                    dict(
                        turn=r.get("turn"),
                        match=mt.group(0),
                        line=next((x for x in text.splitlines() if mt.group(0) in x), mt.group(0)).strip()[:200],
                    )
                )
    return {k: dict(v.most_common()) if isinstance(v, collections.Counter) else v for k, v in out.items()}


def audit(submission, case=None, pictures=None, trajectory=None, name=None):
    xml = K.scene_xml(os.path.abspath(submission))
    signs = []
    scene = scene_layer(xml, case, pictures, signs)
    return dict(
        name=name or os.path.basename(os.path.dirname(xml)),
        submission=xml,
        case_hash=(case or {}).get("case_hash"),
        scene=scene,
        signs=signs,
        trajectory=trajectory_layer(trajectory) if trajectory else None,
    )


HEADER = "| Copy | Signs in the scene | Paths outside /task /output /tmp (commands) | Network attempts | Verifier files read or imported |\n|---|---|---|---|---|"


def row(r):
    """One line of the summary table: the scene's signs by kind, then what the trajectory named (paths by folder)."""
    kinds, t, tops = (
        collections.Counter(s.split(":")[0] for s in r["signs"]),
        r.get("trajectory"),
        collections.Counter(),
    )
    cells = [r["name"], "; ".join(f"{k} x{n}" if n > 1 else k for k, n in kinds.items()) or "none"]
    if t is None:
        return "| " + " | ".join(cells + ["no trajectory"] * 3) + " |"
    for p, n in t["outside"].items():
        tops["/".join(p.split("/")[:3])] += n
    return (
        "| "
        + " | ".join(
            cells
            + [
                " ".join(f"{k} x{n}" for k, n in tops.most_common()) or "none",
                str(len(t["network"])),
                ", ".join(sorted(set(t["sbv_read"]) | set(t["sbv_imported"]))) or "none",
            ]
        )
        + " |"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", nargs="?")
    ap.add_argument("--out")
    ap.add_argument("--case")
    ap.add_argument(
        "--pictures", help="the task's input pictures: an image file in the copy with the same hash is a sign"
    )
    ap.add_argument("--trajectory", help="a Harbor trajectory.jsonl")
    ap.add_argument("--name")
    ap.add_argument(
        "--table", nargs="+", metavar="AUDIT_JSON", help="print the summary table of audits already written"
    )
    a = ap.parse_args()
    if not a.table and not (a.scene and a.out):
        ap.error("SCENE and --out are needed unless --table is given")
    reps = (
        [json.load(open(p)) for p in a.table]
        if a.table
        else [audit(a.scene, K.load(a.case) if a.case else None, a.pictures, a.trajectory, a.name)]
    )
    if not a.table:
        with open(a.out, "w") as f:
            json.dump(reps[0], f, indent=1)
    print(HEADER, *map(row, reps), sep="\n")
