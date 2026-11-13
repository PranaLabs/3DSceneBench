#!/usr/bin/env python
"""Turn checked rooms into Harbor tasks: one task per room and picture condition (8 pictures unless asked), written
twice with the same task name:
    OUT/public/3dscenebench-<id>/    instruction, pictures + calibration, docs/WHAT_IS_TESTED.md, a tools container, a
                                     format-only grader (the checker, checker/check_scene.py, is the only code of
                                     ours in it: never sbv)
    OUT/private/3dscenebench-<id>/   the same task, but the grader is built from the verifier image with the reference room
                                     and its case baked in, and an oracle solution copies the reference room (should
                                     score 100); a PRIVATE marker file at its root says so, for any publish step to
                                     refuse
<id> is the room's number, 001 onwards, given in a random order so it says nothing about the room; kept in
OUT/private/room_ids.json (or --room-ids) and reused, so task names stay the same between runs. The name does not say
how many pictures a task gives, so one run writes one picture condition. A room's pictures (OUT/pictures/<id>-p<N>) are
rendered once and kept, and rendered again when the room's files or its input cameras change
(sbv.appearance.views.input_key). No public file names the reference room. Every task gives the same category words,
sbv/common/categories.txt, with each alias shown beside its main word. Entrants run the public tasks; the official
scores come from `harbor job regrade <job> -p OUT/private`. Rooms come from the pipeline's work folder
(WORK/rooms/<room>, WORK/results/<room>/case.json, ready in check.json). A room's about.md says in a
sentence or two what kind of room it is; its tasks are listed under it. A room's budget.json sets
its grader's CPUs, memory and minutes. The agent's budget is the same for every task, whatever its level or room:
wall-clock work minutes and output tokens (reasoning included), the run stops at whichever comes first (--budget
MINUTESm:TOKENS, default BUDGET). It goes in the instruction, in task.toml [metadata] budget_minutes and
budget_output_tokens (the run infrastructure enforces these), and in [agent] timeout_sec, Harbor's backstop only, behind
the infrastructure's own stop: the minutes plus a margin.
Usage: python -m scenebench_harbor.make_tasks --work WORK --out OUT [--pictures 1 2 4] [--org pranalabs-ai-team]
       [--grader-base 3dscenebench-grader:dev] [--grader-cpus 8] [--room-ids FILE] [--budget 180m:10000000]
"""

import argparse, json, os, re, secrets, shutil, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sbv.package_tasks import schema as PkgS  # noqa: E402
from sbv.nonrigid import schema as NS  # noqa: E402
from sbv.nonrigid import verify as NV  # noqa: E402
from sbv.common import case as K  # noqa: E402
from sbv.appearance import views as App  # noqa: E402
from sbv.common import scene as Words  # noqa: E402
from scenebench_harbor import levels as L  # noqa: E402

CHECKER = """FROM python:3.11-slim
RUN pip install --no-cache-dir mujoco==3.14.0 numpy==1.26.4 scipy==1.17.1 trimesh==3.22.5
COPY check_scene.py vocabulary.json /opt/scenebench/
"""
# Claude Code, Codex and OpenCode, installed when the image is built: with no network a task cannot install them. Harbor
# skips its own install of Claude Code and Codex when the program is already there (so a run must not ask for another
# version); its OpenCode agent always installs, so scenebench_harbor/agents.py holds one that uses the installed
# program. Their web tools are off in the image, whatever flags a run passes: Claude Code's managed settings deny
# WebSearch and WebFetch, and Codex's requirements allow only web_search = "disabled" and turn image generation off.
# Codex's system config also drops its bundled skills: their imagegen skill calls OpenAI's image API from a script,
# through the model host the agent step can reach. Both agents still look at the task's pictures (Claude Code's Read,
# Codex's view_image); OpenCode's system config (/etc/opencode, which its user config cannot override) denies webfetch,
# websearch and codesearch; the OPENCODE_PERMISSION variable would override that, so the `opencode` command is a wrapper
# (OPENCODE) that drops it, and turns off the update check and the downloads the container could not make anyway.
# scripts/check_agent_web_tools.py checks all of this in Docker. Node 22 is Claude Code's oldest; the tarball is
# checked.
NODE = ("22.23.4", "1805bcd35e6815559bd2eb97db438d3fdb0eb3a5e6a977b7c1badca76337e9ce")
AGENT_VERSIONS = {"@anthropic-ai/claude-code": "2.1.309", "@openai/codex": "0.164.0", "opencode-ai": "1.19.3"}
CLAUDE_SETTINGS = {
    "permissions": {"deny": ["WebSearch", "WebFetch"]},
    "env": {"DISABLE_AUTOUPDATER": "1", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"},
}
CODEX_REQUIREMENTS = 'allowed_web_search_modes = ["disabled"]\n\n[features]\nimage_generation = false\n'
OPENCODE_CONFIG = {
    "$schema": "https://opencode.ai/config.json",
    "autoupdate": False,
    "share": "disabled",
    "permission": {"webfetch": "deny", "websearch": "deny", "codesearch": "deny"},
}
OPENCODE_BIN = "/usr/local/lib/node_modules/opencode-ai/bin/opencode.exe"  # where npm puts the program
OPENCODE = f"""#!/bin/sh
unset OPENCODE_PERMISSION OPENCODE_ENABLE_EXA OPENCODE_EXPERIMENTAL_EXA OPENCODE_TEST_MANAGED_CONFIG_DIR
export OPENCODE_DISABLE_AUTOUPDATE=1 OPENCODE_DISABLE_MODELS_FETCH=1 OPENCODE_DISABLE_LSP_DOWNLOAD=1 OPENCODE_DISABLE_DEFAULT_PLUGINS=1
exec {OPENCODE_BIN} "$@"
"""
CODEX_CONFIG = 'web_search = "disabled"\n\n[features]\nimage_generation = false\n\n[skills.bundled]\nenabled = false\n'
AGENT_FILES = {
    "claude-managed-settings.json": (
        "/etc/claude-code/managed-settings.json",
        json.dumps(CLAUDE_SETTINGS, indent=1) + "\n",
    ),
    "codex-requirements.toml": ("/etc/codex/requirements.toml", CODEX_REQUIREMENTS),
    "codex-config.toml": ("/etc/codex/config.toml", CODEX_CONFIG),
    "opencode.json": ("/etc/opencode/opencode.json", json.dumps(OPENCODE_CONFIG, indent=1) + "\n"),
    "opencode": ("/usr/local/bin/opencode", OPENCODE),
}  # file in environment/agents/: (where, text)
AGENTS = (
    f'''RUN apt-get update && apt-get install -y --no-install-recommends ripgrep procps && rm -rf /var/lib/apt/lists/* && \\
    curl -fsSL -o /tmp/node.tar.xz https://nodejs.org/dist/v{NODE[0]}/node-v{NODE[0]}-linux-x64.tar.xz && \\
    echo "{NODE[1]}  /tmp/node.tar.xz" | sha256sum -c - && tar -xJf /tmp/node.tar.xz -C /usr/local --strip-components=1 && \\
    rm /tmp/node.tar.xz && npm install -g {" ".join(f"{p}@{v}" for p, v in AGENT_VERSIONS.items())} && npm cache clean --force && \\
    rm /usr/local/bin/opencode
'''
    + "".join(f"COPY agents/{name} {where}\n" for name, (where, _) in AGENT_FILES.items())
    + "RUN chmod 755 /usr/local/bin/opencode\n"
)
# The entrants' container, the same for every task: tools to build and simulate a room, CPU only. Never the libraries or
# assets we build reference rooms from (tests/test_entrant_image.py keeps a deny list).
TOOLS = (
    """FROM --platform=linux/amd64 python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libegl1 libgomp1 libxrender1 libxi6 libxkbcommon0 \\
        libsm6 libxfixes3 libxxf86vm1 libx11-6 libxext6 libice6 libxkbcommon-x11-0 libopengl0 xz-utils curl tmux asciinema && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 && \\
    pip install --no-cache-dir genesis-world==1.4.3 pysplishsplash==2.18.1 taichi==1.7.4 warp-lang==1.18.0 newton==1.6.1 \\
        mujoco==3.14.0 numpy==2.5.3 scipy==1.18.1 trimesh==5.1.1 pillow==11.3.0
# Blender 5.2 as its Python module: bpy needs Python 3.13 and Taichi at most 3.12, so bpy gets its own Python. It also
# gets MuJoCo, NumPy, SciPy, trimesh and Pillow (trimesh reads OBJ files with it) at the versions above, so one script
# can build in Blender and load the scene in MuJoCo, as the appearance verifier's renderer does (tests/test_entrant_image.py
# keeps the versions the same)
RUN pip install --no-cache-dir uv && uv venv /opt/bpy --python 3.13 && uv pip install --python /opt/bpy/bin/python bpy==5.2.2 \\
        mujoco==3.14.0 numpy==2.5.3 scipy==1.18.1 trimesh==5.1.1 pillow==11.3.0 && \\
    printf '#!/bin/sh\\nexec /opt/bpy/bin/python "$@"\\n' > /usr/local/bin/blender-python && chmod +x /usr/local/bin/blender-python
ENV GS_BACKEND=cpu PYGLET_HEADLESS=true PYOPENGL_PLATFORM=egl QD_NUM_THREADS=4
"""
    + AGENTS
    + """COPY check_scene.py vocabulary.json time_left.py budget.json /opt/scenebench/
RUN chmod +x /opt/scenebench/time_left.py && ln -s /opt/scenebench/time_left.py /usr/local/bin/time_left
"""
)
ENV_DOCKERFILE = (
    TOOLS
    + """COPY pictures /task/pictures
COPY docs /task/docs
COPY instruction.md room.txt categories.txt /task/
RUN mkdir -p /output/scene
WORKDIR /task
"""
)
CHECK_DOCKERFILE = CHECKER + "COPY test.sh /tests/test.sh\n"  # the format check needs MuJoCo only
CHECK_TEST = """#!/bin/bash
# Format-only grader of the public task: the official scores come from the private grader.
python /opt/scenebench/check_scene.py /output/scene --vocabulary /opt/scenebench/vocabulary.json --reward /logs/verifier/reward.json || true
"""
GRADER_DOCKERFILE = """FROM --platform=linux/amd64 {base}
ENTRYPOINT []
ENV SBV_REF_CACHE=/opt/scenebench/reference_renders
COPY room /tests/room
COPY case.json /tests/
COPY pictures /tests/pictures
# the reference room rendered once, while the image builds, so no grading run renders it again
RUN cd /opt/scenebench && .venv/bin/python -m sbv.appearance.views /tests/case.json --pictures {pictures}
COPY test.sh /tests/
"""
GRADER_TEST = """#!/bin/bash
# Private grader: every 3DSceneBench verifier on the copy in /output/scene, scores to /logs/verifier/reward.json.
# reward, the headline Harbor ranks by: the mean of appearance, geometry and functionality (the mean of its four
# levels), each 0 to 100 (sbv/reward.py), in the input-only setting, at every instruction level.
cd ${{SBV_HOME:-/opt/scenebench}}
OUT=${{SBV_LOGS:-/logs/verifier}}; TESTS=${{SBV_TESTS:-/tests}}; SCENE=${{SBV_SCENE:-/output/scene}}; mkdir -p "$OUT" || exit 1
# Harbor gives each attempt a fresh verifier directory. Manual runs must do the same.
for prior in reward.json report.json nonrigid_report.json package_report.json; do
    if [[ -e "$OUT/$prior" ]]; then
        echo "Refusing to reuse $OUT/$prior; set SBV_LOGS to a fresh directory for this attempt." >&2
        exit 1
    fi
done
${{SBV_PYTHON:-.venv/bin/python}} -m sbv.evaluate "$TESTS/case.json" "$SCENE" --out "$OUT/report.json" --procs {cpus} --appearance --pictures {pictures} > "$OUT/summary.txt" 2>&1
"""
# The reward is written once, after the last grading step: a grader stopped during a later step (out of memory, out of
# time) then leaves no reward.json, never an earlier step's reward without the units that were still to run. A report
# that is missing counts as that step's grader failure (sbv/pipeline.py reward_file()).
REWARD_TEST = """${{SBV_PYTHON:-.venv/bin/python}} -c "
import sys; sys.path.insert(0, '.'); from sbv import pipeline
sys.exit(pipeline.reward_file('$OUT/report.json', '$OUT/reward.json', NONRIGID_REPORT, PACKAGE_REPORT))"   # a grader failure: status grader_failed, no reward
"""
# Last in every grader (after the non-rigid step too): signs of cheating for a person to read (sbv/audit.py). It never
# changes the reward, and the script still exits with the reward step's status, so a grader failure stays a failure.
AUDIT_TEST = """STATUS=$?
PICS=(); [ -d "$TESTS/pictures" ] && PICS=(--pictures "$TESTS/pictures")
${{SBV_PYTHON:-.venv/bin/python}} -m sbv.audit "$SCENE" --case "$TESTS/case.json" "${{PICS[@]}}" --out "$OUT/audit.json" > "$OUT/audit.txt" 2>&1
exit $STATUS
"""
# The instruction, laid out as an Agents' Last Exam task card (rdi-berkeley/agents-last-exam, task_card.json): Task
# prompt, Input files, Output files, Evaluation, in that order, each a section of instruction.md; write_card() adds the
# card's Agent routine and Reference files (the files the entrant hands in) in task_card.json. Each task lists only the
# inputs it gives. instruction() puts the pieces below together.
TASK_TEXT = """### Task

Build a functional and reusable world of the room in your pictures, {as_one} The verifiers (the programs that score
your world) will look at it, act on it and run tasks in it. Your world must:

- **Have the same objects** as the room, at the same sizes and in the same places.
- **Behave like the room** when the verifiers act on it. They push, tilt and pick up objects, drop things into them and
  load them. They open and close doors, drawers, lids, racks and hatches.
- **Look like the room** from viewpoints that are not in your pictures.

> If looks and behaviour pull apart, choose behaviour. A hollow cup with the right weight scores better than a
> perfect-looking solid one. Build one scene that explains every picture. Do not make a different model for each
> camera or each configuration. You build only the world: the verifiers bring their own forces, probes and robot."""
RIGID_GOAL = "as one MuJoCo scene."
NONRIGID_GOAL = (
    "as one scene in two files: `scene.xml`, run in MuJoCo,\nand `nonrigid.json` for its non-rigid things, run in "
    'Genesis (see "Non-rigid objects").'
)
GIVEN_L1 = (
    "This task also gives `inventory.json`, `measurements.json` and `depth/`. They tell you the objects, their\n"
    "  sizes and where they stand. Follow them. "
)
REQUIREMENTS = """### Requirements

- **Units and frame.** Use metres and kilograms, with z pointing up and the floor at z = 0. Use the cameras' frame: a
  camera placed as `cameras.json` says must see your world the way its picture shows the room.
- **Names.** Make each object one body directly under `<worldbody>`. Name it `<category>__<n>`, with a category word
  from `/task/categories.txt` and n counting from 0.
- **Wrong words.** If you name an object with the wrong word, it can still count as the reference object. That happens only
  if it stands where the reference object stands, is within 3 times its size, and your word is under the same heading in
  `categories.txt`. Its word is then reported as wrong. In every other case, an object with a wrong or missing word
  counts as missing.
- **One object** is a thing you move or use as a whole. Anything that rests on it, hangs on it or stands in it is a
  separate object. Keep dishes apart from their rack and contents apart from their drawer, and make each book in a
  pile its own object.
- **Fixed or movable.** Make an object fixed (no `<freejoint/>`) only if it is attached to the building, mounted on a
  wall or the ceiling, or connected to the building's gas, water, drain or exhaust. Everything else is movable: give
  its body a `<freejoint/>`, however heavy it is.
- **Moving parts.** Any part a person opens, pulls out, presses or turns must be a child body of its object with one
  joint. Put the joint at the real axis and pivot, give it the real range and stops, and start it where the pictures
  show it. Use a hinge for parts that swing or turn (doors, lids, flaps, knobs, dials). Use a slide for parts that
  slide or are pressed (drawers, sashes, pull-out trays, buttons, keys). Each part must move through its whole range
  when pushed or pulled. Animation does not count. Everything else is rigid.
- **Given values.** {given}We never give weights or friction. Estimate them from what the pictures show.
- **Walls, floor and ceiling.** Make them geoms directly on `<worldbody>`, or bodies named `wall`, `floor`, `ceiling`,
  `room`, `architecture`, `window` or `door_frame` (a number may follow, as in `wall_2` or `floor__0`), or bodies
  whose names start with `room_geometry`. The floor must collide, with its top at z = 0.
- **Front.** Point each object's own x axis out of its front.
- **At rest.** Everything must start at rest, touching what holds it up. Leave no gap over 1 cm and no overlap over 2 mm.
- **Collision shapes.** Keep what matters for using the object. Build concave things from convex pieces. Keep cups,
  bowls and drawers hollow, and keep openings, rims, handles and shelves. Put collision-only geoms in group 3 and
  visible geoms in groups 0 to 2.
- **Physical values.** The verifiers use your masses, friction, balance points, damping and joint ranges exactly as
  you write them. They use their own engine settings and reset any `<option>` in your file.
- **Colours.** Write each object's colour, or an image texture with texture coordinates, into the file. The renders use
  only what `scene.xml` defines, under the verifiers' own lighting. A scene without materials renders grey."""
RULES = """### Rules

- **No internet.** This container can reach only the model APIs of the agents installed in it, and only for the
  agent's own model calls. It is cheating to call a model API from a command or a script, to reach or try to reach any
  other address, or to bring in models, assets or files from outside this container. You must not use our rooms,
  answer keys, or pictures beyond the ones given.
- **No web search or web fetch of any kind.** This includes tools your model's provider runs for you, such as a search
  or fetch tool of your agent or of the model API. A run that uses one fails.
- **Don't make the task easier.** Do not delete required objects, glue loose objects in place, turn off collisions,
  change gravity, or replace a mechanism's motion with an animation.
- **Don't paint the pictures onto boards in front of the cameras.** Appearance is scored from other viewpoints.
- **Missing objects.** A world with objects missing is still scored on what it has. A file that does not load scores 0
  on every part of the reward."""
CHECKING = """### Checking

You can run the format checker as often as you like:

```bash
python /opt/scenebench/check_scene.py /output/scene --vocabulary /opt/scenebench/vocabulary.json
```

It tells you whether your world loads and runs, which names are wrong, which word each alias is scored as, and what
failed (for example a missing mesh, centimetres instead of metres, or a scene that blows up). The checker is the only
code of ours in this container. The verifiers, the reference rooms and their calibrated values are not here. Everything the
verifiers do is described below, under "Evaluation" and "How we simulate and render your world", and in
`/task/docs/WHAT_IS_TESTED.md`."""
SOFTWARE = """### Software in this container

This container has a CPU only (no GPU) and no internet. Build with the tools installed here, the files this task gives,
and what you already know.

| Command | What it has |
|---|---|
| `python` (Python 3.12) | MuJoCo 3.14, Genesis, SPlisHSPlasH, Taichi, Warp, Newton, NumPy, SciPy, PyTorch and trimesh |
| `blender-python` (Python 3.13) | Blender 5.2 as a Python module, which also has MuJoCo, NumPy, SciPy and trimesh, so one script can use both |
| `python /opt/scenebench/check_scene.py` | The format checker (see "Checking") |
| `time_left` | How much of the time is left (see "Budget") |"""
SINGLE_CALL = (
    "If your entry is a single model call that cannot write files, reply with the contents of `scene.xml` and nothing\n"
    'else. Use primitive shapes or inline `<mesh vertex="..."/>` assets. Whoever runs the call saves your reply as\n'
    "`/output/scene/scene.xml`."
)
NONRIGID = """### Non-rigid objects

This room also has non-rigid objects. Each one is of a kind named as in graphics simulation: a **fluid** (a drink in a
cup, water in a vase), a **soft body** (a cushion, a pillow, a plush toy) or a **thin shell** (cloth or paper: a towel,
a curtain, a garment hanging up). Two simulators score your world: MuJoCo runs `scene.xml` and Genesis runs
`nonrigid.json`.

```json
{"schema": "nonrigid/v1", "objects": [
  {"name": "<category>__<n>", "kind": "fluid", "on": "<the container's body in scene.xml>",
   "shape": {"type": "cylinder", "pos": [x, y, z], "radius": r, "height": h},
   "material": {"viscosity": "water | oil | syrup", "density": 1000}},
  {"name": "<category>__<n>", "kind": "soft_body", "on": "<the body it rests on, or floor>",
   "shape": {"type": "box", "pos": [x, y, z], "size": [sx, sy, sz], "yaw_deg": a},
   "material": {"youngs_kpa": E, "poisson": 0.3, "density": kg_per_m3}},
  {"name": "<category>__<n>", "kind": "thin_shell", "on": "<the body of the rail it hangs on>",
   "shape": {"type": "draped_sheet", "rail": [x, y, z], "rail_radius": r, "width": w, "front_length": f, "back_length": b},
   "material": {"fabric": "silk | cotton | denim | wool_coat"}}]}
```

- **Positions and sizes.** A position is the centre of the fluid or the soft body at rest, in metres, in the room's
  frame. Sizes are full sizes. A draped sheet hangs over a rail that runs along x, with `front_length` hanging toward
  +y and `back_length` toward -y.
- **Meshes.** A soft body or a thin shell can be a mesh instead: `{"type": "mesh", "file": "<name>.obj", "pos": [x, y, z],
  "yaw_deg": a}`. The OBJ file goes next to `nonrigid.json`. Its own coordinates are turned by `yaw_deg` about z, then
  moved by `pos`. A soft body's mesh must be closed. A thin shell's mesh also gives `"rail": [x, y, z]`, the point on
  the rail or hanger bar it hangs from.
- **What holds them.** The container, support or rail must be a rigid body in `scene.xml`. Its mass includes what it
  holds: a mug's mass includes its coffee. Keep containers hollow so the fluid fits inside.
- **Don't build them twice.** Do not also add these things to `scene.xml` as bodies that collide. To make them show
  in the renders, give them picture-only geoms there (`contype="0" conaffinity="0"`).
- **How the verifiers test them.** The verifiers run the non-rigid objects of every world with the same Genesis
  settings and apply hidden actions. They tip a container, set a weight on a soft body or drop a ball onto it, and
  pull a cloth up. They compare what happens with what happens in the real room, and check each fluid's
  amount and each soft body's size. Each real object is compared with one of yours: the one with its name, else your
  nearest object with its category word and a size within 2 times, else your nearest object of its kind with a size
  within 3 times, at any distance. Your numbers decide the outcome, not your solver.
- **Checking.** The format checker also checks `nonrigid.json`."""
EVALUATION = """## Evaluation

`/task/docs/WHAT_IS_TESTED.md` explains what each part of the score looks at and what it depends on.

| Part | What it checks |
|---|---|
| **Loads and runs** (the gate) | Your file loads in MuJoCo 3.14 and steps without blowing up. A file that does not load scores 0 on every part. If the scene blows up, the verifiers switch off the objects that cause it and score the rest. |
| **Geometry** | Which objects are there. Their shapes, positions, directions and sizes. What rests on or sits next to what. Whether everything is held up and inside the room. Where a robot base can drive. |
| **Functionality**, in four levels (balance, response, affordance, interaction) | How objects react to pushes and tilts. Whether doors, drawers and lids open by force through their whole range. Whether containers keep what is dropped in, and let it fall out when turned over. Whether shelves and drawers carry loads. Whether objects can be grasped. Whether the room's everyday tasks work, done by a robot (the RB-Y1) or by the verifiers' own grasps and forces. |
| **Appearance** | The verifiers render your world from 16 cameras you were not given and compare the renders with renders of the real room. The cameras come in two groups of eight, `same_height` and `high_low`. Each one is tied to one input camera, stands at least 0.5 m, or 15 degrees around the room centre, away from every input camera, and stands at a height inside the room. |

> **Score.** Your headline score is the average of three parts, each from 0 to 100: appearance, geometry and
> functionality. Functionality is itself the average of its four levels. The three parts weigh the same. The headline
> uses the "input only" setting, which leaves out the verifiers whose result depends on things a picture cannot show,
> such as weight. Inside a part, the objects and tasks that matter more weigh more, as `/task/docs/WHAT_IS_TESTED.md`
> says. Each part is also reported on its own.

**What stays hidden:** the reference room, and which objects, forces, angles, order of operations, physics seeds and cameras
each verifier uses.

### How we simulate and render your world

The verifiers score every world with the same tools and settings. MuJoCo and Genesis are installed here at the same
versions, so you can test with them.

| What | How |
|---|---|
| **Rigid objects** (`scene.xml`) | MuJoCo 3.14 on the CPU, 2 ms steps, the implicitfast integrator, elliptic friction cones, impratio 10, every other option at MuJoCo's default. The verifiers reset any `<option>` and `<size>` in your file. |
| **Robot tasks** | The same engine and steps, with the robot model's own solver settings (the Newton solver, up to 100 iterations, impratio 100), in the reference room and in every world. The robot is the Rainbow Robotics RB-Y1 from MuJoCo Menagerie. |
| **Fluids, soft bodies and thin shells** (`nonrigid.json`, in rooms that have them) | Genesis 1.4.3 on the CPU with 4 threads. This container is set to the same 4 threads, so a fluid you test here pours as it does in the grader. Fluids are SPH particles (6 mm; a unit may use coarser ones), soft bodies MPM (8 mm), thin shells PBD (15 mm), all with 2 ms steps. Each hidden action may use at most 10 minutes of computing. A run that takes longer fails. |
| **Looks** | Blender 4.2's Cycles renderer on the CPU (this container has Blender 5.2), 512 x 512 pixels, 64 samples, one ceiling light and a grey background, from the 16 cameras you were not given. |

Whatever you build with, the verifiers score only the files in `/output/scene`, with these tools."""
EXTRA_DESC = (
    "{k} more picture{s}. Each is aimed at objects the main pictures miss or show poorly. Start from the main "
    "pictures, and look here when they are not enough."
)
L1_PATHS = {
    "inventory.json": "`/task/inventory.json`",
    "measurements.json": "`/task/measurements.json`",
    "depth/": "`/task/depth/`",
}


def table(head, rows):
    """A Markdown table."""
    return "\n".join(
        ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in rows]
    )


def picture_span(names, folder):
    """The pictures' paths in one cell: `/task/<folder>/input_0.png` … `input_7.png` for a run, else each one."""
    nums = [int(n.rsplit("_", 1)[1]) for n in names]
    base = names[0].rsplit("_", 1)[0]
    if len(names) > 2 and nums == list(range(nums[0], nums[0] + len(nums))):
        return f"`/task/{folder}/{names[0]}.png` … `{names[-1]}.png`"
    return ", ".join(f"`/task/{folder}/{n}.png`" for n in names)


def extra_names(pics_dir):
    """The extra pictures in pics_dir/extra (sbv/appearance/extra.py), [] when there are none."""
    p = os.path.join(pics_dir, "extra", "cameras.json")
    return [c["name"] for c in json.load(open(p))["cameras"]] if os.path.exists(p) else []


def outputs(nonrigid):
    """(path, format, required, description) of each file the entrant hands in: the card's reference files."""
    scene = "At the folder's root, with every mesh and texture it uses, given by relative paths."
    if nonrigid:
        scene += (
            " It holds the room, the furniture and every rigid object, including the containers, supports and rails "
            "that the non-rigid things use. Scored in MuJoCo 3.14, as in every room."
        )
    rows = [("/output/scene/scene.xml", "MJCF", "Yes", scene)]
    if nonrigid:
        rows.append(
            (
                "/output/scene/nonrigid.json",
                "json",
                "Yes, in this room",
                "Each fluid, soft body and thin shell (format below), with any OBJ mesh it names next to it. Scored in Genesis "
                "on the CPU, with the part of your `scene.xml` around each object held still.",
            )
        )
    return rows + [
        (
            "/output/scene/manifest.json",
            "json",
            "Optional, not scored",
            "Each object's asset source and the physical values you chose, with a reason.",
        ),
        ("/output/scene/build.log", "text", "Optional, not scored", "What you checked and what still fails."),
    ]


def task_table(said):
    """The room's tasks (room_tasks()) as a numbered table under its first line, or the sentence for a room without."""
    head, items = said.split("\n")[0], [x[2:] for x in said.split("\n") if x.startswith("- ")]
    return head + ("\n\n" + table(["#", "Task"], [(str(i + 1), t) for i, t in enumerate(items)]) if items else "")


def instruction(room, names, extra, nonrigid, said, budget, level_block="", given=(), about=""):
    """(instruction.md, its parts for the card). names: the main pictures; extra: the extra ones; said: room_tasks();
    about: room_about(); level_block: the L1 "what the verifiers do in this room" text (levels.level()); given: the
    level's files, keys of levels.FILES."""
    main = "main " if extra else ""
    pics = [
        (
            picture_span(names, "pictures"),
            "png",
            f"{len(names)} {main}picture{'' if len(names) == 1 else 's'} of the room.",
        ),
        (
            "`/task/pictures/cameras.json`",
            "json",
            "Each camera's intrinsics and camera-to-world pose. The camera axes are OpenGL's (-z forward, +y up). The world is in metres, with z up and the floor at z = 0.",
        ),
    ]
    if extra:
        pics += [
            (
                picture_span(extra, "pictures/extra"),
                "png",
                EXTRA_DESC.format(k=len(extra), s="" if len(extra) == 1 else "s"),
            ),
            ("`/task/pictures/extra/cameras.json`", "json", "The extra pictures' own cameras, in the same format."),
        ]
    pics += [
        ("`/task/categories.txt`", "text", "The category words for naming objects, the same list in every task."),
        (
            "`/task/docs/WHAT_IS_TESTED.md`",
            "markdown",
            "What the verifiers look at in your world, part by part, and what each part depends on.",
        ),
    ]
    pics += [(L1_PATHS[k], "json", L.FILES[k]) for k in L.FILES if k in given]  # inventory, measurements, depth
    prompt = [
        TASK_TEXT.format(as_one=NONRIGID_GOAL if nonrigid else RIGID_GOAL),
        "### Tasks done in your world\n\n" + (about + "\n\n" if about else "") + task_table(said),
        REQUIREMENTS.format(given=GIVEN_L1 if given else ""),
    ]
    if level_block:
        prompt.append("### What the verifiers do in this room\n\n" + level_block)
    prompt += [RULES, CHECKING, "### Budget\n\n" + budget_text(*budget)]
    parts = dict(
        prompt="## Task prompt\n\n" + "\n\n".join(prompt),
        inputs="## Input files\n\nThis task gives these files. Anything not named here is not given.\n\n"
        + table(["Path", "Format", "Description"], pics)
        + "\n\n"
        + SOFTWARE,
        outputs="## Output files\n\nPut your functional and reusable world in `/output/scene/`. Nothing outside that folder is scored.\n\n"
        + table(
            ["Path", "Format", "Required", "Description"], [(f"`{p}`", f, r, d) for p, f, r, d in outputs(nonrigid)]
        )
        + "\n\n"
        + SINGLE_CALL
        + ("\n\n" + NONRIGID if nonrigid else ""),
        evaluation=EVALUATION,
    )
    title = f"# 3DSceneBench: build room {room} as a functional and reusable world in MuJoCo" + (
        " and Genesis" if nonrigid else ""
    )
    return title + "\n\n" + "\n\n".join(parts.values()) + "\n", dict(parts, title=title[2:], inputs_rows=pics)


def routine(names, extra, nonrigid, given, budget):
    """The card's agent routine: the prompt in five steps or fewer (Agents' Last Exam agentMustDo), never shown to the
    agent."""
    files = ", with `inventory.json`, `measurements.json` and `depth/`," if given else ""
    look = (
        f"; look at the {len(extra)} extra pictures in `/task/pictures/extra/` where the main ones are not enough"
        if extra
        else ""
    )
    time = f"{budget[0] // 60} hour{'' if budget[0] == 60 else 's'}" if budget[0] % 60 == 0 else f"{budget[0]} minutes"
    steps = [
        f"Study the {len(names)} {'main ' if extra else ''}picture{'' if len(names) == 1 else 's'} in `/task/pictures/` and their "
        f"`cameras.json`{files} to find every object, its size and its place{look}.",
        "Build the room as `/output/scene/scene.xml`: each object one body named `<category>__<n>` from `/task/categories.txt`, "
        "moving parts on real joints, hollow containers, colours, and masses and friction estimated from the pictures.",
    ]
    if nonrigid:
        steps.append(
            "Describe the room's non-rigid things in `/output/scene/nonrigid.json`, with picture-only geoms for them in `scene.xml`."
        )
    return steps + [
        "Run the format checker until the world loads and runs, and test it with the MuJoCo"
        + (" and Genesis" if nonrigid else "")
        + " installed in the container.",
        f"Stop within {time} or {budget[1]:,} output tokens, whichever comes first (`time_left` shows the time left): the scene in `/output/scene` at that moment is scored.",
    ]


def write_card(out, org, name, parts, names, extra, nonrigid, given, budget):
    """task_card.json at the task's root, in the fields of an Agents' Last Exam card (taskPrompt is instruction.md). Not
    in the container: the agent reads instruction.md only."""
    plain = lambda x: x.replace("`", "")
    card = dict(
        taskId=f"{org}/{name}",
        title=parts["title"],
        summary="Build a functional and reusable world of the room in the pictures: the same objects, the same behaviour when "
        "the verifiers act on it, the same looks from viewpoints not given.",
        category="3D, Animation & Interactive Media",
        software=["MuJoCo 3.14"] + (["Genesis 1.4.3"] if nonrigid else []) + ["Blender 5.2", "Python"],
        taskPrompt=open(os.path.join(out, "instruction.md")).read(),
        agentMustDo=routine(names, extra, nonrigid, given, budget),
        inputFiles=[
            dict(
                name=plain(p).split(" ")[0].rstrip(",").split("/")[-1] or plain(p),
                format=f,
                path=plain(p),
                description=d,
            )
            for p, f, d in parts["inputs_rows"]
        ],
        referenceFiles=[
            dict(name=p.split("/")[-1], format=f, path=p, description=(d if r == "Yes" else f"{r}: {d}"))
            for p, f, r, d in outputs(nonrigid)
        ],
        evaluation=parts["evaluation"].split("\n", 2)[2],
    )
    with open(os.path.join(out, "task_card.json"), "w") as f:
        json.dump(card, f, indent=1, ensure_ascii=False)


NONRIGID_TEST = """${{SBV_PYTHON:-.venv/bin/python}} -m sbv.nonrigid.verify evaluate "$TESTS/nonrigid_case.json" "$SCENE" --out "$OUT/nonrigid_report.json" --procs 1 > "$OUT/nonrigid_summary.txt" 2>&1
"""
# A room's package tasks (sbv/package_tasks, F4.7) run after the non-rigid step, on the copy's matched objects. Without
# a package_report.json the reward step reports grader_failed, never a score.
PACKAGE_TEST = """${{SBV_PYTHON:-.venv/bin/python}} -m sbv.package_tasks.verify evaluate "$TESTS/package_case.json" "$SCENE" --report "$OUT/report.json" --out "$OUT/package_report.json" --procs {cpus} > "$OUT/package_summary.txt" 2>&1
"""
# Every task may be done by the robot or by the verifiers' own grasps and forces in the simulator, and the instruction
# never says which: the entrant makes each one work both ways. Hidden tasks may come on top of the listed ones.
CASE_NONE = (
    "No tasks are listed for this room. Besides the verifiers' checks, hidden tasks may still be done in your world, by "
    "a robot (the RB-Y1) or by the verifiers' own grasps and forces in the simulator."
)
CASE_TASKS = (
    "Besides the verifiers' checks, these tasks will be done in your world. Each one may be done by a robot (the RB-Y1) "
    "or by the verifiers' own grasps and forces in the simulator. We don't tell you which, so make sure each task works "
    "both ways. The forces, positions and checks they use stay hidden, and hidden tasks beyond these may be done too.\n"
)
TASK_WORDS = dict(
    robot="Pick up the {0} and put it on top of the {1}.",
    robot_push="Push the {0} 20 cm across the top it stands on.",
    robot_shelve="Stand the {0} that lies flat upright, spine out, in the empty bay of the {1}.",
)
CATEGORIES = Words.CATEGORIES


def categories(path=CATEGORIES):
    """The public category words, aliases included, the same in every task, so no task reveals its room's inventory."""
    return sorted({w for m, al in Words.word_groups(path).items() for w in [m] + al})


def vocabulary(words):
    """The checker's vocabulary file: the words, and each alias with the word it is scored as
    (checker/check_scene.py)."""
    return {"words": list(words), "aliases": {a: m for a, m in Words.ALIAS.items() if a in set(words)}}


WORDS_HEAD = """These are the category words for naming objects, as <category>__<n> (for example cup__0). The list is the same
in every 3DSceneBench task, so it says nothing about what this room holds. Each line is one kind of thing: use its first
word. The words after "also:" are other words for the same thing, and each is scored as the first word. Each heading
groups related kinds of things. If you use a wrong word, the object can still count by its place, but only if your
word is under the same heading as the right one.
"""


def word_file(path=CATEGORIES):
    """The words as every task gives them, in /task/categories.txt: one kind of thing per line under its topic, as in
    bookcase (also: bookshelf)."""
    groups, topics = Words.word_groups(path), Words.word_topics(path)
    lines = [WORDS_HEAD]
    for t in dict.fromkeys(topics.values()):
        mains = sorted(m for m in groups if topics.get(m) == t)
        if mains:
            lines += [t] + [f"  {m}" + (f" (also: {', '.join(groups[m])})" if groups[m] else "") for m in mains] + [""]
    return "\n".join(lines)


def room_tasks(case, package=None, reword=None):
    """This room's own robot tasks in words, for its instruction: the tasks its builder named in tasks.json that
    calibration kept. The builder's sentence ("words") when given, else one made from the objects' categories, so the
    reference room's object names never appear. package: the room's package_case.json, whose tasks follow in their
    words. reword: {a task's sentence in the frozen case: the sentence the instruction gives instead}, for wording made
    clearer after calibration; the case itself, and so the grader, stay as they were. A sentence in it that no task of
    the room has is an error, so a stale entry cannot pass unseen."""
    reword = dict(reword or {})
    used = set()

    def said(text):
        if text in reword:
            used.add(text)
            return reword[text]
        return text

    cat = {o["slot"]: o["category"].replace("_", " ") for o in case.get("objects", [])}
    lines = []
    for u in case.get("functionality", {}).get("units", []):
        if not u.get("named") or (u["kind"] not in TASK_WORDS and not u.get("words")):
            continue  # declared functions without words are not tasks to announce
        text = said(
            u.get("words")
            or TASK_WORDS.get(u["kind"], "A robot task with the {0}.").format(
                *(cat.get(n, "object") for n in u["slots"]), "object"
            )
        )
        if "__" in text:
            raise ValueError(
                f"task {u['id']}: its words name an object by its id in scene.xml; describe the object instead"
            )
        lines.append(f"- {text}")
    extra = [said(u["words"]) for u in json.load(open(package))["units"]] if package else []
    for text in extra:
        if "__" in text:
            raise ValueError(
                f"package task: its words name an object by its id in scene.xml; describe the object instead"
            )
        lines.append(f"- {text}")
    stale = sorted(set(reword) - used)
    if stale:
        raise ValueError(f"the rewording names {len(stale)} sentence(s) that no task of this room has: {stale[0]!r}")
    return CASE_TASKS + "\n".join(lines) if lines else CASE_NONE


ABOUT_FILE = "about.md"


def room_about(room_dir):
    """What kind of room this is, in its builder's words (about.md next to its scene), for the head
    of the instruction's task list, so the tasks make sense: "" when the room has none. A .md file, so writing one never
    changes the room's calibration inputs."""
    p = os.path.join(room_dir, ABOUT_FILE)
    if not os.path.exists(p):
        return ""
    with open(p) as f:
        text = " ".join(f.read().split())
    if "__" in text:
        raise ValueError(f"{ABOUT_FILE} names an object by its id in scene.xml; describe the room instead")
    return text


def task_name(rid):
    """A task's name without its level: 3dscenebench-<number>."""
    return f"3dscenebench-{rid}"


def room_ids(path, rooms, *, persist=True):
    """Each room's number ("001", "002", ...), kept in path: numbers already there are reused, and new rooms take the
    next free ones in a random order, so a number says nothing about a room's name. A file from before numbers (random
    codes) is kept beside it once, as room_ids.before_numbers.json, and its rooms are numbered anew."""
    ids = {}
    if os.path.exists(path):
        with open(path) as f:
            ids = json.load(f)
    old = {room: i for room, i in ids.items() if not re.fullmatch(r"\d{3,}", str(i))}
    if old:
        ids = {room: i for room, i in ids.items() if room not in old}
    new = [room for room in dict.fromkeys(list(old) + list(rooms)) if room not in ids]
    secrets.SystemRandom().shuffle(new)
    taken = {int(i) for i in ids.values()}
    n = 0
    for room in new:
        n += 1
        while n in taken:
            n += 1
        ids[room] = f"{n:03d}"
        taken.add(n)
    if persist:
        save_room_ids(path, ids)
    return ids


def save_room_ids(path, ids):
    """Persist a checked ID plan, retaining the pre-number map when migrating it."""
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    if os.path.exists(path):
        with open(path) as f:
            previous = json.load(f)
        if any(not re.fullmatch(r"\d{3,}", str(i)) for i in previous.values()):
            with open(os.path.join(folder, "room_ids.before_numbers.json"), "w") as f:
                json.dump(previous, f, indent=1, sort_keys=True)
    with open(path, "w") as f:
        json.dump(ids, f, indent=1, sort_keys=True)


# Hosts the agent step may reach from a task container: the model APIs of the agents installed in it (AGENTS; OpenRouter
# for OpenCode, which has no model host of its own), and nothing else; setup and the grader have no network at all.
# Agents that call their model from the Harbor process (Harbor's Terminus-2, an entrant's own agent) do not need them.
# `--model-hosts` with no host writes tasks with no network at all.
MODEL_HOSTS = ["api.anthropic.com", "api.openai.com", "openrouter.ai"]


BUDGET_FILE = "budget.json"


def budget(room_dir, cpus):
    """The room's grader budget: budget.json next to its scene, {"grader": {"cpus": 8, "memory_mb":
    16384, "minutes": 150}}, each field optional; the defaults are --grader-cpus, 4 GB per 4 CPUs and 90 minutes. A
    field that is not one of these, or not a positive number, stops the run with the reason: a wrong budget must not
    become a task."""
    d = dict(cpus=cpus, memory_mb=4096 * max(1, cpus // 4), minutes=90)
    p = os.path.join(room_dir, BUDGET_FILE)
    if not os.path.exists(p):
        return d
    with open(p) as f:
        b = json.load(f)
    g = b.get("grader", {})
    bad = sorted(set(b) - {"grader"}) + sorted(f"grader.{k}" for k in set(g) - set(d))
    if bad:
        raise SystemExit(f"{p}: unknown field {bad}; the fields are grader.cpus, grader.memory_mb and grader.minutes")
    for k, v in g.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or int(v) < 1:
            raise SystemExit(f"{p}: grader.{k} must be a positive number, at least 1, not {v!r}")
    return dict(d, **{k: int(v) for k, v in g.items()})


# The agent's budget, the same for every task (every level and room): (wall-clock work minutes, output tokens), the run
# stops at whichever comes first.
# Harbor stops an image build after [environment] build_timeout_sec, 10 minutes unless a task says otherwise, and it
# uses this one value for the grader's image too ([verifier.environment] build_timeout_sec is not read for that). The
# tools image takes longer than 10 minutes on a machine that has never built it, and a private grader renders the
# reference room while it builds, so every task gives an hour.
BUILD_TIMEOUT_S = 3600.0
BUDGET = (180, 10_000_000)
MARGIN_S = 15 * 60  # Harbor's timeout: the budget's minutes plus a margin, behind the run infrastructure's own stop
# The agent's container, the same for every task. 16 GB.
AGENT_CPUS, AGENT_MEMORY_MB = 2, 16384
# The public task's format check.
CHECK_CPUS, CHECK_MEMORY_MB = 1, 8192


def budget_text(minutes, tokens):
    """The instruction's Budget section: the same words in every task."""
    time = f"{minutes // 60} hour{'' if minutes == 60 else 's'}" if minutes % 60 == 0 else f"{minutes} minutes"
    return (
        f"You can work for up to {time} or generate up to {tokens:,} output tokens (your reasoning included), whichever\n"
        "comes first. Then the run stops, and the verifiers score the scene that is in `/output/scene` at that moment.\n"
        "Run `time_left` to see how much time is left. It never shows more than you have. "
        f"You have {AGENT_CPUS} CPUs and {AGENT_MEMORY_MB // 1024} GB of memory."
    )


def parse_budget(g):
    """--budget MINUTESm:TOKENS, e.g. 180m:10000000."""
    try:
        m, t = g.split(":")
        m, t = int(m.removesuffix("m")), int(t)
        if m <= 0 or t <= 0:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError(f"{g!r}: give MINUTESm:TOKENS, e.g. 180m:10000000")
    return m, t


def task_toml(
    org,
    name,
    room,
    pictures,
    grader,
    cpus,
    extra_s=0,
    model_hosts=MODEL_HOSTS,
    memory_mb=None,
    minutes=90,
    budget=BUDGET,
):
    """The container has no network; the agent step may reach only model_hosts (MODEL_HOSTS; none: no network at all). A
    private task's grader gets the room's budget (cpus, memory_mb, minutes); the agent gets the budget (work minutes,
    output tokens), the same for every task."""
    agent = f'network_mode = "allowlist"\nallowed_hosts = {json.dumps(list(model_hosts))}\n' if model_hosts else ""
    ver = (
        f"[verifier.environment]\ncpus = {cpus}\nmemory_mb = {memory_mb or 4096 * max(1, cpus // 4)}\nstorage_mb = 20480\nbuild_timeout_sec = 3600.0\n"
        if grader
        else f"[verifier.environment]\ncpus = {CHECK_CPUS}\nmemory_mb = {CHECK_MEMORY_MB}\n"
    )
    return (
        f'schema_version = "1.3"\nartifacts = ["/output/scene"]\n\n[task]\nname = "{org}/{name}"\n'
        f'description = "Build room {room} as a functional and reusable world in MuJoCo from {pictures} picture(s)"\nkeywords = ["scenebench", "3d", "simulation"]\n\n'
        f'[metadata]\nroom = "{room}"\npictures = {pictures}\nbudget_minutes = {budget[0]}\nbudget_output_tokens = {budget[1]}\n\n'
        f"[agent]\ntimeout_sec = {60.0 * budget[0] + MARGIN_S}\n{agent}\n"
        f'[environment]\ncpus = {AGENT_CPUS}\nmemory_mb = {AGENT_MEMORY_MB}\nbuild_timeout_sec = {BUILD_TIMEOUT_S}\nnetwork_mode = "no-network"\n\n[verifier]\nenvironment_mode = "separate"\ntimeout_sec = {60.0 * minutes + extra_s if grader else 600.0}\n'
        f'network_mode = "no-network"\n\n{ver}'
    )


ENTRANT_DOC = os.path.join(
    "docs", "WHAT_IS_TESTED.md"
)  # the one doc a task gives, at /task/docs/ (the instruction points to it)
PRIVATE_FILES = (
    "heldout_cameras.json",
    "input_cameras.json",
    "extra_cameras.json",
)  # a room's own cameras (sbv.appearance.heldout, extra): never public


def write_agent_files(env):
    """The installed agents' settings (AGENT_FILES) into env/agents/, where the image recipe copies them from."""
    os.makedirs(os.path.join(env, "agents"), exist_ok=True)
    for name, (_, text) in AGENT_FILES.items():
        with open(os.path.join(env, "agents", name), "w") as f:
            f.write(text)


def write_task(
    out,
    org,
    name,
    room,
    pictures,
    pics_dir,
    vocab,
    grader=None,
    case=CASE_NONE,
    level_block="",
    nonrigid=None,
    model_hosts=MODEL_HOSTS,
    files=None,
    budget=BUDGET,
    package=None,
    about="",
):
    """room: the room's number, the only room name a task carries. nonrigid: the room's nonrigid_case.json, for rooms
    with nonrigid or fluid objects (adds their section, checks and verifiers). package: its package_case.json, for rooms
    with package tasks (the private grader runs them last; the public task only sees their words). about: room_about().
    grader: (room dir, case, base image, budget) for the private task."""
    shutil.rmtree(
        out, ignore_errors=True
    )  # written afresh: an earlier run's extra pictures or level files must not stay in the task
    os.makedirs(os.path.join(out, "environment"), exist_ok=True)
    os.makedirs(os.path.join(out, "tests"), exist_ok=True)
    names, extra = App.CONDITIONS[pictures], extra_names(pics_dir)
    given = sorted(
        k for k in L.FILES if any(r == k or r.startswith(k) for r in (files or {}))
    )  # the level's files, as levels.FILES names them
    text, parts = instruction(room, names, extra, nonrigid, case, budget, level_block, given, about)
    for d in (out, os.path.join(out, "environment")):
        with open(os.path.join(d, "instruction.md"), "w") as f:
            f.write(text)
    write_card(out, org, name, parts, names, extra, nonrigid, given, budget)
    env = os.path.join(out, "environment")  # the checker is the only code of ours an entrant gets: never sbv
    shutil.copy(os.path.join(ROOT, "checker", "check_scene.py"), env)
    shutil.copytree(pics_dir, os.path.join(env, "pictures"), dirs_exist_ok=True)
    shutil.copy(
        os.path.join(ROOT, "checker", "time_left.py"), env
    )  # /usr/local/bin/time_left: the time left of the budget
    with open(os.path.join(env, "budget.json"), "w") as f:
        json.dump(dict(minutes=budget[0], output_tokens=budget[1]), f)
    os.makedirs(os.path.join(env, "docs"), exist_ok=True)
    shutil.copy(os.path.join(ROOT, ENTRANT_DOC), os.path.join(env, ENTRANT_DOC))
    with open(os.path.join(env, "vocabulary.json"), "w") as f:
        json.dump(vocabulary(vocab), f)
    with open(os.path.join(env, "categories.txt"), "w") as f:
        f.write(word_file())
    with open(os.path.join(env, "room.txt"), "w") as f:
        f.write(room + "\n")  # the room's number, read by the submit-files agent
    write_agent_files(env)
    for rel, data in (files or {}).items():  # a level's extra files (levels.py), copied to /task/
        os.makedirs(os.path.dirname(os.path.join(env, "files", rel)), exist_ok=True)
        with open(os.path.join(env, "files", rel), "w") as f:
            json.dump(data, f)
    with open(os.path.join(env, "Dockerfile"), "w") as f:
        f.write(ENV_DOCKERFILE + ("COPY files /task/\n" if files else ""))
    tests = os.path.join(out, "tests")
    if grader is None:  # public: format-only grader
        for x in ("check_scene.py", "vocabulary.json"):
            shutil.copy(os.path.join(env, x), tests)
        with open(os.path.join(tests, "Dockerfile"), "w") as f:
            f.write(CHECK_DOCKERFILE)
        with open(os.path.join(tests, "test.sh"), "w") as f:
            f.write(CHECK_TEST.replace(" --reward ", " --nonrigid --reward ") if nonrigid else CHECK_TEST)
        leaked = [os.path.join(r, x) for r, _, fs in os.walk(out) for x in fs if x in PRIVATE_FILES]
        if leaked:
            raise SystemExit(
                f"private camera files in a public task: {leaked}"
            )  # inputs go out only as pictures + cameras.json
    else:  # private: the full grader, the reference room inside
        room_dir, case, base, bud = grader
        cpus = bud["cpus"]
        shutil.copytree(room_dir, os.path.join(tests, "room"), dirs_exist_ok=True)
        shutil.copytree(
            pics_dir, os.path.join(tests, "pictures"), dirs_exist_ok=True
        )  # the audit's check for pasted input pictures
        c = {k: v for k, v in case.items() if not k.startswith("_") and k != "case_hash"}
        c["reference"] = "room/scene.xml"
        K.save(c, os.path.join(tests, "case.json"))
        dockerfile, test = GRADER_DOCKERFILE, GRADER_TEST
        if nonrigid:  # the non-rigid verifiers run last, after Appearance
            with open(nonrigid) as f:
                sc = NS.upgrade(json.load(f))
            sc["room"] = "room"  # /tests/room, relative to the case
            with open(os.path.join(tests, "nonrigid_case.json"), "w") as f:
                json.dump(sc, f, indent=1)
            dockerfile, test = (
                dockerfile.replace("COPY case.json /tests/", "COPY case.json nonrigid_case.json /tests/"),
                test + NONRIGID_TEST,
            )
        if package:  # the room's own tasks, their code in /tests/room, frozen by hash
            with open(package) as f:
                pc = json.load(f)
            pc["room"] = "room"
            with open(os.path.join(tests, "package_case.json"), "w") as f:
                json.dump(pc, f, indent=1)
            dockerfile = dockerfile.replace("COPY case.json ", "COPY case.json package_case.json ", 1)
            test += PACKAGE_TEST
        test += REWARD_TEST.replace("NONRIGID_REPORT", "'$OUT/nonrigid_report.json'" if nonrigid else "None").replace(
            "PACKAGE_REPORT", "'$OUT/package_report.json'" if package else "None"
        )
        with open(os.path.join(tests, "Dockerfile"), "w") as f:
            f.write(dockerfile.format(base=base, pictures=pictures))
        with open(os.path.join(tests, "test.sh"), "w") as f:
            f.write((test + AUDIT_TEST).format(cpus=cpus, pictures=pictures))
        with open(os.path.join(out, "PRIVATE"), "w") as f:  # the marker a publish step refuses
            f.write(
                "This task holds a reference room, its case file and the full grader. Never publish it or give it to an entrant.\n"
            )
        sol = os.path.join(out, "solution")
        shutil.copytree(room_dir, os.path.join(sol, "room"), dirs_exist_ok=True)
        with open(os.path.join(sol, "solve.sh"), "w") as f:
            f.write("#!/bin/bash\nmkdir -p /output/scene && cp -r /solution/room/. /output/scene/\n")
        os.chmod(os.path.join(sol, "solve.sh"), 0o755)
    os.chmod(os.path.join(tests, "test.sh"), 0o755)
    runs, extra = 0, 0
    if nonrigid and grader:
        with open(nonrigid) as f:
            runs = nonrigid_seconds(json.load(f)["units"])
    if package and grader:  # each package task may take its own limit, plus a minute
        with open(package) as f:
            extra = sum(u["timeout_s"] + 60 for u in json.load(f)["units"])
    bud = grader[3] if grader else dict(cpus=1, memory_mb=None, minutes=90)
    with open(os.path.join(out, "task.toml"), "w") as f:  # each non-rigid run may take its cap and one retry
        f.write(
            task_toml(
                org,
                name,
                room,
                pictures,
                grader is not None,
                bud["cpus"],
                runs + extra,
                model_hosts,
                bud["memory_mb"],
                bud["minutes"],
                budget,
            )
        )


def nonrigid_runs(units):
    """The Genesis runs the private grader may make, one per unit of a fluid, soft body or thin shell, as
    sbv/nonrigid/verify.py evaluate() picks them: a thin shell's lift counts, an older case's retired cloth swing does
    not (verify.retired()). Old kind names are read."""
    us = [dict(u, kind=NS.kind(u["kind"])) for u in units]
    return [u for u in us if u["kind"] in ("fluid", "soft_body", "thin_shell") and not NV.retired(u)]


def nonrigid_seconds(units):
    """The private grader's time for its non-rigid runs, the same rule for every kind: each run's worst case, its cap
    (verify.copy_cap(): 3 times the true run, at least 10 minutes) plus one retry at RETRY_CAP times the cap, and a
    minute to start. The runs go one after another (--procs 1)."""
    return sum((1 + NV.RETRY_CAP) * NV.copy_cap(u) + 60 for u in nonrigid_runs(units))


def verified_rooms(work):
    """The ready rooms and their cases, after checking that each one's frozen files are all there. A room that declares
    non-rigid objects or package tasks must have the matching frozen case: without it the grader would silently score
    a smaller task. One bad room refuses the whole export, before anything is written.

    A room's receipt is not compared with the present code or machine: the released rooms were frozen on earlier
    commits, and their cases are graded as frozen."""
    rooms = []
    for room in sorted(os.listdir(os.path.join(work, "results"))):
        result = os.path.join(work, "results", room)
        folder = os.path.join(work, "rooms", room)
        check = os.path.join(result, "check.json")
        if not os.path.exists(check):
            continue
        try:
            with open(check) as f:
                record = json.load(f)
            if not isinstance(record, dict):
                raise ValueError("check.json must be an object")
            if not record.get("ready"):
                continue
            if record.get("status") in ("running", "interrupted"):
                raise ValueError(f"its check is marked {record['status']}: the frozen files may be incomplete")
            case = K.load(os.path.join(result, "case.json"))
            if os.path.isfile(NS.side_file(folder)) and not os.path.isfile(NS.side_file(result, "nonrigid_case.json")):
                raise ValueError("the room declares non-rigid objects but its frozen nonrigid_case.json is missing")
            if PkgS.has(folder) and not os.path.isfile(os.path.join(result, "package_case.json")):
                raise ValueError("the room declares package tasks but its frozen package_case.json is missing")
            rooms.append((room, case))
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise SystemExit(f"no tasks written: room {room}: frozen files incomplete: {e}") from e
    return rooms


def check_task_selection(out, selected, ids_path):
    """Refuse old generated task names outside this release; never delete another selection."""
    existing = []
    for side in ("public", "private"):
        folder = os.path.join(out, side)
        if os.path.isdir(folder):
            existing += [
                os.path.join(folder, name)
                for name in sorted(os.listdir(folder))
                if name.startswith("3dscenebench-") and os.path.isdir(os.path.join(folder, name))
            ]
    if existing and not os.path.isfile(ids_path):
        raise SystemExit(
            "no tasks written: an existing task export needs its original room ID map; "
            f"restore {os.path.abspath(ids_path)} or pass the original --room-ids file to preserve room numbers."
        )
    stale = [p for p in existing if os.path.basename(p) not in selected]
    if stale:
        raise SystemExit(
            f"no tasks written: stale generated tasks outside this selection: {stale}. "
            "Remove obsolete task folders explicitly, or use a fresh --out while preserving the original "
            f"room IDs with --room-ids {os.path.abspath(ids_path)}; a new ID map can reassign room numbers."
        )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pictures", nargs="*", default=["8"], choices=list(App.CONDITIONS))
    ap.add_argument(
        "--org", default="pranalabs-ai-team", help="the Harbor Hub organisation the tasks are published under"
    )
    ap.add_argument("--grader-base", default="3dscenebench-grader:dev")
    ap.add_argument("--grader-cpus", type=int, default=8)
    ap.add_argument("--room-ids", help="the room -> number map to keep and reuse (default OUT/private/room_ids.json)")
    ap.add_argument(
        "--model-hosts",
        nargs="*",
        default=MODEL_HOSTS,
        help="hosts the agent step may reach (default: the installed agents' model APIs; none: no network at all)",
    )
    ap.add_argument(
        "--levels",
        nargs="*",
        choices=L.LEVELS,
        help="instruction levels (levels.py); each writes its own task, name suffix -l0, -l1",
    )
    ap.add_argument(
        "--budget",
        type=parse_budget,
        default=BUDGET,
        metavar="MINUTESm:TOKENS",
        help=f"the agent's budget for every task, wall-clock work minutes and output tokens, whichever comes first (default {BUDGET[0]}m:{BUDGET[1]})",
    )
    ap.add_argument(
        "--waiting-words",
        nargs="*",
        default=[],
        metavar="WORDS_TO_ADD_JSON",
        help="rooms' words-to-add.json (the harness's ship-check): their words count as listed, for a look at the tasks "
        "before the release adds them to the list; never for a release (sbv/words.py)",
    )
    ap.add_argument(
        "--rewording",
        metavar="REWORDING_JSON",
        help="{room: {a task's sentence in the frozen case: the sentence the instruction gives instead}}: task wording "
        "made clearer after calibration, without touching the frozen cases or the private tasks",
    )
    a = ap.parse_args(argv)
    rooms = verified_rooms(a.work)
    rewording = json.load(open(a.rewording)) if a.rewording else {}
    unknown_rooms = sorted(set(rewording) - {r for r, _ in rooms})
    if unknown_rooms:
        raise SystemExit(f"no tasks written: the rewording names rooms that are not being built: {unknown_rooms}")
    vocab = categories()
    waiting = set()
    for p in a.waiting_words:
        with open(p) as f:
            waiting |= {e["word"] for e in json.load(f).get("words", [])}
    if waiting:
        print(
            f"preview only: {len(waiting)} waiting words count as listed ({', '.join(sorted(waiting))}); the release adds them to the list first"
        )
    known = set(vocab) | waiting
    unknown = {room: sorted(set(c["vocabulary"]) - known) for room, c in rooms if set(c["vocabulary"]) - known}
    if unknown:
        raise SystemExit(
            f"no tasks written: these rooms use category words that are not in {CATEGORIES}: {unknown}. "
            "The list is public and the same in every task, and only grows: add the waiting words of every room in one "
            "change before the release (sbv/words.py), then run again."
        )
    aliased = {
        room: {
            s: Words.main_word(s.split("__")[0])
            for s in (o["slot"] for o in c.get("objects", []))
            if Words.main_word(s.split("__")[0]) != s.split("__")[0]
        }
        for room, c in rooms
    }
    aliased = {room: al for room, al in aliased.items() if al}
    for room, al in aliased.items():
        # scored the same either way: a case's true words are read as their main words (sbv/common/case.py load()), and
        # so are a copy's, so an entrant may name the object by either word
        print(
            f"note: room {room} names objects by an alias; each is scored as its main word: {al}",
            file=sys.stderr,
        )
    if len(a.pictures) != 1:
        raise SystemExit(
            "no tasks written: one picture condition per run (--pictures 8); the task name does not carry it"
        )
    ids_path = a.room_ids or os.path.join(a.out, "private", "room_ids.json")
    ids = room_ids(ids_path, [r for r, _ in rooms], persist=False)
    chosen = [ids[r] for r, _ in rooms]
    if len(chosen) != len(set(chosen)):
        raise SystemExit(f"no tasks written: duplicate room IDs in {ids_path}; restore the original unique room ID map")
    selected = {task_name(ids[r]) + (f"-{lv}" if lv else "") for r, _ in rooms for lv in a.levels or [None]}
    check_task_selection(a.out, selected, ids_path)
    save_room_ids(ids_path, ids)
    for room, case in rooms:
        rid = ids[room]
        bud = budget(os.path.join(a.work, "rooms", room), a.grader_cpus)
        pc = os.path.join(a.work, "results", room, "package_case.json")  # rooms with package tasks
        package = pc if os.path.exists(pc) and PkgS.has(os.path.join(a.work, "rooms", room)) else None
        try:
            said = room_tasks(case, package, rewording.get(room))
        except ValueError as e:
            raise SystemExit(f'no tasks written for room {room}: {e} (tasks.json "words")')
        try:
            about = room_about(os.path.join(a.work, "rooms", room))
        except ValueError as e:
            raise SystemExit(f"no tasks written for room {room}: {e}")
        if not about:
            print(
                f"warning: room {room} has no {ABOUT_FILE}, so its instruction does not say what kind of room it is ",
                file=sys.stderr,
            )
        for p in a.pictures:
            name = task_name(rid)
            pics = os.path.join(a.out, "pictures", f"{rid}-p{p}")
            key = App.input_key(case)
            keyfile = pics + ".key"
            extra = [
                c["name"] for c in case.get("appearance", {}).get("extra_cameras") or []
            ]  # pictures/extra/ (sbv/appearance/extra.py)
            have = (
                os.path.exists(keyfile)
                and open(keyfile).read() == key
                and os.path.exists(os.path.join(pics, "cameras.json"))
                and all(os.path.exists(os.path.join(pics, n + ".png")) for n in App.CONDITIONS[p])
                and all(os.path.exists(os.path.join(pics, "extra", n + ".png")) for n in extra)
            )
            if not have:  # a new or rebuilt room, changed cameras, or a render that stopped half way
                shutil.rmtree(pics, ignore_errors=True)
                App.input_pictures(case, pics, p)
                App.extra_pictures(case, os.path.join(pics, "extra"))
                with open(keyfile, "w") as f:
                    f.write(key)
            for d in (pics, os.path.join(pics, "extra")):
                if os.path.exists(os.path.join(d, "render.json")):
                    os.remove(os.path.join(d, "render.json"))
            sc = NS.side_file(
                os.path.join(a.work, "results", room), NS.CASE
            )  # rooms with non-rigid objects (old names read too)
            nonrigid = (
                sc if os.path.exists(sc) and os.path.exists(NS.side_file(os.path.join(a.work, "rooms", room))) else None
            )
            gaps = (
                L.audit(os.path.join(a.work, "rooms", room), case) if a.levels else []
            )  # dropped with a reason at calibration
            if gaps:
                print(
                    f"warning: room {room}: declared functions the reference room fails, so no level announces or scores them: {gaps}",
                    file=sys.stderr,
                )
            cams = [
                c
                for d in (pics, os.path.join(pics, "extra"))
                if a.levels and os.path.exists(os.path.join(d, "cameras.json"))
                for c in json.load(open(os.path.join(d, "cameras.json")))["cameras"]
            ]  # L1 depth for every picture, the extra ones too
            for lv in a.levels or [None]:
                n = name + (f"-{lv}" if lv else "")
                block, files = L.level(lv, case, K.reference_path(case), cams) if lv else ("", {})
                kw = dict(
                    case=said,
                    level_block=block,
                    nonrigid=nonrigid,
                    model_hosts=a.model_hosts,
                    files=files,
                    budget=a.budget,
                    package=package,
                    about=about,
                )
                write_task(os.path.join(a.out, "public", n), a.org, n, rid, p, pics, vocab, **kw)
                write_task(
                    os.path.join(a.out, "private", n),
                    a.org,
                    n,
                    rid,
                    p,
                    pics,
                    vocab,
                    grader=(os.path.join(a.work, "rooms", room), case, a.grader_base, bud),
                    **kw,
                )
                print(
                    "wrote",
                    n,
                    "for room",
                    room,
                    f"(agent {a.budget[0]} min, {a.budget[1]:,} output tokens; grader {bud['cpus']} CPUs, {bud['memory_mb']} MB, {bud['minutes']} min)",
                )
    return rooms


if __name__ == "__main__":
    main()
