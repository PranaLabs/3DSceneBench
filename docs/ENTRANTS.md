# Entering 3DSceneBench

This page is for entrants. 3DSceneBench gives you pictures of a room and asks you to build one world of it as a MuJoCo scene. The verifiers (the programs that score your world) then compare it with the room's reference room, the simulated room the pictures came from:

- **Geometry:** what is in it and where.
- **Functionality:** how it behaves when objects are pushed, tilted, opened and handled by a robot, and in the room's own tasks if it has any.
- **Appearance:** how it looks from viewpoints you never saw.
- **Non-rigid objects**, in rooms that have them (fluids, soft bodies, and thin shells such as cloth): how they behave when they are tipped, pressed or pulled.

Contents:
1. What a task gives you
2. What you return
3. Two ways to enter
4. Install
5. Way 1: your agent runs inside the task
6. Way 2: your pipeline builds the worlds, Harbor hands them in
7. Check your world
8. Send your results and read your scores
9. Fair play

---

## 1. What a task gives you

The tasks run on [Harbor](https://github.com/laude-institute/harbor), an open-source task runner. We use v0.23.0. You do not need to know Harbor: this page has every command.

Each task is one room. A room is known only by its number, such as `003`, which says nothing about the room. The task is named by that number: `3dscenebench-003-l0` is room 003. Every task gives 8 main pictures, `input_0.png` to `input_7.png`, plus extra ones (see below).

The input cameras are chosen for each room so that they show the room's objects. `cameras.json` gives the pose of every picture's camera.

**Extra pictures.** Besides the main pictures, a task may give up to 16 more in `/task/pictures/extra/` (`extra_0.png`, ...), with their own `cameras.json` in the same format. That makes 24 pictures at most. Each extra picture is aimed at objects the main pictures miss or show poorly. Start from the main pictures, and look at the extra ones when they are not enough. The instruction says how many extra pictures a task gives. At L1, `depth/` has a file for each extra picture too (`depth/extra_0.json`, ...). The hidden scoring cameras keep the same distance from the extra cameras as from the main ones.

The instruction gives the pictures, the cameras, these rules, and the room's own robot tasks in words. It never gives positions, forces, angles or amounts.

Inside the task's container:

| Path | What it is |
|---|---|
| `/task/instruction.md` | The task text, in four sections: Task prompt, Input files, Output files, Evaluation. Under Task prompt, "Tasks done in your world" lists, in plain words, the tasks your world will be used for, including the room's own tasks if it has any. How they are done stays hidden |
| `/task/docs/WHAT_IS_TESTED.md` | What the verifiers look at in your world, part by part, and what each part depends on. It is the same text as [WHAT_IS_TESTED.md](WHAT_IS_TESTED.md), and the instruction points to it |
| `/task/categories.txt` | The category words for naming objects. It is one fixed list, the same in every task (`sbv/common/categories.txt`), so it does not tell you what the room holds. Each line is one kind of thing, under headings that group related kinds. Some kinds have more than one word, shown as `bookcase (also: bookshelf)`. Every word after "also:" is scored as the first word on its line |
| `/task/pictures/` | The pictures and `cameras.json`: each camera's position, direction and lens. Poses are camera-to-world, with OpenGL camera axes (-z forward, +y up), in metres, with z up and the floor at z = 0 |
| `/task/room.txt` | The room's number, as in the task name |
| `/output/scene/` | Where your answer goes: `scene.xml` with every mesh and texture it uses, and also `nonrigid.json` in rooms with non-rigid objects (section 2) |

Appearance is scored from 16 hidden cameras. Each is tied to one of your input cameras and stands at least 0.5 m, or 15 degrees around the room centre, away from all of them. The same cameras are used for every world. The report warns when your world puts something closer than 0.5 m in front of one of them that the reference room does not have there.

## 2. What you return

Return one folder per room, named after the room's number:

| File | What it is |
|---|---|
| `scene.xml` | Your world as a MuJoCo file (MJCF), at the root of the folder |
| Mesh and texture files | Anywhere inside the folder, named in `scene.xml` by relative paths. Nothing outside the folder, and no absolute paths |

Use the cameras' coordinate frame: a camera placed as `cameras.json` says must see your room the way its picture shows it.

### How to write the file

The verifiers find your objects by their names, and pair them with the reference room's objects by category and position.

| Rule | Example |
|---|---|
| Use metres and kilograms, with z up and the floor at height 0 | A 75 cm table top sits at z = 0.75 |
| Make each object one body directly under `<worldbody>`, named `<category>__<n>`. Use a category word from `/task/categories.txt`, and count n from 0. A word listed after "also:" (an alias) counts as the first word on its line | `cup__0`, `cup__1`, `dining_table__0`. `mug__0` counts as a `cup` |
| One object is a thing that you move or use as a whole. Anything that rests on it, hangs on it or stands in it is a separate object | A cabinet with its doors and drawers is one object. The beaker on the bench is another. A pile of three books is three objects |
| Make an object fixed (no joint) only if it is attached to the building: mounted on a wall or the ceiling, or connected to gas, water, a drain or an exhaust. Everything else is movable and gets a free joint on its body, however heavy it is | Fixed: a wall clock, a sink, a fume hood, a bench with gas taps. Movable: `<body name="cup__0" pos="1.3 1.35 0.74"><freejoint/> ... </body>`, a heavy cabinet standing on the floor |
| Any part a person opens, pulls out, presses or turns must be a child body of its object, with one joint at its real axis, with a range, starting where the pictures show it. Use a hinge for parts that swing or turn (doors, lids, knobs) and a slide for parts that slide or are pressed (drawers, sashes, buttons, keys). Everything else is rigid | `<joint name="drawer_slide" type="slide" axis="1 0 0" range="-0.4 0"/>` |
| Make walls, floor and ceiling geoms directly on `<worldbody>`, or bodies named `wall`, `floor`, `ceiling`, `room`, `architecture`, `window` or `door_frame` (a number may follow, as in `wall_2` or `floor__0`), or bodies whose names start with `room_geometry` | `<geom name="wall_e" type="box" .../>` |
| Point each object's own x axis out of its front | A chair's body frame points +x where a person sitting on it faces |
| Everything starts at rest, touching what holds it up. No gap over 1 cm, no overlap over 2 mm | A book lies on the table, not 3 cm above it |
| Cameras see geoms in groups 0 to 2 that are not transparent. Put collision-only geoms in group 3 | A mug's visual mesh in group 2, its convex collision pieces in group 3 |

### How your objects are paired with the true ones

The verifiers look for each reference object among your objects in three steps:

1. **By word:** the same category word, a similar size (within 2 times), near the reference object's place.
2. **Furniture with moving parts:** a true cabinet or chest still unpaired is looked for among your objects that have joints. So a chest of drawers named `cabinet` still has its drawers tested.
3. **By place:** any word under the same topic of the category list, a similar size (within 3 times), close by (never farther than 1 m).

For same-word pairing, objects named in pick-and-place, push, shelving or activity tasks get extra reach: at least the diagonal of the reference object supporting them. A cup moved along its table can therefore still pair and have its robot task tested.

A reference object that finds no partner is missing, and yours is an extra. An extra lowers the inventory score. A wrong word that still pairs in step 3 is reported as word accuracy, not scored. The exact distances are in [reference/VERIFIERS.md](reference/VERIFIERS.md), section 3.

Some examples. A table lamp a metre beside a floor lamp is not taken for it. A pillow where the reference room has a plant is not taken for the plant, so both are missing. A lamp named `floor_lamp` where the reference room has a `table_lamp` of about its size is still paired, and its physics is tested.

### Physical properties

The verifiers use your masses, friction, balance points, collision shapes, damping and joint ranges exactly as you write them, so estimate them from what the pictures show. The engine settings are the benchmark's own: the verifiers reset any `<option>` and `<size>` in your file.

| Property | What is used | Example |
|---|---|---|
| Mass | The body's mass, from `mass`, `density` or an `<inertial>` | A ceramic mug is about 0.3 kg |
| Friction | The geoms' `friction` (MuJoCo uses the larger value of two touching geoms) | Household things on tables slip at about 0.2 to 0.35: a mug on a wooden table about 0.3, a glass bowl on a stone top about 0.2. A cup on a rubber mat is about 0.8 |
| Balance point | The body's centre of mass | A table lamp's weight sits in its base |
| Collision shape | The geoms that collide. Build concave objects from convex pieces | A mug is hollow, with a base and walls, so a ball dropped in stays inside |
| Doors and drawers | Joint range, damping and stops | A drawer opens 40 cm and stops |

Each world is scored twice: once with every verifier, and once without the verifiers whose result depends on things a picture cannot show, such as weight. The second is the "input only" setting, and the headline score uses it.

### How we simulate and render your world

The verifiers score every world with the same tools and settings. MuJoCo and Genesis are installed in the task's container at the same versions, so you can test with them.

- Rigid objects (`scene.xml`): MuJoCo 3.14 on the CPU, 2 ms steps, the implicitfast integrator, elliptic friction cones, impratio 10, every other option at MuJoCo's default. The verifiers reset any `<option>` and `<size>` in your file.
- Robot tasks: the same engine and steps, with the robot model's own solver settings (the Newton solver, up to 100 iterations, impratio 100), in the reference room and in every world. The robot is the Rainbow Robotics RB-Y1 from MuJoCo Menagerie.
- Fluids, soft bodies and thin shells (`nonrigid.json`, in rooms that have them): Genesis 1.4.3 on the CPU with 4 threads. The task container is set to the same 4 threads, so a fluid you test there pours as it does in the grader. Fluids are SPH particles (6 mm; a unit may use coarser ones), soft bodies MPM (8 mm) and cloth PBD (15 mm), all with 2 ms steps. Each hidden action may use at most 10 minutes of computing. A run that takes longer fails.
- Looks: Blender 4.2's Cycles renderer on the CPU (the container has Blender 5.2), 512 x 512 pixels, 64 samples, one ceiling light and a grey background, from the 16 hidden cameras. The light hangs 5 cm under the reference room's ceiling, or under yours if yours is lower. The reference room is rendered with the light at the same height. Your ceiling counts whether it collides or is picture-only (`contype="0" conaffinity="0"`), as long as a picture-only ceiling covers at least a quarter of the room seen from above.

Whatever you build with, the verifiers score only the files in `/output/scene`, with these tools.

### Non-rigid objects (some rooms)

Some rooms also have non-rigid objects. Their kinds are named as in graphics simulation: fluids in containers, soft bodies such as a pillow or a seat pad, and thin shells (cloth or paper) such as a towel on a rail. The instruction of such a task has a section "Non-rigid objects". Two simulators score these worlds:

| What | Where you describe it | What scores it |
|---|---|---|
| The room, the furniture and every rigid object, including the containers, supports and rails that non-rigid objects use | `scene.xml` | MuJoCo 3.14, as for every room |
| Fluids, soft bodies and thin shells | `nonrigid.json`, next to `scene.xml` | Genesis 1.4.3 on the CPU. Each object runs with the part of your `scene.xml` around it, held still |

You can build and test with any tool in the task's container (section 5). Only the two files are scored. Every world runs with the same Genesis settings, so your numbers decide the outcome, not your solver.

The side file looks like this:

```json
{"schema": "nonrigid/v1", "objects": [
  {"name": "tea__0", "kind": "fluid", "on": "cup__0",
   "shape": {"type": "cylinder", "pos": [1.20, 0.40, 0.80], "radius": 0.035, "height": 0.06},
   "material": {"viscosity": "water", "density": 1000}},
  {"name": "pillow__0", "kind": "soft_body", "on": "bed__0",
   "shape": {"type": "box", "pos": [0.50, 1.10, 0.62], "size": [0.50, 0.30, 0.12], "yaw_deg": 90},
   "material": {"youngs_kpa": 20, "poisson": 0.3, "density": 40}},
  {"name": "towel__0", "kind": "thin_shell", "on": "towel_rail__0",
   "shape": {"type": "draped_sheet", "rail": [2.0, 0.3, 1.0], "rail_radius": 0.012, "width": 0.4, "front_length": 0.35, "back_length": 0.35},
   "material": {"fabric": "cotton"}}]}
```

| Kind | `shape` (metres, in the room's frame) | `material` |
|---|---|---|
| `fluid` | `cylinder`: the fluid at rest. `pos` is its centre | `viscosity`: `water`, `oil` or `syrup`. `density` in kg per cubic metre |
| `soft_body` | `box`: `pos` is its centre, `size` its full sizes, `yaw_deg` its turn about z. Or a closed mesh (below) | `youngs_kpa` (stiffness, in kPa), `poisson`, `density` in kg per cubic metre |
| `thin_shell` | `draped_sheet`: a sheet over a rail that runs along x, with `front_length` hanging toward +y and `back_length` toward -y. Or a mesh (below) | `fabric`: `silk`, `cotton`, `denim` or `wool_coat` |

- `name` follows the same `<category>__<n>` rule as `scene.xml`, and must be unique.
- `on` names a rigid body in `scene.xml`. For a fluid it is the container. For a soft body it is what the body rests on (or `floor`). For a thin shell it is the body of its rail or hanger.
- A soft body or thin shell can be a mesh instead: `{"type": "mesh", "file": "pillow.obj", "pos": [x, y, z], "yaw_deg": a}`. The OBJ file goes next to `nonrigid.json`. Its own coordinates are turned by `yaw_deg` about z, then moved by `pos`. A soft body's mesh must be closed. A cloth mesh also gives `"rail": [x, y, z]`, the point on the rail or hanger bar it hangs from.
- Keep containers hollow, so the fluid fits inside. A container's mass in `scene.xml` includes what it holds: a cup's mass includes its tea.
- Do not also build these objects in `scene.xml` as bodies that collide, or the Genesis run would meet a rigid copy of them. To make them show in the appearance renders, give them picture-only geoms there (`contype="0" conaffinity="0"`).

The verifiers apply hidden actions and compare what happens with the reference room:
- they tip a container and measure how much spills;
- they set a weight on a soft body, or drop a ball on it, and measure how far it presses in;
- they pull a cloth up with a fixed force and measure how far it rises: a cloth more than 15 cm tall hangs, and its hem is pulled; a lower one lies flat, and one edge is pulled.

They also check each fluid's amount and each soft body's size. Each reference object is compared with one of yours: the one with its name, else your nearest object with its category word and a size within 2 times, else your nearest object of its kind with a size within 3 times, at any distance. The rules are in [reference/VERIFIERS.md](reference/VERIFIERS.md), section 4.5. Each object's action in a room (its angle, push, drop height, spot and timing) is drawn for that room and stays hidden.

### Appearance

Write each object's colour, or its image texture with texture coordinates, into `scene.xml`. A texture on a shape without texture coordinates is tiled `texrepeat` times per metre, as MuJoCo does with `texuniform="true"`. The renders use only what the MuJoCo file defines, under the verifiers' own lighting. A scene without materials renders grey.

## 3. Two ways to enter

| You have | Way in | Section |
|---|---|---|
| An agent that works in a terminal (Claude Code, Codex, or your own) | Harbor runs your agent on each task, and its commands run in the task's closed container | 5 |
| A pipeline that runs on your own machine (image to 3D, a scene generator, an agent working on your machine) | You build the worlds first, then our submit-files agent hands them to Harbor | 6 |

The same verifiers score both, so the scores are comparable. Way 1 builds inside our closed container (no internet beyond the agent's own model API, section 5). Way 2 builds on your own machine, which we cannot close. Its results are listed as the **open-build** track, apart from the closed-container results. The grader and the scores are the same, so you can compare how you are doing, but the build conditions are not.

## 4. Install

| What | How | Needed for |
|---|---|---|
| Docker | [Docker Desktop](https://www.docker.com/products/docker-desktop/) on Mac or Windows, Docker Engine on Linux. On an Apple Silicon Mac, turn on "Use Rosetta for x86_64/amd64 emulation" in Docker Desktop's settings | Ways 1 and 2 |
| Harbor | `uv tool install harbor==0.23.0`, or `pip install harbor==0.23.0` where you have pip (Python 3.12 or newer) | Ways 1 and 2 |
| This repository | Clone and enter the repository, then install the core packages as below | Way 2 (the submit-files agent lives here), and for checking files on your machine |

Use Harbor from its Python 3.12-or-newer environment for the `harbor` commands. The separate Python 3.11 environment below holds the tested core verifier and local checker packages:

```bash
git clone https://github.com/PranaLabs/3DSceneBench
cd 3DSceneBench
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements/core.txt
```

With uv instead of the last two lines:

```bash
uv venv -p 3.11 .venv
uv pip install --python .venv/bin/python -r requirements/core.txt
```

Then get the tasks. They are not in this repository: `harbor download pranalabs-ai-team/3dscenebench-mini@v1.0.0`, run from the repository's root, fetches a folder `3dscenebench-mini/` with one sub-folder per task, so a task's instruction is at `3dscenebench-mini/3dscenebench-003-l0/instruction.md`. This is 3DSceneBench mini: seven of the 49 rooms, one from each category, seven tasks. Each task carries its reference room in its `tests/` folder, for grading only; the other 42 rooms are coming soon. Build the grader image once, as "Requirements" in the [README](../README.md#requirements) shows: every task grades with it. The run and submission commands need this folder first.

## 5. Way 1: your agent works in the task's container

Harbor starts each task's container, gives your agent the instruction, and lets it work with 2 CPUs and 16 GB of memory. The run goes on until it reaches the task's budget: a wall-clock work time or a number of output tokens (everything the model generates, its reasoning included), whichever comes first. The budget is the same for every task, whatever its room: 3 hours or 10,000,000 output tokens. The scene in `/output/scene` when the run stops is scored.

The instruction's "Budget" section and `task.toml` (`[metadata]` `budget_minutes` and `budget_output_tokens`) give the task's budget. Inside the container, `time_left` prints the time used and the time left. It counts from the container's start, a little before your agent starts, so it never shows more time than the run really has. Output tokens are counted outside the container, so `time_left` cannot show them.

The container is the same for every task. It has the checker at `/opt/scenebench/check_scene.py` and the `time_left` tool. These are the only code of ours in it: the verifiers, the reference rooms and their calibrated values are not there. It also has these tools, all on the CPU (there is no GPU):

| Tool | For | How to call it |
|---|---|---|
| Blender 5.2 | Modelling, cloth, soft bodies, Mantaflow liquid, renders | `blender-python` (Blender as a Python module, in its own Python 3.13, which also has MuJoCo, NumPy, SciPy and trimesh, so one script can use Blender and MuJoCo) |
| MuJoCo 3.14 | Rigid bodies: the simulator that scores `scene.xml` | `python` (3.12), or `blender-python` |
| Genesis | Fluids, soft bodies, thin shells: the simulator that scores `nonrigid.json` in rooms that have one | `python` |
| SPlisHSPlasH, Taichi, Warp, Newton | Other simulators | `python` |
| NumPy, SciPy, PyTorch, trimesh | Maths, learning, meshes | `python` |

**The container has no internet, except the model APIs of the agents installed in it.** While your agent works, the container can reach `api.anthropic.com`, `api.openai.com` and `openrouter.ai`, and nothing else. While the task starts and while it is graded, it reaches nothing. Those three addresses are for the agent's own model calls only. The installed tools are everything your agent's commands can use. It is cheating to call a model API from a command or a script, to reach or try to reach any other address, or to bring in models, assets or files from outside the container. Only the files in `/output/scene` are scored, with the tools in section 2 ("How we simulate and render your world"), so everything your world needs must be inside that folder.

**Claude Code, Codex and OpenCode** are installed in the container (Claude Code 2.1.309, Codex 0.164.0, OpenCode 1.19.3), so Harbor's `claude-code` and `codex` agents start without installing anything; for OpenCode use the agent in this repository, as below. Their web tools are switched off in the container itself. Claude Code has no WebSearch or WebFetch. Codex runs with web search disabled, image generation off and without its bundled skills (one of them generates images through OpenAI's API). OpenCode has no webfetch, websearch or codesearch. All three can still open the task's pictures. Pass the same settings on the command line too:

```bash
harbor run -p 3dscenebench-mini -a claude-code -m anthropic/claude-opus-5-5 --ak disallowed_tools=WebFetch,WebSearch -n 1
harbor run -p 3dscenebench-mini -a codex -m openai/<model> --ak web_search=disabled -n 1
harbor run -p 3dscenebench-mini -a scenebench_harbor.agents:OpenCode -m openrouter/<model> -n 1
```

**Any other model runs in OpenCode.** The third line is for a model with no agent of its own: OpenCode calls it through OpenRouter with `OPENROUTER_API_KEY`. Run it from this repository's root, so Harbor finds `scenebench_harbor.agents`. Harbor's own `opencode` agent would try to install OpenCode when the task starts, and stop there. For another provider, name it in `-m` as OpenCode does (`<provider>/<model>`) and add `--allow-agent-host <the provider's API host>`.

The key for the first two comes from `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`. Do not pass `--ak version=...`: the container cannot install another version, so the task would stop at setup. To reach your model through a proxy, add `--allow-agent-host <proxy host>` and set `ANTHROPIC_BASE_URL` or `OPENAI_BASE_URL`. The proxy may pass only your agent's model calls. Other agents that install themselves when a task starts (Gemini CLI and others) cannot run here, because the container has no network to install them.

Your own agent: write a class in Harbor's agent style (see Harbor's docs on custom agents). It calls its model from the Harbor process on your machine and runs its commands in the container with Harbor's `environment.exec`. Pass its import path:

```bash
harbor run -p 3dscenebench-mini -a my_package.my_agent:MyAgent -n 1
```

Harbor's Terminus-2 runs too (`-a terminus-2`), but it sees only terminal text, not the task's pictures.

`-n 1` runs one task at a time. Each task is graded when its run ends, and the tests are timed, so two graders running at once can change a result. To run only some tasks, add `-i` with a name pattern: `-i "3dscenebench-003-l0"` runs only room 003.

## 6. Way 2: your pipeline builds the worlds, Harbor hands them in

Step 1. Read the inputs. Every task folder holds:
- its instruction in `3dscenebench-mini/<task>/instruction.md`;
- its pictures in `3dscenebench-mini/<task>/environment/pictures/`;
- what is tested in `3dscenebench-mini/<task>/environment/docs/WHAT_IS_TESTED.md`;
- its room number in `3dscenebench-mini/<task>/environment/room.txt`;

Step 2. Build one world per room into a folder of your own, with one sub-folder per room number:

```
my_copies/
    003/
        scene.xml
        meshes/bed.obj
        textures/wood.png
    019/
        scene.xml
```

Step 3. Check each world (section 7).

Step 4. Hand the worlds in. `PYTHONPATH` tells Harbor where this repository is, so it can find the agent:

```bash
PYTHONPATH=/full/path/to/3DSceneBench harbor run -p 3dscenebench-mini -a scenebench_harbor.submit_files:SubmitFiles --ak folder=/full/path/to/my_copies -n 1
```

The submit-files agent builds nothing. For each task it reads the room's number and copies `my_copies/<number>/` into the task's `/output/scene/`. A room with no sub-folder fails with a clear message.

## 7. Check your world

From this repository:

```bash
.venv/bin/python checker/check_scene.py my_copies/003 --vocabulary 3dscenebench-mini/3dscenebench-003-l0/environment/vocabulary.json
```

For rooms that require `nonrigid.json`, add `--nonrigid` to make missing or invalid declarations fail the command.

Inside a task's container, the same check is `python /opt/scenebench/check_scene.py /output/scene --vocabulary /opt/scenebench/vocabulary.json`.

It prints a short report:
- whether the world loads and runs;
- how many objects it found;
- any names that are not `<category>__<n>`;
- the word each alias is scored as (`bookshelf__0 is scored as bookcase`);
- any word that is not on the list;
- any problems in `nonrigid.json`, when there is one;
- the details of what failed, such as a missing mesh file, centimetres instead of metres, or a scene that blows up when stepped.

What the grader does with a failure:
- A file that does not load scores 0 on every part.
- If the scene blows up, the verifiers switch off (make inert) the objects that cause it and score the rest of the world.

## 8. Read your scores

Each task is graded on your machine when its run ends. Harbor starts a separate grader container, which holds the task's reference room and the verifiers, runs every check in your world and in the reference room, and writes the scores to that trial's `verifier/reward.json`. Your agent's container never holds the reference room.

To grade a saved job again without running your agent again:

```bash
harbor job regrade jobs/<your job> -p 3dscenebench-mini -n 1 -o graded
```

**The leaderboard.** To have your run added, write to **contact@pranalab.ai**.

Each room's report has these scores:

| Score | Range | Meaning |
|---|---|---|
| `reward` | 0 to 100 | The headline score: the average of appearance, geometry and functionality |
| `reward_appearance`, `reward_geometry`, `reward_functionality` | 0 to 100 | The three parts |
| `reward_balance`, `reward_response`, `reward_affordance`, `reward_interaction` | 0 to 100 | The four levels of functionality |
| `parse`, `sim_ready` | 0 or 1 | The files load; the world loads and runs 6 s without blowing up |
| `inventory`, `shape`, `pose_size`, `relations`, `placement` | 0 to 100 | The five parts of geometry |
| `coverage_full` | 0 to 100 | The share of the reference objects your world has |
| `reward_visible` | 0 to 100 | The same reward, over only the reference objects your pictures show |
| `siglip2`, `dinov3`, `lpips` | raw | The image metrics behind appearance, from the hidden cameras |

In rooms with non-rigid objects, `nonrigid_report.json` beside `reward.json` has a row for each of them. In rooms with their own coded tasks, `package_report.json` has a row per task.

Every verifier behind these scores, with its exact rule, is in [reference/VERIFIERS.md](reference/VERIFIERS.md).

## 9. Fair play

The no-internet rule below is for the closed-container track (way 1). Open-build entries (way 2) are scored the same way and listed apart. For a run you ask us to list, we trust the job folder you send. We may rerun some of your tasks ourselves to compare, so keep your agent or pipeline and its settings.

We review every run we list:
- we check the run's settings;
- we look through its trajectory for web, web-fetch and image tool calls, and for model API calls made from commands or scripts;
- a model judges each rewarded run against the task and these rules, and a person reads every finding.

A run caught breaking a rule scores 0 on that task. We do not accept an entry whose settings bring outside material in (mounted folders, extra instructions or system prompts, agent memory, an earlier run's trajectory), turn the web tools back on, or change the task, its timeouts or its resources. A run over the budget is listed for review. Tell us if you think a finding is wrong.

| Allowed | Not allowed |
|---|---|
| The tools installed in the task container | Any internet access from the container beyond your agent's own model calls, or files brought in from outside it (3D models, asset libraries, image-to-3D models) |
| Your model's API, for your agent's own model calls (from the Harbor process, or from Claude Code or Codex in the container) | Calling a model API from your agent's commands or scripts; web search; image generation; turning the installed agents' settings back on |
| Any tool in the container | Our rooms and their answer keys: nothing from a task's `tests/` folder may reach your agent, your pipeline or your world |
| | Pictures of the room beyond those the task gives |
