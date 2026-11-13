# The verifiers

Every check 3DSceneBench runs on a submitted world, and how each one decides. For a plain-words overview, read [../WHAT_IS_TESTED.md](../WHAT_IS_TESTED.md) first.

A verifier always runs twice. First on the reference room ("calibration"), which freezes the expected outcome in the room's case file. Then on each world, which must match it. A world never sees the reference room or its case.

| Part | Verifiers | Question | In `reward.json` |
|---|---|---|---|
| Gate | F1.1 to F1.4 | Does the world load and run? | `parse`, `sim_ready` |
| Appearance | A1 to A3 | Does it look like the room from unseen viewpoints? | `reward_appearance` |
| Geometry | G1 to G3, S0.1, P0 | Are the right objects there, the right shape and size, in the right places? | `reward_geometry` |
| Balance | F2.1, F2.6, S0.3 | Does everything stay put as built? | `reward_balance` |
| Response | F2.2 to F2.4, S1.1 to S1.3 | Does it react like the room when pushed, tilted, tipped or pressed? | `reward_response` |
| Affordance | F3.1 to F3.6 | Do doors open, containers hold, objects lift? | `reward_affordance` |
| Interaction | F4.1 to F4.7 | Can the room's tasks be done in it? | `reward_interaction` |

Verifier numbers are permanent and never reused, so a gap (F2.5, F2.7, F3.4) is a retired verifier. Section 7 says how the parts combine into one `reward`.

## 1. Gate

A world that fails the gate scores 0 on every functionality level. Appearance and geometry still run if the scene loads.

| No. | Verifier | Passes when |
|---|---|---|
| F1.1 | Package | Every mesh, texture and include the scene names exists inside its folder. No absolute paths, nothing outside the folder |
| F1.2 | Units and frame | Metres and z up: median object size between 2 cm and 5 m, scene under 100 m, objects at or above z = 0 |
| F1.3 | Definitions | MuJoCo compiles the scene, and every mass, inertia, size, friction, damping and joint range is finite and allowed |
| F1.4 | Load and step | 6 s of simulation finish without a numerical failure, within 300 CPU seconds |

**One broken object does not fail the whole world.** An object with a bad value, or with hidden constraints its reference object does not have (welds, gravity compensation, contact excludes), fails only its own verifiers. When the 6 s run blows up, the gate reruns it with the most suspect object made inert: it stays in place and touches nothing. Up to three objects can be made inert this way. They fail their verifiers, and the rest of the world is scored. A blow-up that three inert objects do not explain fails the world.

## 2. Appearance

The world is rendered in Blender from 16 held-out cameras and compared with the reference room rendered from the same cameras.

- **Cameras.** Sixteen, in two groups of eight, each linked to one of the eight main input cameras. The default proposal turns each input 22.5 degrees about the room centre: one group at the input's height, one alternately 0.55 m higher and lower. A camera may move along its ray to find a clear spot; if none exists, it turns in 2.5-degree increments to the nearest valid ray. A room may instead supply its own validated camera sets. In either case, every held-out camera stands at least 0.5 m apart on the floor, or 15 degrees about the room centre, from every input camera. The cameras are fixed on the reference room, so a world's geometry never moves them. The entrant gets no pictures from them.
- **Rendering.** Cycles on the CPU, 512 x 512, 64 samples, one ceiling light and a grey background, the same for every scene. Only what the MuJoCo file defines is drawn: colours, image textures, and geoms in groups 0 to 2.
- **The floor.** Any two rooms look somewhat alike to image models. So each room has its own zero point: the reference room's walls, floor and ceiling alone, with no objects, scored against the reference room. For each scored similarity `s` and its floor `f`, the score is `100 * min(1, max(0, (s - f) / max(1 - f, 0.05)))`, rounded to one decimal place. The minimum span of 0.05 limits amplification in nearly empty rooms and can leave even a perfect similarity of 1 below 100 when `f` exceeds 0.95. Without a recorded floor, the score is `50 * (s + 1)`.

| No. | Verifier | How it decides | Scored |
|---|---|---|---|
| A1 | SigLIP-2 similarity | Cosine similarity of image embeddings, mean over the 16 views. Overall look and content | Yes |
| A2 | DINOv3 similarity | Cosine similarity of the CLS token, mean over the 16 views. Outlines and arrangement of parts | Yes |
| A3 | LPIPS distance | Pixel-pattern distance, mean over the 16 views | Reported only |

The same metrics at the input cameras are reported apart and never scored: a world could paint the pictures onto boards in front of those cameras.

## 3. Geometry

Geometry looks at the world as submitted, before any simulation.

**Pairing.** Every per-object check first pairs each reference object with one submitted object, in three steps:

1. **By word.** Same category word, box diagonal within 2 times, centre within the reference object's reach. This is its frozen case gate, or otherwise max(0.3 m, its box diagonal). For an object named in a pick-and-place, push, shelving or activity unit, the reach is at least the diagonal of the reference object supporting it, so a cup moved along its table can still pair.
2. **Furniture with moving parts.** A reference object with doors or drawers, still unpaired, among the world's leftover objects that have joints.
3. **By place.** Any word under the same topic of the category list, sizes within 3 times, centres within the reference object's size, and never farther than min(1 m, max(0.3 m, half its size)).

A reference object with no partner is missing, and every check on it fails. An object found under another word keeps its verifiers; the word is reported as word accuracy, not scored.

**Weights.** Every geometry part is a mean weighted by object. An object a robot task moves or places things on weighs 2. An object with n moving parts weighs 1 + ln(n). Everything else weighs 1.

**Groups.** Five or more small objects of one word under one container (books on a shelf, fries in a tray) are scored as one object: present, about as many (within 30%), in the right region, and at rest.

| No. | Verifier | How it decides | Score |
|---|---|---|---|
| G1.1 | Inventory | F1 = 2M / (objects built + objects required), where M is the reference objects the pairing found. Each extra object counts against it | `inventory` |
| G2.1 | Shape | Surface F-score of each paired object against the true one: 3,000 surface samples, scaled to unit size, best of four quarter turns, within 3% of the diagonal | `shape` |
| G2.3 | Pose and size | Position within 10 cm, each side within a factor of 1.25 of the true length, and a long object's long axis within 30 degrees. Credit is smooth, as described below. A missing object earns 0 | `pose_size` |
| G3.1 | Relations | Each "on", "inside", "next to" and "facing" relation of the reference room holds. An "on" relation also needs contact once the world has settled, so a lamp fixed 5 mm above its desk fails | `relations` |
| G3.2 | Placement | Each object is held up by what is under it (or mounted, as in the reference room), is not sunk more than 2 mm into another, and is inside the room. An extra object that floats or sinks counts against it | `placement` |
| S0.1 | Fluid amount | The fluid's volume is within a factor of 2 of the true one | with pose and size |
| P0 | Soft body and thin shell pose | G2.3 on the declared shape of a non-rigid object | with pose and size |

For rigid G2.3, each applicable check earns full credit within its bar, then follows a half-cosine ramp to zero. Position and angle errors reach zero at twice their bar (20 cm and 60 degrees for the limits above). Size uses the largest side-length ratio in either direction: full credit through 1.25, falling on the log of the ratio to zero at the reference bar of 2. The object's credit is the product of its applicable check credits.

Reported, not scored: count constraints (G1.2), Chamfer distance (G2.2) and navigation (G3.3, where a robot base can drive).

## 4. Functionality

Functionality simulates the world in MuJoCo with the benchmark's own settings. The verifiers bring their own forces, probes and robot.

- **Units and points.** Each verifier runs as units, such as one push per object. A level is its points earned over its points possible. A unit is worth one point, a robot task one point per checkpoint, and an object's n moving parts 1 + ln(n) points together.
- **Two settings.** The input-only setting leaves out units whose outcome depends on what a picture cannot show, such as weight. The reward uses it. The physics-hint setting counts every unit and is reported beside it.
- **Not at rest.** An object that ends 5 cm or 30 degrees from where it was put, under gravity alone, is tested no further: its other units fail.
- **Probes.** The ball, mug, load box and gripper are the verifiers' own. A world's default settings do not reach them.

### 4.1 Balance

| No. | Verifier | Passes when |
|---|---|---|
| F2.1 | At rest (one per object) | The object is free, fixed or jointed as in the reference room, has a valid mass, inertia and friction, and after 5 s under gravity has moved under 1 cm and is still |
| F2.6 | Edge drop | A probe mug set 3 cm inside a top's edge stays, and 3 cm past it falls |
| S0.3 | Fluid containment | At least 95% of the fluid is inside its holder and at rest before the hidden action starts |

### 4.2 Response

The world's number must be within the pass bar of the reference room's, either way (see [Pass bars](#pass-bars)).

| No. | Verifier | What is compared |
|---|---|---|
| F2.2 | Push | The force at which a slowly growing sideways push makes the object slide 2 cm or turn 5 degrees |
| F2.3 | Tilt | The slope at which it slips or tips: it must hold at a gentler slope and give way at a steeper one |
| F2.4 | Slide | How far it slides when flicked: it must stop short at a slower start and pass 10 cm at a faster one |
| S1.1 | Fluid spill | The share of the fluid that pours out when its container is tipped 60 to 90 degrees |
| S1.2 | Soft body press | How deep a 1 kg ball dropped from 15 to 35 cm presses in |
| S1.3 | Thin shell lift | How far a cloth rises when a strip of it is pulled up with a fixed force |

Push, tilt and slide overlap on purpose. Push sees weight times friction, tilt sees friction and tipping, and slide sees friction in motion.

### 4.3 Affordance

| No. | Verifier | Passes when |
|---|---|---|
| F3.1 | Door or drawer (one per object) | Each joint matches the true one, component by component (below). The unit is the share of the object's joints that pass |
| F3.2 | Container retention (one per declared opening) | A 2 cm ball sent in through the opening gets all the way in and stays: carried along when a movable container is lifted, walled in by a fixed one |
| F3.3 | Grasp and lift | Two fingers lift the object 20 cm at about the reference room's grip force: it slips at a weaker grip and lifts at a stronger one |
| F3.5 | Declared contents | What the reference room says a container holds rises with it when it is lifted 10 cm |
| F3.6 | Turn over | What a container or small holder holds falls out when it is turned upside down, as in the reference room |

Components of a joint (F3.1), all required:

| Component | Check |
|---|---|
| Presence, type | The joint and its moving part exist, hinge or slide as in the reference room |
| Place | The moving part is within 10 cm of the true one, or a quarter of its size |
| Axis, pivot | Axis within 10 degrees; a hinge's axis line within 5 cm of the true pivot |
| Travel | At least 90% of the true travel |
| Clearance | The part moves through its whole travel without hitting anything |
| Handle | If the true part has a handle, the world's has one a robot hand can pinch |
| Rest, actuation, full travel | It stays or closes as the true one, and opens under about the true force. Physics-hint setting only |

### 4.4 Interaction

Tasks are done in the world by a robot, the Rainbow Robotics RB-Y1 from MuJoCo Menagerie, driven by a scripted controller that reads object positions from the scene. Calibration proposes short tasks that work in the reference room, and a room may name its own. The instruction tells entrants the named tasks in plain words; the others stay hidden.

**How a task scores.** Each move is cut into stages, in order. A task earns one point per stage passed before the first that fails. A task that breaks a run rule, or whose object is missing, earns none.

| No. | Move | Stages, in order |
|---|---|---|
| F4.1 | Pick and place | Reach the object; grasp; lift; reach the goal; set it down upright; let go with nothing else moved (6) |
| F4.2 | Push | Reach the push path; touch; at the mark; upright; let go with nothing else moved (5) |
| F4.3 | Shelve a book | Reach it; push the spine out; pinch the spine; reach the bay; stand it upright, spine out; let go (6) |
| F4.4 | Open or close | Find the handle; reach it; half way; the whole way; let go (5) |
| F4.6 | Press | Reach the part; pressed down; back up with nothing else pressed (3) |
| F4.5 | Activity | Several of the moves above in one unbroken run. Its points are the sum of its steps' stages, at most 12 |
| F4.7 | Package task | A task the room brings as its own code, for outcomes no built-in move reaches. Scored by the stages its code reports |

"Nothing else moved" means no other object moved over 2 cm. Each move also has a time limit of 150 s (shelving 240 s).

### 4.5 Non-rigid objects

Rooms with fluids, soft bodies or thin shells describe them in a side file, `nonrigid.json` ([../ENTRANTS.md](../ENTRANTS.md), section 2). Genesis 1.4.3 runs each one on the CPU, with the nearby part of the world's rigid scene held still around it. Every world runs with the same solver settings, so its numbers decide, not its solver.

| Kind | Solver |
|---|---|
| Fluid | SPH, 6 mm particles |
| Soft body | MPM, 8 mm particles |
| Thin shell | PBD, 15 mm particles |

**Pairing.** Each reference object is compared with the submitted object of the same name, else the nearest with its category word and a size within 2 times, else the nearest of its kind with a size within 3 times.

**One unit per object**, in response, scored in three stages. A stage counts only when the ones before it pass:

1. **Kind.** The world's object is of the reference object's kind, with a shape and support the simulator can build.
2. **Material.** The same viscosity class for a fluid, the same fabric class (light or heavy) for a thin shell, and for a soft body a density within a factor of 2 and a stiffness within the room's own band.
3. **Behaviour.** The hidden action (S1.1 to S1.3, with S0.3 for a fluid): the world's readout is within a factor of 2 of the true one.

**A test counts only if it can decide.** Calibration reruns each reference object with one property halved and doubled (for a thin shell, with the other fabric class). If those wrong versions read the same as the right one, the test cannot tell them apart and is not counted.

**Hidden actions.** The ranges in section 4.2 are public. The value each object gets is drawn per room from a seed and stays hidden.

A submitted object more than 4 times the true size is not run and fails its unit. A run that goes over its time cap twice fails too.

## 5. Run rules

Rules, not verifiers: they have no score, and a run that breaks one fails its unit.

| No. | Rule |
|---|---|
| R1 | No sinking: nothing stays sunk more than 5 mm into anything for 0.3 s |
| R2 | No tunnelling: nothing passes through a wall or another object between steps |
| SX | Non-rigid runs: the run finishes in time, nothing blows up, a dropped weight does not pass through, a soft body stays on its support, a cloth stays on its rail and starts at rest |

## 6. Calibration checks

These decide which tests exist. They run once, on the reference room. Worlds never see them.

| Check | Rule |
|---|---|
| The reference room passes | A test the reference room fails is dropped. A reference room must score 100 on itself |
| Stable under jitter | Every loose object is moved by up to 1 mm, three times. A test whose outcome changes is dropped: it would pass or fail worlds by luck |
| Switches once | For tilt and slide, the outcome switches exactly once as the stimulus grows |
| Hint-only | Ten plausible wrong guesses of weight and friction rerun each test. If the outcome holds in fewer than nine, the test depends on what pictures cannot show and counts only in the physics-hint setting |
| Robot tasks survive nudges | Each robot task is rerun with the objects around it moved by up to 5 cm. A task that fails more than one of five runs is refused |

## 7. The reward

`reward` is the mean of appearance, geometry and functionality, each on 0 to 100.

- **Appearance:** the mean of the normalized SigLIP-2 and DINOv3 scores, using the floor and minimum-span formula in section 2.
- **Geometry:** the mean of inventory, shape, pose and size, relations and placement.
- **Functionality:** the mean of balance, response, affordance and interaction, in the input-only setting.

A part a room does not have is left out. A world that does not load scores 0 on every part. A world whose own files cannot be rendered scores 0 on appearance. If the grader itself fails, the run gets no reward and is graded again.

| Key | Meaning |
|---|---|
| `reward` | The overall score |
| `reward_appearance`, `reward_geometry`, `reward_functionality` | The three parts |
| `reward_balance`, `reward_response`, `reward_affordance`, `reward_interaction` | The four functionality levels |
| `parse`, `sim_ready` | 1 when the files load; 1 when the whole gate passes |
| `inventory`, `shape`, `pose_size`, `relations`, `placement` | The five geometry parts |
| `reward_visible` | The same reward over only the objects the input pictures show |
| `reward_functionality_hint` | Functionality with every unit counted, hint-only ones included |
| `reward_functionality_reference_bar` | Functionality at the reference pass bars |
| `missing_pct`, `extra_pct`, `position_error_median_cm`, `size_over_bar_pct`, `unsupported_pct`, `penetrating_pct` | Where a world loses geometry, in plain units |
| `verifier_version` | The version of this code that produced the score |

`report.json`, beside it, holds every unit's result and reason.

## Pass bars

Some verifiers compare one number in the world with the same number in the reference room. The world passes when the two are within a factor of each other, either way: a factor of 2 passes anything from half to double.

Each has two bars. The main bar sets every score. The reference bar is reported beside it, so anyone can see how much the choice matters.

| Kind of number | Verifiers | Main bar | Reference bar | Why |
|---|---|---|---|---|
| An amount guessed from pictures: weight, friction, damping | Push, tilt, slide, joint actuation, grasp | 2 | 1.25 | Work that scores physical amounts guessed from pictures counts a guess within a factor of 2 as right: image2mass (CoRL 2017), Q-Spatial Bench, SpatialVLM |
| A length measured in the pictures | Pose and size | 1.25 | 2 | Depth from pictures is scored by the share within a factor of 1.25 ("delta 1") |

One bar only: positions (10 cm), yes-or-no outcomes (at rest, edge drop, containers), robot tasks (a share of stages), appearance, and the non-rigid readouts (a factor of 2).

## Code

| Part | Where |
|---|---|
| Gate | [sbv/functionality/f1_gate.py](../../sbv/functionality/f1_gate.py) |
| Appearance | [sbv/appearance/](../../sbv/appearance/) |
| Geometry | [sbv/geometry/](../../sbv/geometry/), pairing in [sbv/common/case.py](../../sbv/common/case.py) |
| Balance, response | [sbv/functionality/f2_physical.py](../../sbv/functionality/f2_physical.py) |
| Affordance | [sbv/functionality/f3_functions.py](../../sbv/functionality/f3_functions.py) |
| Interaction | [sbv/functionality/f4_robot/](../../sbv/functionality/f4_robot/), [sbv/package_tasks/](../../sbv/package_tasks/) |
| Non-rigid | [sbv/nonrigid/](../../sbv/nonrigid/) |
| Run rules | [sbv/functionality/integrity.py](../../sbv/functionality/integrity.py) |
| Reward | [sbv/reward.py](../../sbv/reward.py) |
