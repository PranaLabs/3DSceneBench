# What is tested

This page says what the verifiers (the programs that score your world) look at in your functional and reusable world. It lists the parts of the score and, for each part, the properties of the room it depends on. Your task instruction says how to build, check and return your world.

## How your objects are matched

Most checks are about single objects. So the verifiers first pair each reference object with one object of your world, using:

- its **category word**: the part of the name before `__`, from the category list (`/task/categories.txt`);
- its **place**: your object has to stand near where the true one stands;
- its **size**: the two have to be of a similar size.

A reference object that finds no partner counts as missing, and every check on it fails. An extra object of yours that pairs with nothing lowers the inventory score. It also counts against placement when it floats, sinks or leaves the room, and it can get in the way of other checks (it blocks a drawer, or a robot hits it).

Some objects matter more than others. Objects that a robot task moves or puts things on, and objects with several moving parts, weigh more in the scores that are averaged over objects.

## The parts of the score

Each part is scored from 0 to 100. `reward.json` holds every part on its own.

| Part | Question |
|---|---|
| Gate | Does your world load and run? |
| Appearance | Does it look like the room from viewpoints you were not given? |
| Geometry | Are the right objects there, with the right shape and size, in the right places? |
| Functionality: balance | Does everything stay put as built? |
| Functionality: response | Does each object react like the real one when pushed or tipped? |
| Functionality: affordance | Do doors and drawers work, do containers hold, can objects be picked up? |
| Functionality: interaction | Can the room's tasks be done in your world? |

Rooms with fluids, soft bodies or cloth add checks on those things to geometry, balance and response.

## Gate

- Every file the scene names is in your folder.
- Metres, z up, and values MuJoCo accepts.
- The scene loads and runs without blowing up.

A file that does not load scores 0 on every part. If the scene runs but blows up, the verifiers switch off the objects that cause it and still score the rest of your world. The switched-off objects fail their checks.

## Appearance

The verifiers render your world from hidden cameras placed around the room, away from the cameras of your pictures. They compare these renders with renders of the reference room from the same cameras. The lighting is the same for every world.

What it depends on:
- what is in the room and where, as seen from new viewpoints;
- the outlines of the objects and how their parts are laid out;
- colours, materials and image textures, as your MuJoCo file defines them.

## Geometry

Geometry looks at your world as you hand it in, before anything moves.

- **Inventory:** the right objects, in the right numbers, by category.
- **Shape:** the surface of each matched object, compared with the true one's.
- **Pose and size:** each object's position, the direction of its long side, and its size along each side.
- **Relations:** what is on, inside or next to what, and which way things face.
- **Placement:** each object is held up by what is under it, does not sink into other objects, and stays inside the room.
- **Fluids and soft bodies:** the amount of each fluid and the size of each soft body.

The verifiers also measure and report where a robot base can drive through the room. This is not part of the score.

## Functionality

Functionality simulates your world in MuJoCo, with the same settings for every world. The verifiers bring their own forces, probes and robot. Your scene supplies only the world. Each level's score is the share of its checks your world passes. A task with several steps earns credit for each step it completes.

An object that does not stay where you put it (it slides, falls or tips over on its own) is not checked any further.

### Balance

- Each object rests as built: free objects settle where they are, and fixed objects stay fixed.
- Each top's edge is where it looks: what sits near the edge stays, and what is past it falls.
- A fluid stays in its container.

What it depends on: supports, collision shapes that match the visible ones, and which objects are fixed and which are free.

### Response

- How hard an object is to push before it moves or tips.
- At what slope it slips or tips.
- How far it slides when flicked.
- How much fluid a container pours when tipped, how much a soft body gives under a weight, and how far a cloth rises when pulled up.

What it depends on: mass, friction, the centre of mass, the footprint and the collision shape, how runny a fluid is, how stiff a soft body is, and what a cloth is made of. These are judged as amounts a builder could estimate from pictures. They need to be plausible for the object, not exact.

### Affordance

- **Doors, drawers, lids, knobs and keys:** the kind of joint (hinge or slide), its axis and pivot, how far it travels, that it can move without hitting anything, and that it has a handle where the real one has one.
- **Containers:** a hollow thing keeps what goes into it. Its inside must be open to enter and closed below.
- **Contents:** what a container holds in the reference room stays with it when it is carried.
- **Turning over:** when a container or a small holder is turned upside down, what it holds falls out, as in the reference room. For example, a ball put into a cup, or a cup standing on a coaster.
- **Grasp:** an object can be gripped and lifted like the real one.

What it depends on: moving parts as separate child bodies with real joints, hollow collision shapes for hollow things, friction and mass.

### Interaction

Tasks are done in your world in two ways:

- **By a robot.** The Rainbow Robotics RB-Y1 from MuJoCo Menagerie, the same for every world, picks things up and places them, pushes them to a mark, stands a book in a shelf, opens and closes doors and drawers, presses keys and buttons, and does several moves in a row.
- **Without the robot.** The verifiers act on your objects directly, with their own grasp or their own forces, for tasks a robot arm cannot do well.

A room may have either kind of task or both, and the number of tasks differs from room to room. Your instruction lists the room's own tasks in words, under "Tasks done in your world". Each task is scored by how many of its steps are completed in order.

What it depends on: everything above together, at the reference room's sizes and places. That means the heights of surfaces, room to reach and move, handles, and objects that can be gripped.

## Fluids, soft bodies and thin shells

Rooms that have them describe them in `nonrigid.json` (see "Non-rigid objects", under "Output files" in your task instruction). A second simulator, Genesis, runs them with the rigid room around them. The verifiers check each fluid's amount and each soft body's size, check that they stay in their holder or on their support, and check how they respond to a hidden action: a container is tipped, a weight is pressed onto a soft body, or a cloth is pulled up.

## Rules every simulation follows

Every run in your world must also stay physically sound. Nothing may stay sunk into anything, nothing may pass through a wall or another object, and non-rigid runs must finish in time without blowing up. A run that breaks one of these rules fails its check.

## What stays hidden

The reference room, the hidden cameras, which objects each check picks, the forces, angles, amounts and positions used, the physics seeds, and the exact limits of every check.
