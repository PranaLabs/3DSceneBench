# Notes for coding agents

Read this before you run the benchmark for someone, grade a job, or change anything here.

## If you are asked to run the benchmark

Follow "Getting started" in [README.md](README.md), in its order: Requirements, then one of the two ways, then "How to get your result". In short:

**Requirements**

1. Check that Docker is running and Python is 3.12 or newer. Say what is missing; do not install Docker yourself.
2. From the repository root: `uv tool install harbor==0.23.0` and `uv tool install huggingface_hub`, then `harbor auth login`. The login needs the person: ask them to do it.
3. `harbor download pranalabs-ai-team/3dscenebench-mini@v1.0.0` writes the folder `3dscenebench-mini/` with seven tasks, one for each scene of 3DSceneBench mini.
4. **The image models.** The person must accept the DINOv3 licence on its Hugging Face page and run `hf auth login` themselves; ask them, and wait. Then download the three models as the README's "Requirements" shows.
5. **The grader.** Build both images with exactly these names: `docker/build.sh 3dscenebench:v1.0.0`, then `docker/build_grader.sh 3dscenebench-grader:v1.0.0 3dscenebench:v1.0.0`. Every task grades with `3dscenebench-grader:v1.0.0`, so another name will not be found.
6. Check the memory Docker may use against the table in the README's "Requirements": three scenes need 48 GB to grade. Say which scenes the machine cannot grade before you run them.
7. Run the free self-check in the README's "Requirements", step 3: hand in scene 003's own reference room. It must score 100 on everything. If it does not, stop and report; do not spend model credit on a setup that does not grade correctly.

**Which way.** Ask if it is not clear. Way 1 tests a coding agent or a model: Harbor starts that agent inside each scene's container. Way 2 tests worlds the person's own pipeline has already built.

**Way 1**

1. The tested model's API key must be set the way its agent expects, for example `ANTHROPIC_API_KEY` for Claude Code. Ask for it; never print it.
2. Run one scene first, with `-i "3dscenebench-003-l0"`, using the command in the README's "Way 1".
3. Ask before running the other scenes. Each may use up to 3 hours or 10 million output tokens of the tested model.
4. Keep `-n 1`. The tests are timed, and two graders running at once can change a result.

**Way 2**

1. The worlds go in `my_copies/<scene number>/scene.xml`, one folder per scene.
2. Check each with `checker/check_scene.py`, as in the README's "Way 2", and report what it says.
3. Hand the folder in with the submit-files command in the README's "Way 2".

**Results**

- Each scene is graded when its run ends. From each trial's `verifier/reward.json`, report `reward`, `reward_appearance`, `reward_geometry` and `reward_functionality`, in one table. `python3 scripts/summarize.py jobs/<job>` prints it.
- Say which scenes could not be graded and why. A grader that runs out of memory reports `status: grader_failed`, which is not a score.
- If the person wants the run on the leaderboard, tell them to write to contact@pranalab.ai. Do not send that email yourself.

**A fair run.** You are the helper, not the agent being tested; that agent runs inside each scene's container. Do not:

- turn its web search or web fetch back on, or remove the `--ak` setting that disables them;
- edit anything under `3dscenebench-mini/` (the instruction, the images, `task.toml`, the Dockerfiles, `tests/`);
- read a task's `tests/` folder, or pass anything from it to the tested agent or into a world: it holds the reference room. The self-check above is the one exception;
- give the tested agent extra instructions, a system prompt, memory, mounted folders or files from outside;

The full rules are in [docs/ENTRANTS.md](docs/ENTRANTS.md), section 9.

## If you are asked to grade a saved job again

Harbor keeps each world a run saved, so a job can be graded again without running the tested agent again.

1. The grader must be built, as in Requirements above.
2. `harbor job regrade jobs/<job> -p 3dscenebench-mini -n 1 -o graded`. Keep `-n 1`.
3. Report the same table from each graded trial's `verifier/reward.json`.

Do not edit anything under `3dscenebench-mini/` or in the job's folder.

## If you are asked to change this code

The rest of this file is for that.

## Setup

- Python 3.11. Install with `python3.11 -m venv .venv` (or `uv venv -p 3.11 .venv`), then `.venv/bin/python -m pip install -r requirements/core.txt`.
- `requirements/render.txt`, `metrics.txt` and `nonrigid.txt` add Blender, the image metrics and Genesis. Self-tests that need them skip themselves when they are missing.

## Checks before you finish

```bash
ruff format .
```

```bash
.venv/bin/python -m unittest discover -s tests -t .
```

## Where things are

| Path | What it holds |
|---|---|
| `sbv/` | The verifiers: appearance, geometry, functionality, non-rigid, package tasks, the audit |
| `scenebench_harbor/` | The Harbor side: the task builder, the agents, the fair-play judge |
| `checker/` | The public checker that ships in every task |
| `docker/`, `requirements/` | The images and the Python packages |
| `scripts/` | Maintenance scripts |
| `tests/` | Self-tests |
| `docs/` | The entrant guide, what is tested, the verifier reference |

## Rules

- **Never commit a reference room** or anything made from one: scenes, cases, answer files, calibrated values, private tasks.
- A change that can move a score is a new verifier version. Say so in the pull request and do not mix it with other changes.
- Every `.py` file under `sbv/` is part of the verifier version, comments included. Do not touch those files for style alone.
- In code and in `docs/`, the programs that score a world are "verifiers", what they run are "tests", and this repository's own tests are "self-tests". The answer a world is compared with is the "reference room".
- No keys, tokens, machine names or home-folder paths in any file.
