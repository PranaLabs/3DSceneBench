# Where the judge comes from

`instruction.md` and `test.sh` are adapted from Terminal-Bench's trajectory judge
(harbor-framework/terminal-bench, commit e67a2cb, `leaderboard/src/leaderboard/ci/analysis-task-template/`),
Copyright the Terminal-Bench authors, licensed under the Apache License 2.0 (`LICENSE-terminal-bench`).

Changes in `instruction.md`: the inputs also allow our agents' `trajectory.jsonl`; rule 2 names the task's own
`## Rules` section; rule 3 ("Failed search ≠ hacking") is replaced by "Trying counts", because our tasks allow no
internet beyond the agent's own model calls; the reward-hacking criterion drops "General web reference is fine" and
lists our task rules. `test.sh` is unchanged apart from a comment naming its source.
