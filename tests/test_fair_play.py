"""Fair play review (scenebench_harbor/fair_play.py) on made-up Harbor job folders: a clean run, runs that cheat in the
trajectory and in their settings, a judge verdict, and the fair_play.json that apply writes. Run from the repository
root:
    python -m unittest tests.test_fair_play -v
"""

import json, os, tempfile, unittest
from scenebench_harbor import fair_play as F


def trial(root, entrant, name, reward=50.0, config=None, atif=None, jsonl=None):
    d = os.path.join(root, entrant, "job", name)
    os.makedirs(os.path.join(d, "agent"))
    os.makedirs(os.path.join(d, "verifier"))
    with open(os.path.join(d, "verifier", "reward.json"), "w") as f:
        json.dump({"reward": reward}, f)
    with open(os.path.join(d, "result.json"), "w") as f:
        json.dump({"task_name": "pranalabs/" + name.split("__")[0]}, f)
    if config is not None:
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(config, f)
    if atif is not None:
        with open(os.path.join(d, "agent", "trajectory.json"), "w") as f:
            json.dump({"steps": atif}, f)
    if jsonl is not None:
        with open(os.path.join(d, "agent", "trajectory.jsonl"), "w") as f:
            f.write("".join(json.dumps(r) + "\n" for r in jsonl))
    return d


def read(p):
    with open(p) as f:
        return f.read()


def step(i, name, **args):
    return {
        "step_id": i,
        "source": "agent",
        "message": "",
        "tool_calls": [{"tool_call_id": str(i), "function_name": name, "arguments": args}],
    }


class FairPlay(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="sbv_fair_")
        self.root = os.path.join(self.tmp, "root")
        pub = os.path.join(self.tmp, "public")
        for t in ("scenebench-aa-p4-l0", "scenebench-bb-p4-l0"):
            os.makedirs(os.path.join(pub, t))
            open(os.path.join(pub, t, "instruction.md"), "w").close()
        claude = {"agent": {"name": "claude-code", "kwargs": {"disallowed_tools": "WebFetch,WebSearch"}}}
        trial(
            self.root,
            "clean",
            "scenebench-aa-p4-l0__a",
            config=claude,
            atif=[step(1, "Bash", command="python build.py"), step(2, "Read", file_path="/task/pictures/input_0.png")],
        )
        trial(
            self.root,
            "clean",
            "scenebench-bb-p4-l0__b",
            reward=0.0,
            config=claude,
            atif=[step(1, "Bash", command="ls")],
        )  # reward 0: not judged
        trial(
            self.root,
            "web",
            "scenebench-aa-p4-l0__c",
            config=claude,
            atif=[step(1, "WebSearch", query="ikea billy bookcase obj")],
        )
        trial(
            self.root,
            "web",
            "scenebench-bb-p4-l0__d",
            config={"agent": {"name": "codex", "kwargs": {"web_search": "disabled"}}},
            atif=[
                step(
                    3,
                    "exec_command",
                    cmd="python - <<'EOF'\nfrom openai import OpenAI\nOpenAI().images.generate(prompt='oak')\nEOF",
                )
            ],
        )
        trial(
            self.root,
            "tricks",
            "scenebench-aa-p4-l0__e",
            config={
                "agent": {
                    "name": "claude-code",
                    "kwargs": {"allowed_tools": "WebSearch"},
                    "override_timeout_sec": 99999,
                },
                "environment": {"mounts": [{"source": "/assets"}]},
            },
            atif=[step(1, "Bash", command="ls")],
        )
        trial(
            self.root,
            "ours",
            "scenebench-aa-p4-l0__f",
            jsonl=[
                {
                    "kind": "reply",
                    "turn": 4,
                    "tool_calls": [{"name": "bash", "arguments": {"command": "pip install trimesh"}}],
                },
                {
                    "kind": "reply",
                    "turn": 5,
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "bash", "arguments": '{"command": "echo wall socket__0"}'},
                        }
                    ],
                },
            ],
        )
        os.makedirs(os.path.join(self.root, "none", "job", "scenebench-aa-p4-l0__g"))  # no agent folder: not a trial
        self.pub = pub
        self.out = os.path.join(self.tmp, "review")
        self.found = F.check(self.root, pub, self.out, hub=False)

    def levels(self, key):
        return [lv for lv, _ in self.found[key]["findings"]]

    def test_hard_evidence_and_signs_from_both_trajectory_formats(self):
        self.assertEqual(self.levels("clean/job/scenebench-aa-p4-l0__a"), [])
        self.assertEqual(self.levels("web/job/scenebench-aa-p4-l0__c"), ["hard"])  # a web tool call
        self.assertEqual(self.levels("web/job/scenebench-bb-p4-l0__d"), ["hard"])  # an image API call from a script
        self.assertEqual(
            self.levels("ours/job/scenebench-aa-p4-l0__f"), ["review", "sign"]
        )  # no config; pip install is a sign, a wall socket is not
        self.assertNotIn("none/job/scenebench-aa-p4-l0__g", self.found)

    def test_settings_that_void_an_entry_or_leave_the_defaults(self):
        got = dict((t, lv) for lv, t in self.found["tricks/job/scenebench-aa-p4-l0__e"]["findings"])
        self.assertEqual(got["environment.mounts: mounts host folders"], "void")
        self.assertEqual(got["agent kwarg allowed_tools='WebSearch' turns web or image tools back on"], "void")
        self.assertEqual(got["agent.override_timeout_sec = 99999, not the default"], "default")
        self.assertIn("review", got.values())  # claude-code without the team's flag

    def test_judge_tasks_for_rewarded_trials_only(
        self,
    ):  # as Terminal-Bench: a trial that scored 0 cannot inflate the board
        index = F.load(os.path.join(self.out, "findings.json"))["judge_tasks"]
        self.assertNotIn("clean/job/scenebench-bb-p4-l0__b", index.values())
        self.assertIn("clean/job/scenebench-aa-p4-l0__a", index.values())
        task = os.path.join(
            self.out, "judge_tasks", next(k for k, v in index.items() if v == "clean/job/scenebench-aa-p4-l0__a")
        )
        self.assertTrue(os.path.exists(os.path.join(task, "environment", "trajectory.json")))
        self.assertTrue(os.path.exists(os.path.join(task, "environment", "task", "instruction.md")))
        toml = read(os.path.join(task, "task.toml"))
        self.assertIn('allowed_hosts = ["api.anthropic.com"]', toml)
        self.assertIn('[environment]\nnetwork_mode = "no-network"', toml)

    def test_report_and_apply_zero_trials_and_void_entries(self):
        index = F.load(os.path.join(self.out, "findings.json"))["judge_tasks"]
        folder = next(
            k for k, v in index.items() if v == "clean/job/scenebench-aa-p4-l0__a"
        )  # the judge flags the clean run
        jt = os.path.join(self.tmp, "judge", "job", folder + "__x")
        os.makedirs(os.path.join(jt, "artifacts", "app"))
        with open(os.path.join(jt, "config.json"), "w") as f:
            json.dump({"task": {"path": os.path.join(self.out, "judge_tasks", folder)}}, f)
        with open(os.path.join(jt, "result.json"), "w") as f:
            json.dump({}, f)
        v = {k: {"reasoning": "r", "summary": k, "verdict": k == "task_level_reward_hacking"} for k in F.KEYS}
        with open(os.path.join(jt, "artifacts", "app", "prediction.json"), "w") as f:
            f.write(json.dumps(v) + "\nextra text")
        F.report(self.out, [os.path.join(self.tmp, "judge")])
        md = read(os.path.join(self.out, "report.md"))
        self.assertIn("- [x] `clean/job/scenebench-aa-p4-l0__a` · judge, reward hacking: task_level_reward_hacking", md)
        self.assertIn("- [ ] `ours/job/scenebench-aa-p4-l0__f` · sign: turn 4: `pip install`", md)
        md = md.replace(
            "- [x] `clean/job/scenebench-aa-p4-l0__a`", "- [ ] `clean/job/scenebench-aa-p4-l0__a`"
        )  # a person unticks a false positive
        with open(os.path.join(self.out, "report.md"), "w") as f:
            f.write(md)
        F.apply(os.path.join(self.out, "report.md"), self.root)
        rec = lambda e: F.load(os.path.join(self.root, e, "fair_play.json"))
        self.assertEqual((rec("clean")["entry_invalid"], rec("clean")["disqualified_trials"]), (False, {}))
        self.assertEqual(
            sorted(rec("web")["disqualified_trials"]), ["job/scenebench-aa-p4-l0__c", "job/scenebench-bb-p4-l0__d"]
        )
        self.assertTrue(rec("tricks")["entry_invalid"])
        self.assertFalse(rec("ours")["entry_invalid"])

    def test_entries_must_be_run_with_harbor_run_upload(
        self,
    ):  # the Hub's record, not a folder we were sent
        def hub(
            entrant, created, trials, missing=False
        ):  # trials: (name, Hub created_at, start and end on the entrant's clock)
            job = (
                dict(id="1", missing=True)
                if missing
                else dict(
                    id="1",
                    job_name="job",
                    created_at=created,
                    trials=[dict(trial_name=n, created_at=c, started_at=a, finished_at=b) for n, c, a, b in trials],
                )
            )
            with open(os.path.join(self.root, entrant, F.HUB), "w") as f:
                json.dump(dict(jobs=[job]), f)

        # streamed from a machine on Pacific time: its clock says 01:01 to 02:01, no zone; each trial reached the Hub as
        # it ended
        hub(
            "clean",
            "2026-10-23T08:00:05+00:00",
            [
                ("scenebench-aa-p4-l0__a", "2026-10-23T09:01:30+00:00", "2026-10-23T01:01:00", "2026-10-23T02:01:00"),
                ("scenebench-bb-p4-l0__b", "2026-10-23T10:02:00+00:00", "2026-10-23T02:01:30", "2026-10-23T03:02:00"),
            ],
        )
        # the same kind of run uploaded the next day: every row within a minute of the job's
        hub(
            "web",
            "2026-10-24T09:00:00Z",
            [
                ("scenebench-aa-p4-l0__c", "2026-10-24T09:00:20Z", "2026-10-23T08:00:00", "2026-10-23T09:00:00"),
                ("scenebench-bb-p4-l0__d", "2026-10-24T09:00:40Z", "2026-10-23T09:00:00", "2026-10-23T10:00:00"),
            ],
        )
        hub("tricks", "2026-10-23T08:00:00+00:00", [])  # a trial the Hub lacks
        hub("ours", None, [], missing=True)
        found = F.check(self.root, self.pub, os.path.join(self.tmp, "review2"))
        entry = lambda e: [t for lv, t in found.get(f"{e}/(entry)", dict(findings=[]))["findings"] if lv == "void"]
        self.assertEqual(entry("clean"), [])
        self.assertIn("uploaded after the run, not with `harbor run --upload`", entry("web")[0])
        self.assertEqual(entry("tricks"), ["1 scored trial(s) not on the Hub: scenebench-aa-p4-l0__e"])
        self.assertIn("is not on Harbor Hub", entry("ours")[0])
        os.remove(os.path.join(self.root, "clean", F.HUB))
        found = F.check(self.root, self.pub, os.path.join(self.tmp, "review3"))
        self.assertIn("not fetched from Harbor Hub", entry("clean")[0])

    def test_more_than_one_trial_per_task_waits_for_a_person(self):  # the rule is one trial per task
        trial(self.root, "clean", "scenebench-aa-p4-l0__z", atif=[step(1, "Bash", command="ls")])
        F.check(self.root, self.pub, self.out, hub=False)
        F.report(self.out, [])
        self.assertIn(
            "- [ ] `clean/(entry)` · 2 trials for pranalabs/scenebench-aa-p4-l0: the rule is one trial per task",
            read(os.path.join(self.out, "report.md")),
        )


class RunSettings(unittest.TestCase):
    def test_hints_from_outside_void_the_entry_and_odd_options_are_listed(self):
        cfg = {
            "agent": {
                "name": "claude-code",
                "kwargs": {
                    "disallowed_tools": "WebFetch,WebSearch",
                    "memory_dir": "/x",
                    "append_system_prompt": "hint",
                    "temperature": 0,
                },
            }
        }
        got = F.settings_findings(cfg)
        self.assertEqual(
            sorted(t.split(":")[0] for lv, t in got if lv == "void"),
            ["agent kwarg append_system_prompt", "agent kwarg memory_dir"],
        )
        self.assertTrue(any(lv == "review" and "temperature" in t for lv, t in got))
        self.assertTrue(any("not Docker" in t for _, t in F.settings_findings({"environment": {"type": "daytona"}})))
        self.assertEqual(F.settings_findings({"agent": {"name": "codex", "kwargs": {"web_search": "disabled"}}}), [])

    def test_a_run_over_the_budget_is_listed(self):
        trial = tempfile.mkdtemp(prefix="sbv_fair_")
        with open(os.path.join(trial, "result.json"), "w") as f:
            json.dump(
                {
                    "agent_execution": {"started_at": "2026-11-01T10:00:00Z", "finished_at": "2026-11-01T13:14:00Z"},
                    "agent_result": {"n_output_tokens": 12_000_000},
                },
                f,
            )
        got = [t for _, t in F.run_findings(trial)]
        self.assertEqual(len(got), 2)
        self.assertIn("194 minutes", got[0])
        self.assertIn("12,000,000", got[1])
        with open(os.path.join(trial, "result.json"), "w") as f:
            json.dump(
                {
                    "agent_execution": {"started_at": "2026-11-01T10:00:00Z", "finished_at": "2026-11-01T12:59:00Z"},
                    "agent_result": {"n_output_tokens": 500_000},
                },
                f,
            )
        self.assertEqual(F.run_findings(trial), [])


if __name__ == "__main__":
    unittest.main()
