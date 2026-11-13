"""The task container is closed: setup and the grader have no network at all, and the agent step reaches only the model
APIs of the agents installed in the image (Claude Code, Codex). `--model-hosts` with no host writes tasks with no
network at all, for agents that call their model from the Harbor process."""

import tomllib, unittest
from scenebench_harbor.make_tasks import task_toml, MODEL_HOSTS


class ClosedEnvironment(unittest.TestCase):
    def test_only_the_agent_step_reaches_the_model_apis(self):
        self.assertEqual(MODEL_HOSTS, ["api.anthropic.com", "api.openai.com", "openrouter.ai"])
        for grader in (False, True):
            t = tomllib.loads(task_toml("o", "n", "r", 4, grader, 8 if grader else 1))
            self.assertEqual(t["environment"]["network_mode"], "no-network")
            self.assertEqual((t["agent"]["network_mode"], t["agent"]["allowed_hosts"]), ("allowlist", MODEL_HOSTS))
            self.assertEqual(t["verifier"]["network_mode"], "no-network")
            self.assertEqual(t["environment"]["build_timeout_sec"], 3600.0)  # Harbor's 10 minutes stop a first build

    def test_no_model_hosts_means_no_network_at_all(self):
        t = tomllib.loads(task_toml("o", "n", "r", 4, False, 1, model_hosts=[]))
        self.assertEqual(t["environment"]["network_mode"], "no-network")
        self.assertNotIn("network_mode", t["agent"])
        self.assertNotIn("allowed_hosts", t["agent"])


if __name__ == "__main__":
    unittest.main()
