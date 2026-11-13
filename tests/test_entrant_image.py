"""The entrants' container never holds what we build reference rooms from: no Infinigen, no SceneSmith assets or example
scenes, no Objaverse dumps, nothing from our room repositories. Nor the verifiers: the checker is the only code of ours
in a public task, in both its halves. Checks the Dockerfiles a public task is built from and every file copied next to
them. Run from the repository root:
    python -m unittest tests.test_entrant_image -v
"""

import ast, json, os, pathlib, re, sys, tempfile, tomllib, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from scenebench_harbor import make_tasks as M  # noqa: E402

DENY = ("infinigen", "scenesmith", "scenebench-gt-rooms", "scene-room-workshop", "objaverse")


def denied(text):
    return [w for w in DENY if re.search(w.replace("-", "[-_ ]?"), text, re.I)]


class EntrantImage(unittest.TestCase):
    def test_the_dockerfiles_install_nothing_on_the_deny_list(self):
        for name in ("ENV_DOCKERFILE", "CHECK_DOCKERFILE"):
            self.assertEqual(denied(getattr(M, name)), [], name)
            self.assertNotIn("sbv", getattr(M, name), name)

    def test_genesis_runs_with_the_graders_thread_count(
        self,
    ):  # fluid readouts change with the thread count (verify.THREADS)
        from sbv.nonrigid.verify import THREADS

        self.assertIn(f"QD_NUM_THREADS={THREADS}\n", M.TOOLS)

    def test_terminus_finds_its_terminal_installed(self):  # with no network it cannot install tmux or asciinema itself
        apt = M.TOOLS[M.TOOLS.index("apt-get install") : M.TOOLS.index("rm -rf /var/lib/apt/lists")]
        self.assertIn(" tmux ", apt)
        self.assertIn(" asciinema ", apt)

    def test_the_agents_are_installed_with_their_web_tools_off(self):
        for pkg, version in M.AGENT_VERSIONS.items():
            self.assertIn(f" {pkg}@{version}", M.TOOLS)
        self.assertIn(f'echo "{M.NODE[1]}  /tmp/node.tar.xz" | sha256sum -c -', M.TOOLS)  # the Node tarball is checked
        tmp = tempfile.mkdtemp(prefix="sbv_entrant_")
        pics = os.path.join(tmp, "pics")
        os.makedirs(pics)
        open(os.path.join(pics, "input_0.png"), "w").close()
        out = os.path.join(tmp, "task")
        M.write_task(out, "pranalabs", "scenebench-room-p1", "room", "1", pics, ["cup"])
        env = os.path.join(out, "environment")
        with open(os.path.join(env, "Dockerfile")) as f:
            docker = f.read()
        for name, where in (
            ("claude-managed-settings.json", "/etc/claude-code/managed-settings.json"),
            ("codex-requirements.toml", "/etc/codex/requirements.toml"),
            ("codex-config.toml", "/etc/codex/config.toml"),
            ("opencode.json", "/etc/opencode/opencode.json"),
            ("opencode", "/usr/local/bin/opencode"),
        ):
            self.assertIn(f"COPY agents/{name} {where}\n", docker)
        with open(os.path.join(env, "agents", "claude-managed-settings.json")) as f:
            claude = json.load(f)
        with open(os.path.join(env, "agents", "codex-requirements.toml"), "rb") as f:
            required = tomllib.load(f)
        with open(os.path.join(env, "agents", "codex-config.toml"), "rb") as f:
            config = tomllib.load(f)
        self.assertEqual(sorted(claude["permissions"]["deny"]), ["WebFetch", "WebSearch"])
        self.assertEqual(
            (required["allowed_web_search_modes"], required["features"]["image_generation"]), (["disabled"], False)
        )
        self.assertEqual(
            (config["web_search"], config["features"]["image_generation"], config["skills"]["bundled"]["enabled"]),
            ("disabled", False, False),
        )
        with open(os.path.join(M.ROOT, "docs", "ENTRANTS.md")) as f:
            guide = f.read()  # the guide names the versions installed
        with open(os.path.join(env, "agents", "opencode.json")) as f:
            opencode = json.load(f)
        self.assertEqual(opencode["permission"], dict(webfetch="deny", websearch="deny", codesearch="deny"))
        with open(os.path.join(env, "agents", "opencode")) as f:
            wrapper = f.read()  # the variable that would override the config never reaches the program
        self.assertIn("unset OPENCODE_PERMISSION", wrapper)
        self.assertIn(f"exec {M.OPENCODE_BIN}", wrapper)
        self.assertIn("rm /usr/local/bin/opencode\n", M.TOOLS)
        self.assertIn(
            "RUN chmod 755 /usr/local/bin/opencode\n", docker
        )  # npm's link first, or the copy would land on the program
        v = M.AGENT_VERSIONS
        self.assertIn(
            f"(Claude Code {v['@anthropic-ai/claude-code']}, Codex {v['@openai/codex']}, OpenCode {v['opencode-ai']})",
            guide,
        )

    def test_the_opencode_agent_uses_the_installed_program(self):
        import asyncio, types

        try:
            from scenebench_harbor.agents import OpenCode
        except ModuleNotFoundError:
            self.skipTest("Harbor is not installed here")
        calls = []

        class Env:
            def __init__(self, code):
                self.code = code

            async def exec(self, command, **k):
                calls.append(command)
                return types.SimpleNamespace(return_code=self.code, stdout="1.19.3")

        agent = OpenCode.__new__(OpenCode)
        asyncio.run(agent.install(Env(0)))
        self.assertEqual(calls, ["opencode --version"])  # nothing fetched
        with self.assertRaisesRegex(RuntimeError, "not installed"):
            asyncio.run(agent.install(Env(1)))

    def test_blenders_python_has_mujoco_at_the_main_pythons_versions(self):  # one script can use Blender and MuJoCo
        main = M.TOOLS[M.TOOLS.index("RUN pip install") : M.TOOLS.index("# Blender")]
        bpy = M.TOOLS[M.TOOLS.index("--python /opt/bpy/bin/python") : M.TOOLS.index("printf")]
        pins = lambda s: dict(re.findall(r"([\w-]+)==([\d.]+)", s))
        for pkg in ("mujoco", "numpy", "scipy", "trimesh", "pillow"):
            self.assertEqual(pins(bpy).get(pkg), pins(main)[pkg], pkg)

    def test_a_public_task_copies_nothing_on_the_deny_list(self):
        tmp = tempfile.mkdtemp(prefix="sbv_entrant_")
        pics = os.path.join(tmp, "pics")
        os.makedirs(pics)
        for f in ("input_0.png", "cameras.json"):
            open(os.path.join(pics, f), "w").close()
        out = os.path.join(tmp, "task")
        M.write_task(out, "pranalabs", "scenebench-room-p1", "room", "1", pics, ["cup"])
        with open(os.path.join(out, "environment", "Dockerfile")) as f:
            self.assertEqual(denied(f.read()), [])
        paths = [os.path.relpath(os.path.join(d, f), out) for d, _, fs in os.walk(out) for f in fs]
        self.assertEqual([p for p in paths if denied(p)], [])
        self.assertEqual(
            sorted(p for p in paths if p.endswith(".py")),  # no sbv anywhere: the checker and the time_left tool only
            ["environment/check_scene.py", "environment/time_left.py", "tests/check_scene.py"],
        )
        self.assertEqual([p for p in paths if "sbv" in p], [])
        for tool in ("check_scene.py", "time_left.py"):
            with open(os.path.join(out, "environment", tool)) as f:
                tree = ast.parse(f.read())
            names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names] + [
                n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
            ]
            self.assertEqual(
                [n for n in names if n.split(".")[0] in ("sbv", "scenebench_harbor", "tests")], [], tool
            )  # imports none of our code
        self.assertEqual(
            denied("RUN pip install infinigen && git clone https://github.com/x/sceneBench_gt_rooms"),
            ["infinigen", "scenebench-gt-rooms"],
        )

    def test_a_task_gives_what_is_tested_in_the_container(
        self,
    ):  # the container has no network: the instruction links nothing outside it
        from tests import rooms

        tmp = tempfile.mkdtemp(prefix="sbv_entrant_")
        pics = os.path.join(tmp, "pics")
        os.makedirs(pics)
        for f in ("input_0.png", "cameras.json"):
            open(os.path.join(pics, f), "w").close()
        room = os.path.dirname(rooms.write(os.path.join(tmp, "room")))
        envs = {}
        for side in ("public", "private"):
            out = os.path.join(tmp, side)
            envs[side] = os.path.join(out, "environment")
            M.write_task(
                out,
                "pranalabs",
                "scenebench-room-p1",
                "room",
                "1",
                pics,
                ["cup"],
                grader=(room, dict(schema="x"), "base", dict(cpus=2, memory_mb=None, minutes=90))
                if side == "private"
                else None,
            )
        env = envs["public"]
        self.assertEqual(os.listdir(os.path.join(env, "docs")), ["WHAT_IS_TESTED.md"])  # this one doc, no other
        with (
            open(os.path.join(env, "docs", "WHAT_IS_TESTED.md")) as f,
            open(os.path.join(M.ROOT, "docs", "WHAT_IS_TESTED.md")) as g,
        ):
            self.assertEqual(f.read(), g.read())
        with open(os.path.join(env, "Dockerfile")) as f:
            self.assertIn("\nCOPY docs /task/docs\n", f.read())
        with open(os.path.join(env, "instruction.md")) as f:
            text = f.read()
        self.assertIn("`/task/docs/WHAT_IS_TESTED.md`", text)
        self.assertNotIn("github.com/PranaLabs", text)
        self.assertNotIn("http", text)
        files = lambda d: {
            os.path.relpath(os.path.join(r, x), d): pathlib.Path(r, x).read_bytes()
            for r, _, fs in os.walk(d)
            for x in fs
        }
        self.assertEqual(
            files(envs["public"]), files(envs["private"])
        )  # the public and private containers are the same


if __name__ == "__main__":
    unittest.main()
