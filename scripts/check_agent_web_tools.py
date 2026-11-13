#!/usr/bin/env python
"""Check, in Docker, that the agents installed in the task image (make_tasks.AGENTS) offer their model no web tools and
no image generation, and can still look at pictures and run commands. Builds the agent part of the image on
python:3.12-slim, starts a fake model API inside a container with no network, runs Claude Code, Codex and OpenCode
against it, and reads what each sent with its first model request. Each agent runs three ways: as Harbor runs it, asking
for the web tools back on the command line, and without our settings files (the control: it must show the web tools, or
the check could not see them). No key and no model call leave the machine. Rerun after changing AGENT_VERSIONS, NODE or
AGENT_FILES. From the repository root:
    python scripts/check_agent_web_tools.py [--image sbv-agent-tools:check]
"""

import argparse, json, os, subprocess, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the repository root
from scenebench_harbor import make_tasks as M  # noqa: E402

FAKE_API = r"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
class H(BaseHTTPRequestHandler):
    def _any(self):
        body = self.rfile.read(int(self.headers.get("content-length") or 0))
        with open("/tmp/requests.jsonl", "a") as f: f.write(json.dumps(dict(path=self.path, body=body.decode(errors="replace"))) + "\n")
        self.send_response(400); self.send_header("content-type", "application/json"); self.end_headers()
        self.wfile.write(b'{"type":"error","error":{"type":"invalid_request_error","message":"fake API"}}')
    do_GET = do_POST = _any
    def log_message(self, *a): pass
HTTPServer(("127.0.0.1", 8787), H).serve_forever()
"""
CLAUDE = "claude --verbose --output-format=stream-json --permission-mode=bypassPermissions {extra} --print 'Say hi'"
# Codex offers web search and image generation only to OpenAI-like providers, so the fake one says it is one
FAKE_PROVIDER = (
    '-c model_provider=fake -c \'model_providers.fake={name="fake",base_url="http://127.0.0.1:8787/v1",env_key="OPENAI_API_KEY",'
    'wire_api="responses",supports_standalone_web_search=true,supports_websockets=false,requires_openai_auth=true}\''
)
CODEX = (
    f"codex exec --skip-git-repo-check --dangerously-bypass-approvals-and-sandbox {FAKE_PROVIDER} {{extra}} 'Say hi'"
)
# OpenCode: its web tools come back through a variable, not a flag, so "asking" sets the variable
OPENCODE = (
    "env {extra} opencode --model=openrouter/fake/model run --format=json --dangerously-skip-permissions -- 'Say hi'"
)
OPENCODE_BACK = """OPENCODE_PERMISSION='{"webfetch":"allow","websearch":"allow"}' OPENCODE_ENABLE_EXA=1"""
OPENCODE_PROVIDER = '{"permission":{"webfetch":"allow"},"provider":{"openrouter":{"models":{"fake/model":{}},"options":{"baseURL":"http://127.0.0.1:8787/v1"}}}}'
SETTINGS = " ".join(where for where, _ in M.AGENT_FILES.values())
# name: (command with {extra} for flags, flags asking for the web tools back, words the model must not see, words it
# must see)
AGENTS = {
    "claude-code": (
        CLAUDE,
        "--allowedTools WebSearch,WebFetch",
        ["WebSearch", "WebFetch"],
        ['"name":"Read"', '"name":"Bash"'],
    ),
    "codex": (
        CODEX,
        "-c web_search=live -c features.image_generation=true",
        ["web.run", "imagegen", 'image_gen"'],
        ["view_image", '"exec"'],
    ),
    "opencode": (
        OPENCODE,
        OPENCODE_BACK,
        ['"name":"webfetch"', '"name":"websearch"', '"name":"codesearch"'],
        ['"name":"read"', '"name":"bash"'],
    ),
}
ENV = (
    "-e ANTHROPIC_BASE_URL=http://127.0.0.1:8787 -e ANTHROPIC_API_KEY=sk-ant-fake -e IS_SANDBOX=1 -e OPENAI_API_KEY=sk-fake "
    f"-e OPENROUTER_API_KEY=sk-fake -e OPENCODE_CONFIG_CONTENT={OPENCODE_PROVIDER}"
)  # the user config asks for webfetch too


def first_request(image, cmd, drop_settings=False):
    """The body of the agent's first model request (the one carrying its instructions and tools)."""
    pre = (
        f"rm -f {SETTINGS}; ln -s {M.OPENCODE_BIN} /usr/local/bin/opencode; " if drop_settings else ""
    )  # the program without its wrapper
    script = f"{pre}python /opt/fake_api.py & sleep 3; timeout 150 {cmd} >/tmp/out.txt 2>&1; cat /tmp/requests.jsonl 2>/dev/null"
    out = subprocess.run(
        ["docker", "run", "--rm", "--network", "none", *ENV.split(), image, "sh", "-c", script],
        capture_output=True,
        text=True,
    ).stdout
    bodies = [json.loads(x)["body"] for x in out.splitlines() if x.startswith("{")]
    return next((b for b in bodies if '"tools"' in b or "additional_tools" in b), None)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default="sbv-agent-tools:check")
    a = ap.parse_args(argv)
    ctx = tempfile.mkdtemp(prefix="sbv_agents_")
    M.write_agent_files(ctx)
    with open(os.path.join(ctx, "fake_api.py"), "w") as f:
        f.write(FAKE_API)
    with open(os.path.join(ctx, "Dockerfile"), "w") as f:
        f.write(
            "FROM --platform=linux/amd64 python:3.12-slim\nRUN apt-get update && apt-get install -y --no-install-recommends curl xz-utils "
            "&& rm -rf /var/lib/apt/lists/*\n" + M.AGENTS + "COPY fake_api.py /opt/fake_api.py\nWORKDIR /task\n"
        )
    subprocess.run(["docker", "build", "-q", "-t", a.image, ctx], check=True)
    for prog, pkg in (("claude", "@anthropic-ai/claude-code"), ("codex", "@openai/codex"), ("opencode", "opencode-ai")):
        got = subprocess.run(
            ["docker", "run", "--rm", "--network", "none", a.image, prog, "--version"], capture_output=True, text=True
        ).stdout.strip()
        print(f"{prog} --version: {got} (pinned {M.AGENT_VERSIONS[pkg]})")
    bad = []
    for agent, (cmd, back, banned, needed) in AGENTS.items():
        for way, extra, drop in (
            ("as Harbor runs it", "", False),
            ("asking for them back", back, False),
            ("control, no settings files", "", True),
        ):
            c = cmd.replace("{extra}", extra)
            body = first_request(a.image, c, drop)
            if body is None:
                bad.append(f"{agent} {way}")
                print(f"{agent} {way}: no model request")
                continue
            seen, missing = [w for w in banned if w in body], [w for w in needed if w not in body]
            print(
                f"{agent} {way}: web or image tools seen: {seen or 'none'}; picture or shell tools missing: {missing or 'none'}"
            )
            if missing or (bool(seen) != drop):
                bad.append(f"{agent} {way}")  # the control must see them, the others never
    print(
        "FAILED: " + ", ".join(bad) if bad else "OK: no web tools or image generation offered, pictures and shell kept"
    )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
