"""Harbor's OpenCode agent for the task container, where OpenCode is already installed (make_tasks.AGENTS).

Harbor's own `opencode` agent installs OpenCode from the network when a task starts, which the container does not have,
so the task would stop at setup. This one checks the installed program and changes nothing else. From the repository
root:

    harbor run -p tasks/public -a scenebench_harbor.agents:OpenCode -m openrouter/<model> -n 4
"""

from harbor.agents.installed.opencode import OpenCode as HarborOpenCode


class OpenCode(HarborOpenCode):
    async def install(self, environment):
        """No install: the program in the image, or a clear reason."""
        result = await environment.exec(command="opencode --version")
        if result.return_code != 0:
            raise RuntimeError(
                "OpenCode is not installed in this container; rebuild the task with scenebench_harbor/make_tasks.py"
            )
