"""A Harbor agent for pipelines that are not agents: it copies a copy of the room you already built, from a folder on
your machine, into the task's /output/scene, so the verifiers grade it like any agent's answer.
    PYTHONPATH=/path/to/3DSceneBench harbor run -p tasks/public -a scenebench_harbor.submit_files:SubmitFiles \
        --ak folder=/path/to/my_copies
The folder holds one sub-folder per room id, each with scene.xml and the files it names (docs/ENTRANTS.md):
    my_copies/<room id>/scene.xml
PYTHONPATH must name this repository's root: Harbor does not look for agents in the current folder.
"""

from pathlib import Path
from pydantic import Field
from harbor.agents.base import BaseAgent
from harbor.agents.options import AgentOptions


class SubmitFilesOptions(AgentOptions):
    folder: str = Field(description="Folder with one sub-folder per room id, each holding scene.xml and its files")


class SubmitFiles(BaseAgent):
    options_model = SubmitFilesOptions

    @staticmethod
    def name():
        return "scenebench-submit-files"

    def version(self):
        return "1.0.0"

    async def setup(self, environment):
        pass

    async def run(self, instruction, environment, context):
        room = ((await environment.exec("cat /task/room.txt")).stdout or "").strip()
        src = Path(self.options.folder).expanduser().resolve() / room
        if not room or not (src / "scene.xml").is_file():
            raise FileNotFoundError(f"no scene.xml in {src}: the folder needs one sub-folder per room id ({room!r})")
        await environment.upload_dir(src, "/output/scene")
        self.logger.info("submitted %s for room %s", src, room)
