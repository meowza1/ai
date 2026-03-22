from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class Skill:
    name: str
    description: str
    handler: Callable[[str], str]


class SkillRegistry:
    def __init__(self):
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def run(self, name: str, text: str) -> str:
        if name not in self._skills:
            return f"Skill '{name}' not found."
        return self._skills[name].handler(text)

    def list_skills(self) -> list[str]:
        return sorted(self._skills.keys())


def default_registry() -> SkillRegistry:
    reg = SkillRegistry()
    reg.register(Skill("summarize", "Condense text to key points", lambda t: " ".join(t.split()[:80])))
    reg.register(Skill("reason_boost", "Adds explicit reasoning cue", lambda t: f"Let's solve this carefully: {t}"))
    reg.register(Skill("code_style", "Suggests cleaner code style", lambda t: f"Refactor suggestion:\n{t}"))
    return reg
