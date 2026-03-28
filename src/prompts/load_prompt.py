from __future__ import annotations

from pathlib import Path


def load_system_prompt() -> str:
    path = Path(__file__).with_name("system_prompt.txt")
    return path.read_text(encoding="utf-8").strip() + "\n"

