"""
Shared task classification and spoken-sentence formatting helpers.

This module centralizes parsing logic so all runners produce consistent labels.
"""
from __future__ import annotations

import re
from typing import Dict, Optional, Tuple


TaskLabel = str
TaskName = str


_PROMPT_ECHO_PATTERNS = [
    r"start\s+your\s+response\s+with\s+exactly\s+one\s+word",
    r"is\s+the\s+crosswalk\s+walk\s+signal\s+red\s+or\s+green\??",
    r"are\s+there\s+stairs\s+or\s+steps\s+visible\??",
    r"is\s+there\s+an\s+obstacle\s+ahead\??",
    r"i\s+am\s+a\s+navigation\s+assistant\s+for\s+a\s+visually\s+impaired\s+user",
    r"^\s*assistant\s*:\s*",
    r"^\s*user\s*:\s*",
    r"red\|green\|unknown",
    r"yes\|no\|unknown",
]


def _normalize_for_parse(text: str) -> str:
    """Strip common prompt-echo fragments before regex parsing."""
    normalized = (text or "").strip().lower()
    for pat in _PROMPT_ECHO_PATTERNS:
        normalized = re.sub(pat, " ", normalized, flags=re.IGNORECASE | re.MULTILINE)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _first_token_label(text: str, labels: Tuple[str, ...]) -> Optional[str]:
    """Extract explicit first token label if present."""
    if not text:
        return None
    m = re.match(r"\W*([a-z]+)\b", text)
    if not m:
        return None
    token = m.group(1)
    return token if token in labels else None


def parse_label(task: TaskName, text: str) -> TaskLabel:
    """Normalize free-form model output to task label."""
    low = _normalize_for_parse(text)

    if task == "crosswalk_signal":
        first = _first_token_label(low, ("red", "green", "unknown"))
        if first is not None:
            return first
        has_red = bool(re.search(r"\bred\b", low))
        has_green = bool(re.search(r"\bgreen\b", low))
        if has_red and has_green:
            return "unknown"
        if has_red:
            return "red"
        if has_green:
            return "green"
        return "unknown"

    if task in ("stairs", "obstacles"):
        first = _first_token_label(low, ("yes", "no", "unknown"))
        if first is not None:
            return first
        has_no = bool(re.search(r"\bno\b|\bnone\b|\bclear\b", low))
        has_yes = bool(re.search(r"\byes\b|\bpresent\b|\bobstacle\b|\bstairs\b|\bstep\b", low))
        if has_yes and has_no:
            return "unknown"
        if has_no:
            return "no"
        if has_yes:
            return "yes"
        return "unknown"

    return "unknown"


def fallback_label(task: TaskName) -> TaskLabel:
    """Safety-biased fallback label when parsing fails."""
    if task == "crosswalk_signal":
        # Conservative default for pedestrian safety.
        return "red"
    if task in ("stairs", "obstacles"):
        # Conservative default for navigation safety.
        return "yes"
    return "unknown"


def spoken_sentence(task: TaskName, label: TaskLabel, confidence: Optional[float] = None) -> str:
    """
    Convert a task label into a short spoken sentence for TTS.

    `confidence` is currently optional and only controls uncertain phrasing when provided.
    """
    uncertain = confidence is not None and confidence < 0.5

    if task == "crosswalk_signal":
        if label == "red":
            return "I think the pedestrian signal is red. Please wait."
        if label == "green":
            return "I think the pedestrian signal is green. You can proceed carefully."
        return "I cannot confidently read the pedestrian signal right now."

    if task == "stairs":
        if label == "yes":
            return "There are stairs ahead. Move carefully."
        if label == "no":
            return "I do not see stairs ahead."
        return "I am not sure whether stairs are ahead."

    if task == "obstacles":
        if label == "yes":
            return "There is an obstacle ahead. Please be careful."
        if label == "no":
            return "The path ahead looks clear."
        return "I am not sure whether there is an obstacle ahead."

    if uncertain:
        return "I am not fully confident in this prediction."
    return "I cannot classify this scene."


def classify_and_format(
    task: TaskName,
    model_text: str,
    confidence: Optional[float] = None,
    avoid_unknown: bool = False,
) -> Tuple[TaskLabel, str]:
    """Return (classification_label, spoken_sentence) for a model output."""
    label = parse_label(task, model_text)
    if avoid_unknown and label == "unknown":
        label = fallback_label(task)
    return label, spoken_sentence(task, label, confidence=confidence)


def backend_state(requested: str, active: str, reason: Optional[str]) -> Dict[str, Optional[str]]:
    """Canonical backend-state dictionary used across reports."""
    return {
        "requested": requested,
        "active": active,
        "fallback_reason": reason,
    }
