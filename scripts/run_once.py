import argparse
from pathlib import Path
import re
from typing import Union, Optional

import torch
from PIL import Image

from src.prompts.load_prompt import load_system_prompt
from src.vision.siglip_encoder import SiglipPatchEncoder
from src.vision.token_compression import compress_27x27_tokens, recommended_targets
from src.vlm.model import SimplePrefixVLM


def main() -> None:
    p = argparse.ArgumentParser()
    # Changed --image to --image-path to match your command
    p.add_argument("--image-path", dest="image", type=str, required=True) 
    p.add_argument("--compression", type=int, default=81)
    p.add_argument("--siglip", type=str, default="google/siglip-base-patch16-384")
    p.add_argument("--llm", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    # Added these so your command doesn't throw an "unrecognized arguments" error
    p.add_argument("--task", type=str) 
    p.add_argument("--llm-mode", type=str, default="fp16") 
    p.add_argument("--max_new_tokens", type=int, default=24)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    img = Image.open(args.image).convert("RGB")
    system_prompt = load_system_prompt()
    user_prompt = _task_prompt(args.task)

    enc = SiglipPatchEncoder.from_pretrained(args.siglip, device=device, dtype=dtype)
    patches = enc.encode(img)  # (1, N, D), N depends on checkpoint/grid
    n_tokens = int(patches.shape[1])
    valid_targets = recommended_targets(n_tokens)
    if args.compression not in valid_targets:
        raise ValueError(
            f"compression={args.compression} is not valid for {n_tokens} input tokens. "
            f"Try one of: {valid_targets}"
        )
    patches_c = compress_27x27_tokens(patches, target_tokens=args.compression)

    vlm = SimplePrefixVLM.from_pretrained(args.llm, device=device, dtype=dtype)
    out = vlm.generate(
        image_tokens=patches_c,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_new_tokens=args.max_new_tokens,
    )
    print(_normalize_for_task(out, task=args.task))


def _task_prompt(task: Optional[str]) -> str:
    if task == "crosswalk_signal":
        return (
            "Crosswalk walk signal is it red or green? "
            "Start your response with exactly one word: red|green|unknown. "
            "Then give one short action clause."
        )
    if task == "stairs":
        return (
            "Are there stairs or steps? "
            "Start your response with exactly one word: yes|no. "
            "Then give one short action clause."
        )
    if task == "obstacles":
        return (
            "Is there an obstacle ahead? "
            "Start your response with exactly one word: yes|no. "
            "Then give one short action clause."
        )
    return "Give short navigation advice."


def _normalize_for_task(text: str, *, task: Optional[str]) -> str:
    t = (text or "").strip()
    if task == "crosswalk_signal":
        low = t.lower()
        if re.search(r"\bred\b", low):
            return "red"
        if re.search(r"\bgreen\b", low):
            return "green"
        return "unknown"
    if task in ("stairs", "obstacles"):
        low = t.lower()
        if re.search(r"\bno\b|\bnone\b|\bclear\b", low):
            return "no"
        if re.search(r"\byes\b|\bpresent\b|\bobstacle\b|\bstairs\b|\bstep\b", low):
            return "yes"
        return "unknown"
    return t


if __name__ == "__main__":
    main()