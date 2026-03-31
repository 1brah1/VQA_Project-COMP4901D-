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
from src.vlm.llm_loader import (
    load_llm_fp16_or_fp32,
    infer_expected_hidden_size,
    validate_loaded_identity,
)


def _load_image_proj(vlm: SimplePrefixVLM, image_proj_path: Optional[str], device: str) -> Optional[str]:
    if not image_proj_path:
        return None
    proj_path = Path(image_proj_path)
    if proj_path.is_dir():
        proj_path = proj_path / "image_proj.pt"
    if not proj_path.exists():
        raise FileNotFoundError(f"image_proj not found: {proj_path}")
    state = torch.load(str(proj_path), map_location="cpu")
    state_dtype = None
    for val in state.values():
        if isinstance(val, torch.Tensor):
            state_dtype = val.dtype
            break
    if state_dtype is not None:
        vlm.image_proj = vlm.image_proj.to(device=device, dtype=state_dtype)
    vlm.image_proj.load_state_dict(state)
    vlm.image_proj.eval()
    return str(proj_path)


def main() -> None:
    p = argparse.ArgumentParser()
    # Changed --image to --image-path to match your command
    p.add_argument("--image-path", dest="image", type=str, required=True) 
    p.add_argument("--compression", type=int, default=81)
    p.add_argument("--siglip", type=str, default="google/siglip-base-patch16-384")
    p.add_argument("--llm", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--image-proj", type=str, default=None, help="Optional image_proj.pt file or directory")
    p.add_argument("--expected-hidden-size", type=int, default=None)
    # Added these so your command doesn't throw an "unrecognized arguments" error
    p.add_argument("--task", type=str) 
    p.add_argument("--llm-mode", type=str, default="fp16") 
    p.add_argument("--max_new_tokens", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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

    loaded = load_llm_fp16_or_fp32(args.llm, device=device, dtype=dtype)
    expected_hidden_size = args.expected_hidden_size
    if expected_hidden_size is None:
        expected_hidden_size = infer_expected_hidden_size(args.llm)
    validate_loaded_identity(loaded.identity, expected_hidden_size)
    llm_dtype = next(loaded.model.parameters()).dtype
    vlm = SimplePrefixVLM.from_loaded_llm(
        tokenizer=loaded.tokenizer,
        llm=loaded.model,
        device=device,
        dtype=llm_dtype,
        image_token_dim=768,
    )
    _load_image_proj(vlm, args.image_proj, device)
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
            "Task: crosswalk signal classification. "
            "Answer with exactly one label token and nothing else. "
            "Allowed labels: red, green. "
            "If uncertain, choose the safer label red."
        )
    if task == "stairs":
        return (
            "Task: stairs detection. "
            "Answer with exactly one label token and nothing else. "
            "Allowed labels: yes, no. "
            "If uncertain, choose the safer label yes."
        )
    if task == "obstacles":
        return (
            "Task: obstacle detection. "
            "Answer with exactly one label token and nothing else. "
            "Allowed labels: yes, no. "
            "If uncertain, choose the safer label yes."
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