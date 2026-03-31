#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


_ROOT = Path(__file__).resolve().parent.parent

import sys

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.prompts.load_prompt import load_system_prompt
from src.vision.siglip_encoder import SiglipPatchEncoder
from src.vision.token_compression import compress_27x27_tokens
from src.vlm.llm_loader import load_llm_fp16_or_fp32
from src.vlm.model import SimplePrefixVLM


@dataclass
class TrainItem:
    sample_id: str
    task: str
    label: str
    image_tokens_cpu: torch.Tensor


_TASK_PROMPTS: Dict[str, str] = {
    "crosswalk_signal": (
        "Is the crosswalk walk signal red or green? "
        "Start your response with exactly one word: red|green|unknown. "
        "Then give one short action clause."
    ),
    "stairs": (
        "Are there stairs or steps visible? "
        "Start your response with exactly one word: yes|no|unknown. "
        "Then give one short action clause."
    ),
    "obstacles": (
        "Is there an obstacle ahead? "
        "Start your response with exactly one word: yes|no|unknown. "
        "Then give one short action clause."
    ),
}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _label_for_item(task: str, labels: Dict[str, str]) -> str:
    if task == "crosswalk_signal":
        return str(labels.get("walk_signal", "unknown"))
    if task == "stairs":
        return str(labels.get("stairs_present", "unknown"))
    if task == "obstacles":
        return str(labels.get("obstacle_present", "unknown"))
    return "unknown"


def _load_items(labels_path: Path) -> List[Tuple[str, str, str, Path]]:
    obj = json.loads(labels_path.read_text(encoding="utf-8"))
    out: List[Tuple[str, str, str, Path]] = []
    for row in obj.get("items", []):
        sample_id = str(row["id"])
        task = str(row["task"])
        label = _label_for_item(task, row.get("labels", {}))
        image_path = (_ROOT / row["path"]).resolve()
        if image_path.exists():
            out.append((sample_id, task, label, image_path))
    return out


def _build_prompt(tokenizer: any, system_prompt: str, user_prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "System: {0}\nUser: {1}\nAssistant:".format(system_prompt.strip(), user_prompt.strip())


def _prepare_dataset(
    encoder: SiglipPatchEncoder,
    labels_path: Path,
    compression: int,
    device: str,
    dtype: torch.dtype,
    max_items: int,
) -> List[TrainItem]:
    items = _load_items(labels_path)
    if max_items > 0 and len(items) > max_items:
        items = items[:max_items]
    prepared: List[TrainItem] = []
    print("[train] preparing {0} items".format(len(items)))
    with torch.no_grad():
        for idx, (sample_id, task, label, image_path) in enumerate(items, 1):
            image = Image.open(image_path).convert("RGB")
            tok = encoder.encode(image)
            tok = compress_27x27_tokens(tok, target_tokens=compression)
            tok = tok.detach().to(device="cpu", dtype=torch.float32)
            prepared.append(TrainItem(sample_id=sample_id, task=task, label=label, image_tokens_cpu=tok))
            if idx % 8 == 0 or idx == len(items):
                print("[train] prepared {0}/{1}".format(idx, len(items)))
    return prepared


def _save_checkpoint(
    ckpt_dir: Path,
    state: Dict[str, object],
    step: int,
) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / ("step_{0:06d}.pt".format(step))
    torch.save(state, str(path))
    latest = ckpt_dir / "latest.pt"
    torch.save(state, str(latest))
    return path


def _load_checkpoint(path: Path) -> Dict[str, object]:
    return torch.load(str(path), map_location="cpu")


def _latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    latest = ckpt_dir / "latest.pt"
    if latest.exists():
        return latest
    pts = sorted(ckpt_dir.glob("step_*.pt"))
    if not pts:
        return None
    return pts[-1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Jetson-safe trainer for SimplePrefixVLM (image projection or full fine-tune)"
    )
    parser.add_argument("--labels", type=str, default="data/train/labels.json")
    parser.add_argument("--siglip", type=str, default="google/siglip-base-patch16-384")
    parser.add_argument("--llm", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--compression", type=int, default=192)
    parser.add_argument("--train-mode", type=str, default="image_proj", choices=["image_proj", "full"])
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--llm-dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--proj-dtype", type=str, default="fp32", choices=["fp16", "fp32"])
    parser.add_argument("--amp", action="store_true", help="Enable AMP mixed precision on CUDA")
    parser.add_argument("--grad-checkpointing", action="store_true", help="Enable LLM gradient checkpointing")
    parser.add_argument(
        "--allow-fp32-fallback",
        action="store_true",
        help="Allow FP32 fallback if FP16 logits are non-finite",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--empty-cache-steps", type=int, default=20)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/image_proj")
    parser.add_argument("--save-llm-checkpoints", action="store_true")
    parser.add_argument("--resume", type=str, default="latest", help="latest|none|/path/to/ckpt.pt")
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    requested_llm_dtype = torch.float16 if args.llm_dtype == "fp16" else torch.float32
    proj_dtype = torch.float16 if args.proj_dtype == "fp16" else torch.float32
    use_amp = False

    labels_path = (_ROOT / args.labels).resolve() if not Path(args.labels).is_absolute() else Path(args.labels)
    ckpt_dir = (_ROOT / args.checkpoint_dir).resolve() if not Path(args.checkpoint_dir).is_absolute() else Path(args.checkpoint_dir)
    output_dir: Optional[Path] = None
    if args.output_dir:
        output_dir = (_ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    if args.train_mode == "full" and output_dir is None:
        raise ValueError("--output-dir is required when --train-mode full")

    encoder = SiglipPatchEncoder.from_pretrained(args.siglip, device=device, dtype=requested_llm_dtype)
    loaded = load_llm_fp16_or_fp32(args.llm, device=device, dtype=requested_llm_dtype)
    llm_dtype = next(loaded.model.parameters()).dtype
    use_amp = bool(args.amp and device == "cuda" and llm_dtype == torch.float16)
    effective_dtype = str(loaded.identity.get("effective_dtype", str(llm_dtype)))
    if args.train_mode == "full" and effective_dtype == "torch.float32" and not args.allow_fp32_fallback:
        raise RuntimeError(
            "LLM loaded in FP32 due to non-finite FP16 logits. "
            "Re-run with --allow-fp32-fallback to proceed, or use a smaller model."
        )
    vlm = SimplePrefixVLM.from_loaded_llm(
        tokenizer=loaded.tokenizer,
        llm=loaded.model,
        device=device,
        dtype=llm_dtype,
        image_token_dim=768,
    )
    print(
        "[train] device={0} llm_dtype={1} proj_dtype={2} amp={3} mode={4}".format(
            device,
            llm_dtype,
            proj_dtype,
            use_amp,
            args.train_mode,
        )
    )
    if args.train_mode == "image_proj":
        for p in vlm.llm.parameters():
            p.requires_grad = False
        vlm.llm.eval()
    else:
        for p in vlm.llm.parameters():
            p.requires_grad = True
        vlm.llm.train()
        if args.grad_checkpointing and hasattr(vlm.llm, "gradient_checkpointing_enable"):
            # Reduce activation memory for Jetson-scale full fine-tunes.
            vlm.llm.gradient_checkpointing_enable()
            if hasattr(vlm.llm, "config"):
                vlm.llm.config.use_cache = False

    vlm.image_proj = vlm.image_proj.to(device=device, dtype=proj_dtype)
    vlm.image_proj.train()

    dataset = _prepare_dataset(
        encoder=encoder,
        labels_path=labels_path,
        compression=args.compression,
        device=device,
        dtype=llm_dtype,
        max_items=args.max_items,
    )
    if not dataset:
        print("[train] no dataset items available")
        return 1

    if args.train_mode == "image_proj":
        train_params = list(vlm.image_proj.parameters())
    else:
        train_params = list(vlm.llm.parameters()) + list(vlm.image_proj.parameters())
    optimizer = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    system_prompt = load_system_prompt()
    tokenizer = vlm.tokenizer
    llm = vlm.llm
    embed = llm.get_input_embeddings()

    global_step = 0
    start_epoch = 0
    start_idx = 0

    if args.resume != "none":
        ckpt_path: Optional[Path]
        if args.resume == "latest":
            ckpt_path = _latest_checkpoint(ckpt_dir)
        else:
            ckpt_path = Path(args.resume)
        if ckpt_path and ckpt_path.exists():
            state = _load_checkpoint(ckpt_path)
            vlm.image_proj.load_state_dict(state["image_proj_state"])  # type: ignore[index]
            optimizer.load_state_dict(state["optimizer_state"])  # type: ignore[index]
            if args.train_mode == "full" and "llm_state" in state:
                vlm.llm.load_state_dict(state["llm_state"])  # type: ignore[arg-type]
            if use_amp and "scaler_state" in state:
                scaler.load_state_dict(state["scaler_state"])  # type: ignore[arg-type]
            global_step = int(state.get("global_step", 0))
            start_epoch = int(state.get("epoch", 0))
            start_idx = int(state.get("sample_idx", 0))
            print("[train] resumed from {0} (step={1}, epoch={2}, idx={3})".format(ckpt_path, global_step, start_epoch, start_idx))

    running_loss = 0.0
    running_count = 0
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs):
        order = list(range(len(dataset)))
        random.shuffle(order)

        begin_i = start_idx if epoch == start_epoch else 0
        for i in range(begin_i, len(order)):
            item = dataset[order[i]]
            user_prompt = _TASK_PROMPTS.get(item.task, _TASK_PROMPTS["obstacles"])
            prompt = _build_prompt(tokenizer, system_prompt, user_prompt)

            prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids[0]
            target_ids = tokenizer(item.label, add_special_tokens=False, return_tensors="pt").input_ids[0]
            if target_ids.numel() == 0:
                continue

            prompt_ids = prompt_ids.to(device)
            target_ids = target_ids.to(device)
            text_ids = torch.cat([prompt_ids, target_ids], dim=0).unsqueeze(0)

            text_embeds = embed(text_ids)
            img_tokens = item.image_tokens_cpu.to(device=device, dtype=proj_dtype)
            img_prefix = vlm.image_proj(img_tokens)
            img_prefix = img_prefix.to(dtype=text_embeds.dtype)

            inputs_embeds = torch.cat([img_prefix, text_embeds], dim=1)
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)

            labels = torch.full((1, inputs_embeds.shape[1]), -100, dtype=torch.long, device=device)
            image_len = img_prefix.shape[1]
            prompt_len = prompt_ids.shape[0]
            labels[0, image_len + prompt_len : image_len + prompt_len + target_ids.shape[0]] = target_ids

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=llm_dtype):
                out = llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
                loss = out.loss / float(args.grad_accum)
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += float(out.loss.detach().cpu().item())
            running_count += 1

            do_step = ((i + 1) % args.grad_accum) == 0
            if do_step:
                if args.clip_grad > 0:
                    if use_amp:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(train_params, args.clip_grad)
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if args.empty_cache_steps > 0 and device == "cuda" and (global_step % args.empty_cache_steps == 0):
                    torch.cuda.empty_cache()

                if args.save_every > 0 and (global_step % args.save_every == 0):
                    state: Dict[str, object] = {
                        "image_proj_state": vlm.image_proj.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "global_step": global_step,
                        "epoch": epoch,
                        "sample_idx": i + 1,
                        "args": vars(args),
                        "train_mode": args.train_mode,
                    }
                    if use_amp:
                        state["scaler_state"] = scaler.state_dict()
                    if args.train_mode == "full" and args.save_llm_checkpoints:
                        state["llm_state"] = vlm.llm.state_dict()
                    ckpt = _save_checkpoint(ckpt_dir, state, global_step)
                    avg = running_loss / max(1, running_count)
                    print("[train] step={0} avg_loss={1:.4f} ckpt={2}".format(global_step, avg, ckpt))
                    running_loss = 0.0
                    running_count = 0

                if args.max_steps > 0 and global_step >= args.max_steps:
                    final_path = ckpt_dir / "image_proj_final.pt"
                    torch.save(vlm.image_proj.state_dict(), str(final_path))
                    print("[train] reached max_steps={0}; saved {1}".format(args.max_steps, final_path))
                    if args.train_mode == "full" and output_dir is not None:
                        output_dir.mkdir(parents=True, exist_ok=True)
                        vlm.llm.save_pretrained(str(output_dir))
                        vlm.tokenizer.save_pretrained(str(output_dir))
                        torch.save(vlm.image_proj.state_dict(), str(output_dir / "image_proj.pt"))
                    return 0

        start_idx = 0
        elapsed = time.time() - t0
        avg = running_loss / max(1, running_count)
        print("[train] epoch={0} done, elapsed={1:.1f}s, avg_loss={2:.4f}".format(epoch + 1, elapsed, avg))

    final_path = ckpt_dir / "image_proj_final.pt"
    torch.save(vlm.image_proj.state_dict(), str(final_path))
    final_state: Dict[str, object] = {
        "image_proj_state": vlm.image_proj.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "global_step": global_step,
        "epoch": args.epochs,
        "sample_idx": 0,
        "args": vars(args),
        "train_mode": args.train_mode,
    }
    if use_amp:
        final_state["scaler_state"] = scaler.state_dict()
    if args.train_mode == "full" and args.save_llm_checkpoints:
        final_state["llm_state"] = vlm.llm.state_dict()
    _save_checkpoint(ckpt_dir, final_state, global_step)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.train_mode == "full":
            vlm.llm.save_pretrained(str(output_dir))
            vlm.tokenizer.save_pretrained(str(output_dir))
        torch.save(vlm.image_proj.state_dict(), str(output_dir / "image_proj.pt"))
        meta = {
            "train_mode": args.train_mode,
            "global_step": global_step,
            "epochs": args.epochs,
            "llm": args.llm,
            "siglip": args.siglip,
            "compression": args.compression,
            "llm_dtype": str(llm_dtype),
            "proj_dtype": str(proj_dtype),
        }
        (output_dir / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("[train] complete; saved {0}".format(final_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
