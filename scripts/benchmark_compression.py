from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import psutil
import torch
from PIL import Image
from tqdm import tqdm

from src.prompts.load_prompt import load_system_prompt
from src.vision.siglip_encoder import SiglipPatchEncoder
from src.vision.token_compression import compress_27x27_tokens, recommended_targets
from src.vlm.model import SimplePrefixVLM
from src.vlm.llm_loader import (
    load_llm_awq,
    load_llm_fp16_or_fp32,
    infer_expected_hidden_size,
    validate_loaded_identity,
)


@dataclass
class EvalItem:
    id: str
    path: str
    task: str
    labels: Dict[str, str]


def load_labels(path: Path) -> List[EvalItem]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    items = []
    for it in obj.get("items", []):
        items.append(EvalItem(id=it["id"], path=it["path"], task=it["task"], labels=dict(it["labels"])))
    return items


def parse_crosswalk_signal(text: str) -> str:
    t = text.lower()
    if re.search(r"\bred\b", t):
        return "red"
    if re.search(r"\bgreen\b", t):
        return "green"
    return "unknown"


def parse_yes_no(text: str) -> str:
    t = text.lower()
    if re.search(r"\bno\b|\bnone\b|\bclear\b", t):
        return "no"
    if re.search(r"\byes\b|\bpresent\b|\bobstacle\b|\bstairs\b|\bstep\b", t):
        return "yes"
    return "unknown"


def memory_mb() -> float:
    rss = psutil.Process().memory_info().rss / (1024 * 1024)
    if torch.cuda.is_available():
        return float(max(rss, torch.cuda.max_memory_allocated() / (1024 * 1024)))
    return float(rss)


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
    p.add_argument("--labels", type=str, default="data/eval/labels.json")
    p.add_argument("--siglip", type=str, default="google/siglip-base-patch16-384")
    p.add_argument("--llm", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--image-proj", type=str, default=None, help="Optional image_proj.pt file or directory")
    p.add_argument("--expected-hidden-size", type=int, default=None,
                   help="Optional strict check for loaded LLM hidden size.")
    p.add_argument("--llm_mode", type=str, default="fp16", choices=["fp16", "awq"])
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducible runs.")
    p.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature when --do_sample is set.")
    p.add_argument("--top_k", type=int, default=50, help="Top-k sampling cutoff when --do_sample is set.")
    p.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling cutoff when --do_sample is set.")
    p.add_argument("--do_sample", action="store_true", help="Enable stochastic decoding.")
    p.add_argument("--compression", type=int, nargs="+", default=None)
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="Default generation cap (used if --max_new_tokens_eval is not set).",
    )
    p.add_argument(
        "--max_new_tokens_eval",
        type=int,
        default=None,
        help="If set, use this cap for the benchmark run (recommended 8-16 for label tasks on Jetson).",
    )
    p.add_argument(
        "--prompt_style",
        type=str,
        default="label_plus_action",
        choices=["label_plus_action", "label_only"],
        help="Prompt format: label_only often improves classification accuracy by forcing a one-token answer.",
    )
    p.add_argument("--out", type=str, default="reports/benchmark.json")
    args = p.parse_args()
    eval_cap = args.max_new_tokens_eval if args.max_new_tokens_eval is not None else args.max_new_tokens

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    items = load_labels(Path(args.labels))
    system_prompt = load_system_prompt()

    enc = SiglipPatchEncoder.from_pretrained(args.siglip, device=device, dtype=dtype)

    if args.llm_mode == "fp16":
        loaded = load_llm_fp16_or_fp32(args.llm, device=device, dtype=dtype)
        llm_dtype = next(loaded.model.parameters()).dtype
        vlm = SimplePrefixVLM.from_loaded_llm(
            tokenizer=loaded.tokenizer,
            llm=loaded.model,
            device=device,
            dtype=llm_dtype,
            image_token_dim=768,
        )
    else:
        loaded = load_llm_awq(args.llm, device=device)
        llm_dtype = next(loaded.model.parameters()).dtype
        vlm = SimplePrefixVLM.from_loaded_llm(
            tokenizer=loaded.tokenizer,
            llm=loaded.model,
            device=device,
            dtype=llm_dtype,
            image_token_dim=768,
        )

    expected_hidden_size = args.expected_hidden_size
    if expected_hidden_size is None:
        expected_hidden_size = infer_expected_hidden_size(args.llm)
    validate_loaded_identity(loaded.identity, expected_hidden_size)
    image_proj_path = _load_image_proj(vlm, args.image_proj, device)

    results: Dict[str, Any] = {
        "device": device,
        "dtype": str(dtype),
        "siglip": args.siglip,
        "llm": args.llm,
        "image_proj": image_proj_path,
        "llm_mode": args.llm_mode,
        "seed": args.seed,
        "generation_config": {
            "max_new_tokens": eval_cap,
            "do_sample": bool(args.do_sample),
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
        },
        "model_identity": loaded.identity,
        "expected_hidden_size": expected_hidden_size,
        "benchmark_config": {
            "max_new_tokens_used": eval_cap,
            "max_new_tokens_default": args.max_new_tokens,
            "max_new_tokens_eval": args.max_new_tokens_eval,
            "prompt_style": args.prompt_style,
            "image_proj": image_proj_path,
            "metrics": {
                "accuracy_scored": "only items where gt AND pred are both not 'unknown'; fraction that match",
                "accuracy_gt_known": "all items with gt != 'unknown'; pred must match gt (unknown pred = wrong)",
                "unknown_rate": "fraction of gt-known items where prediction is 'unknown'",
                "by_task": "same metrics split by labels task (crosswalk_signal, stairs, obstacles)",
                "n_gen_tokens": "count of token ids passed to decode after generate()",
            },
        },
        "compression": {},
    }

    if not items:
        raise ValueError("No items found in labels file.")
        
    sample_img = Image.open(items[0].path).convert("RGB")
    sample_tokens = int(enc.encode(sample_img).shape[1])
    default_targets = recommended_targets(sample_tokens)
    comp_list = args.compression if args.compression else default_targets

    for comp in comp_list:
        if comp not in recommended_targets(sample_tokens):
            raise ValueError(
                f"compression={comp} is not valid for {sample_tokens} input tokens. "
                f"Valid defaults: {recommended_targets(sample_tokens)}"
            )
        comp_key = str(comp)
        rows = []
        correct = 0
        total = 0
        scored = 0
        n_gt_known = 0
        correct_gt_known = 0

        t_encode = []
        t_comp = []
        t_llm = []
        t_total = []
        gen_token_counts = []

        for it in tqdm(items, desc=f"comp={comp}"):
            img = Image.open(it.path).convert("RGB")

            t0 = time.perf_counter()
            te0 = time.perf_counter()
            patches = enc.encode(img)
            te1 = time.perf_counter()

            tc0 = time.perf_counter()
            patches_c = compress_27x27_tokens(patches, target_tokens=comp)
            tc1 = time.perf_counter()

            user_prompt = _task_prompt(it.task, prompt_style=args.prompt_style)

            tl0 = time.perf_counter()
            gen_out = vlm.generate(
                image_tokens=patches_c,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=eval_cap,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                do_sample=args.do_sample,
                return_num_new_tokens=True,
            )
            text, n_gen = cast(Tuple[str, int], gen_out)
            tl1 = time.perf_counter()
            t1 = time.perf_counter()

            pred = _score_and_pred(task=it.task, text=text)
            gt = _ground_truth(task=it.task, labels=it.labels)
            is_scored = gt != "unknown" and pred != "unknown"
            is_correct = (pred == gt) and is_scored

            if gt != "unknown":
                n_gt_known += 1
                if pred == gt:
                    correct_gt_known += 1

            total += 1
            correct += int(is_correct)
            scored += int(is_scored)
            gen_token_counts.append(n_gen)

            t_encode.append(te1 - te0)
            t_comp.append(tc1 - tc0)
            t_llm.append(tl1 - tl0)
            t_total.append(t1 - t0)

            rows.append(
                {
                    "id": it.id,
                    "task": it.task,
                    "path": it.path,
                    "gt": gt,
                    "pred": pred,
                    "text": text,
                    "response_length_chars": len(text),
                    "response_length_words": len(text.split()),
                    "prompt_echo_detected": _prompt_echo_detected(text),
                    "stop_reason": "max_new_tokens" if n_gen >= eval_cap else "eos_or_stop",
                    "unknown_subtype": _unknown_subtype(task=it.task, pred=pred, text=text),
                    "n_gen_tokens": n_gen,
                    "timing_s": {
                        "encode": te1 - te0,
                        "compress": tc1 - tc0,
                        "llm": tl1 - tl0,
                        "total": t1 - t0,
                    },
                }
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results["compression"][comp_key] = {
            "n_items": total,
            "n_scored": scored,
            "accuracy_scored": (correct / scored) if scored else None,
            "n_gt_known": n_gt_known,
            "correct_gt_known": correct_gt_known,
            "accuracy_gt_known": (correct_gt_known / n_gt_known) if n_gt_known else None,
            "unknown_rate": ((n_gt_known - scored) / n_gt_known) if n_gt_known else None,
            "accuracy_overall": (correct / total) if total else None,
            "by_task": _aggregate_by_task(rows),
            "timing_s": _summarize_timings(t_encode, t_comp, t_llm, t_total),
            "gen_new_tokens": _summarize_ints(gen_token_counts),
            "peak_mem_mb": memory_mb(),
            "items": rows,
        }

    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


def _summarize_timings(encode, comp, llm, total) -> Dict[str, Any]:
    def s(x):
        if not x:
            return None
        x = np.array(x, dtype=np.float64)
        return {
            "mean": float(x.mean()),
            "p50": float(np.quantile(x, 0.50)),
            "p95": float(np.quantile(x, 0.95)),
        }
    return {"encode": s(encode), "compress": s(comp), "llm": s(llm), "total": s(total)}


def _summarize_ints(values: List[int]) -> Optional[Dict[str, Any]]:
    if not values:
        return None
    x = np.array(values, dtype=np.float64)
    return {
        "mean": float(x.mean()),
        "p50": float(np.quantile(x, 0.50)),
        "p95": float(np.quantile(x, 0.95)),
        "max": int(x.max()),
    }


def _aggregate_by_task(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    buckets: Dict[str, Dict[str, int]] = {}
    for r in rows:
        task = str(r.get("task", "unknown"))
        if task not in buckets:
            buckets[task] = {
                "n_items": 0,
                "n_gt_known": 0,
                "correct_gt_known": 0,
                "n_scored": 0,
                "correct_scored": 0,
                "unknown_ambiguous": 0,
                "unknown_low_confidence": 0,
            }
        b = buckets[task]
        b["n_items"] += 1
        gt = r.get("gt", "unknown")
        pred = r.get("pred", "unknown")
        if gt != "unknown":
            b["n_gt_known"] += 1
            if pred == gt:
                b["correct_gt_known"] += 1
        if gt != "unknown" and pred != "unknown":
            b["n_scored"] += 1
            if pred == gt:
                b["correct_scored"] += 1
        subtype = r.get("unknown_subtype")
        if subtype == "ambiguous":
            b["unknown_ambiguous"] += 1
        elif subtype == "low_confidence":
            b["unknown_low_confidence"] += 1

    out: Dict[str, Any] = {}
    for task in sorted(buckets.keys()):
        b = buckets[task]
        nk = b["n_gt_known"]
        ns = b["n_scored"]
        out[task] = {
            "n_items": b["n_items"],
            "n_gt_known": nk,
            "correct_gt_known": b["correct_gt_known"],
            "accuracy_gt_known": (b["correct_gt_known"] / nk) if nk else None,
            "n_scored": ns,
            "accuracy_scored": (b["correct_scored"] / ns) if ns else None,
            "unknown_rate": ((nk - ns) / nk) if nk else None,
            "unknown_ambiguous": (b["unknown_ambiguous"] / nk) if nk else None,
            "unknown_low_confidence": (b["unknown_low_confidence"] / nk) if nk else None,
        }
    return out


def _prompt_echo_detected(text: str) -> bool:
    low = (text or "").lower()
    patterns = [
        r"start\s+your\s+response\s+with\s+exactly\s+one\s+word",
        r"crosswalk\s+walk\s+signal",
        r"are\s+there\s+stairs\s+or\s+steps",
        r"is\s+there\s+an\s+obstacle\s+ahead",
        r"assistant\s*:",
        r"user\s*:",
    ]
    return any(re.search(p, low) is not None for p in patterns)


def _unknown_subtype(*, task: str, pred: str, text: str) -> Optional[str]:
    if pred != "unknown":
        return None
    low = (text or "").lower()
    if task == "crosswalk_signal":
        has_red = bool(re.search(r"\bred\b", low))
        has_green = bool(re.search(r"\bgreen\b", low))
        if has_red and has_green:
            return "ambiguous"
        return "low_confidence"
    if task in ("stairs", "obstacles"):
        has_no = bool(re.search(r"\bno\b|\bnone\b|\bclear\b", low))
        has_yes = bool(re.search(r"\byes\b|\bpresent\b|\bobstacle\b|\bstairs\b|\bstep\b", low))
        if has_yes and has_no:
            return "ambiguous"
        return "low_confidence"
    return "low_confidence"


def _task_prompt(task: str, prompt_style: str = "label_plus_action") -> str:
    if prompt_style == "label_only":
        if task == "crosswalk_signal":
            return "Answer with exactly one word only: red|green|unknown"
        if task == "stairs":
            return "Answer with exactly one word only: yes|no|unknown"
        if task == "obstacles":
            return "Answer with exactly one word only: yes|no|unknown"
        return "Answer with one short phrase only."

    if task == "crosswalk_signal":
        return (
            "Crosswalk walk signal is it red or green? "
            "Start your response with exactly one word: red|green|unknown. "
            "Then give a short action (one short clause)."
        )
    if task == "stairs":
        return (
            "Are there stairs or steps? "
            "Start your response with exactly one word: yes|no. "
            "Then give a short action (one short clause)."
        )
    if task == "obstacles":
        return (
            "Is there an obstacle ahead? "
            "Start your response with exactly one word: yes|no. "
            "Then give a short action (one short clause)."
        )
    return "Give short navigation advice for safe walking."


def _ground_truth(task: str, labels: Dict[str, str]) -> str:
    if task == "crosswalk_signal":
        return labels.get("walk_signal", "unknown")
    if task == "stairs":
        return labels.get("stairs_present", "unknown")
    if task == "obstacles":
        return labels.get("obstacle_present", "unknown")
    return "unknown"


def _score_and_pred(*, task: str, text: str) -> str:
    if task == "crosswalk_signal":
        return parse_crosswalk_signal(text)
    if task in ("stairs", "obstacles"):
        return parse_yes_no(text)
    return "unknown"


if __name__ == "__main__":
    main()