#!/usr/bin/env python3
"""
Lightweight realtime VQA server for Jetson.

Accepts length-prefixed JPEG frames over TCP and returns JSON responses.
Designed to pair with realtime_vqa_client.py running on the laptop.
"""
from __future__ import annotations

import argparse
import io
import json
import socket
import struct
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.prompts.load_prompt import load_system_prompt
from src.vision.siglip_encoder import SiglipPatchEncoder
from src.vision.token_compression import compress_27x27_tokens, recommended_targets
from src.vlm.llm_loader import (
    load_llm_fp16_or_fp32,
    infer_expected_hidden_size,
    validate_loaded_identity,
)
from src.vlm.model import SimplePrefixVLM
from src.classification import classify_and_format


_TASK_PROMPTS: Dict[str, str] = {
    "crosswalk_signal": (
        "Task: crosswalk signal classification. "
        "Answer with exactly one label token and nothing else. "
        "Allowed labels: red, green. "
        "If uncertain, choose the safer label red."
    ),
    "stairs": (
        "Task: stairs detection. "
        "Answer with exactly one label token and nothing else. "
        "Allowed labels: yes, no. "
        "If uncertain, choose the safer label yes."
    ),
    "obstacles": (
        "Task: obstacle detection. "
        "Answer with exactly one label token and nothing else. "
        "Allowed labels: yes, no. "
        "If uncertain, choose the safer label yes."
    ),
}


def _recv_exact(conn: socket.socket, nbytes: int) -> Optional[bytes]:
    buf = bytearray()
    while len(buf) < nbytes:
        chunk = conn.recv(nbytes - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _send_message(conn: socket.socket, payload: bytes) -> None:
    header = struct.pack("!I", len(payload))
    conn.sendall(header + payload)


def _load_image_proj(vlm: SimplePrefixVLM, image_proj_path: Optional[str], device: str) -> Optional[str]:
    if not image_proj_path:
        return None
    proj_path = Path(image_proj_path)
    if proj_path.is_dir():
        proj_path = proj_path / "image_proj.pt"
    if not proj_path.exists():
        raise FileNotFoundError("image_proj not found: {0}".format(proj_path))
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


def _handle_client(
    conn: socket.socket,
    encoder: SiglipPatchEncoder,
    vlm: SimplePrefixVLM,
    system_prompt: str,
    task: str,
    target_tokens: int,
    max_new_tokens: int,
    verbose: bool,
) -> None:
    valid_targets: Optional[list] = None
    frame_count = 0

    while True:
        header = _recv_exact(conn, 4)
        if not header:
            break
        length = struct.unpack("!I", header)[0]
        if length == 0:
            break
        payload = _recv_exact(conn, length)
        if payload is None:
            break

        frame_count += 1
        image = Image.open(io.BytesIO(payload)).convert("RGB")
        t0 = time.perf_counter()
        with torch.no_grad():
            image_tokens = encoder.encode(image)
            if valid_targets is None:
                valid_targets = recommended_targets(int(image_tokens.shape[1]))
                if target_tokens not in valid_targets:
                    raise ValueError(
                        "compression={0} invalid for {1} tokens. Try: {2}".format(
                            target_tokens,
                            int(image_tokens.shape[1]),
                            valid_targets,
                        )
                    )
            image_tokens = compress_27x27_tokens(image_tokens, target_tokens=target_tokens)
            user_prompt = _TASK_PROMPTS.get(task, _TASK_PROMPTS["obstacles"])
            text = vlm.generate(
                image_tokens=image_tokens,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=max_new_tokens,
            )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        label, spoken = classify_and_format(task, text)

        if verbose:
            print("[server] frame={0} label={1} time={2:.0f}ms".format(frame_count, label, dt_ms))

        response = {
            "text": text,
            "label": label,
            "spoken": spoken,
            "latency_ms": dt_ms,
            "task": task,
        }
        _send_message(conn, json.dumps(response).encode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Realtime VQA server (Jetson)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--siglip", type=str, default="google/siglip-base-patch16-384")
    parser.add_argument("--llm", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--image-proj", type=str, default=None)
    parser.add_argument("--compression", type=int, default=192)
    parser.add_argument("--task", type=str, default="obstacles")
    parser.add_argument("--expected-hidden-size", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print("[server] device={0} dtype={1}".format(device, dtype))

    encoder = SiglipPatchEncoder.from_pretrained(args.siglip, device=device, dtype=dtype)
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
    image_proj_path = _load_image_proj(vlm, args.image_proj, device)
    if image_proj_path:
        print("[server] image_proj loaded: {0}".format(image_proj_path))

    system_prompt = load_system_prompt()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((args.host, args.port))
        sock.listen(1)
        print("[server] listening on {0}:{1}".format(args.host, args.port))
        conn, addr = sock.accept()
        with conn:
            print("[server] connected from {0}:{1}".format(addr[0], addr[1]))
            try:
                _handle_client(
                    conn,
                    encoder=encoder,
                    vlm=vlm,
                    system_prompt=system_prompt,
                    task=args.task,
                    target_tokens=args.compression,
                    max_new_tokens=args.max_new_tokens,
                    verbose=args.verbose,
                )
            except Exception as exc:
                print("[server] ERROR: {0}".format(exc))
                return 1

    print("[server] shutdown")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
