#!/usr/bin/env python3
"""
Laptop client for realtime VQA.

Captures webcam frames, sends JPEGs to Jetson server, and plays TTS locally.
"""
from __future__ import annotations

import argparse
import json
import socket
import struct
import sys
import time
from typing import Optional

try:
    import cv2  # type: ignore
except Exception as exc:
    print("[client] ERROR: OpenCV not available. Install opencv-python on the laptop. ({0})".format(exc))
    raise SystemExit(1)


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


def _init_tts(enabled: bool):
    if not enabled:
        return None
    try:
        import pyttsx3  # type: ignore

        engine = pyttsx3.init()
        return engine
    except Exception as exc:
        print("[client] WARNING: pyttsx3 unavailable ({0}). Continuing without TTS.".format(exc))
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Realtime VQA client (laptop)")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--interval-ms", type=int, default=800)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--no-tts", action="store_true")
    args = parser.parse_args()

    tts_engine = _init_tts(enabled=not args.no_tts)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("[client] ERROR: failed to open camera {0}".format(args.camera))
        return 1

    sock = socket.create_connection((args.host, args.port))
    print("[client] connected to {0}:{1}".format(args.host, args.port))

    last_sent = 0.0
    frame_count = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[client] ERROR: camera read failed")
                break

            now = time.perf_counter()
            if (now - last_sent) * 1000.0 < float(args.interval_ms):
                continue
            last_sent = now

            encode_ok, encoded = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)],
            )
            if not encode_ok:
                print("[client] WARNING: JPEG encode failed; skipping frame")
                continue

            payload = encoded.tobytes()
            _send_message(sock, payload)

            header = _recv_exact(sock, 4)
            if not header:
                print("[client] server closed connection")
                break
            length = struct.unpack("!I", header)[0]
            if length == 0:
                print("[client] server sent zero-length response")
                break
            resp = _recv_exact(sock, length)
            if resp is None:
                print("[client] server closed connection")
                break

            data = json.loads(resp.decode("utf-8"))
            text = str(data.get("spoken") or data.get("text") or "").strip()
            label = str(data.get("label") or "")
            latency = data.get("latency_ms")
            frame_count += 1

            print("[client] #{0} label={1} latency={2}ms text={3}".format(frame_count, label, int(latency or 0), text))

            if tts_engine is not None and text:
                tts_engine.say(text)
                tts_engine.runAndWait()

            if args.max_frames > 0 and frame_count >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        try:
            _send_message(sock, b"")
        except Exception:
            pass
        sock.close()
        cap.release()

    print("[client] shutdown")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
