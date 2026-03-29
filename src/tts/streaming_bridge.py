"""
src/tts/streaming_bridge.py
===========================
Word-triggered streaming bridge from VLM output to VibeVoice TTS.

Two public classes
------------------
VibeVoiceTTSService
    Standalone Python wrapper around VibeVoice-Realtime (no FastAPI).
    Adapted from demo/web/app.py::StreamingTTSService with a configurable
    voices_dir so it can be used outside the demo directory structure.

WordBufferedTTSBridge
    Accepts tokens from a VLM generator, buffers until `word_threshold`
    complete words have accumulated, then fires VibeVoice TTS in a
    background thread.  Audio is played via sounddevice (if installed) or
    collected into self.audio_chunks.  Timing events are recorded in
    self.events (BridgeEvents) for latency profiling.

Python 3.8-compatible.
"""
from __future__ import annotations

import copy
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from typing import Dict, Iterator, List, Optional

import numpy as np
import torch

# Apply compat patches before importing VibeVoice (Python 3.8 + transformers 4.46.3)
from src.compat_patches import apply_vibevoice_compat_patches, apply_forward_filters
apply_vibevoice_compat_patches()

# VibeVoice imports — optional so the module loads even without them installed.
try:
    from vibevoice.modular.modeling_vibevoice_streaming_inference import (
        VibeVoiceStreamingForConditionalGenerationInference,
    )
    from vibevoice.processor.vibevoice_streaming_processor import (
        VibeVoiceStreamingProcessor,
    )
    from vibevoice.modular.streamer import AudioStreamer
    _VIBEVOICE_AVAILABLE = True
except ImportError:
    _VIBEVOICE_AVAILABLE = False

SAMPLE_RATE = 24_000
_SENTINEL = object()  # queue poison-pill


# ─────────────────────────────────────────────────────────────────────────────
# VibeVoiceTTSService
# ─────────────────────────────────────────────────────────────────────────────

class VibeVoiceTTSService:
    """
    Standalone Python TTS service wrapping VibeVoice-Realtime.

    Parameters
    ----------
    model_path      : HuggingFace repo ID or local path to the VibeVoice model.
    voices_dir      : Path to the directory containing .pt voice preset files
                      (e.g. /path/to/VibeVoice/voices/streaming_model).
    device          : "cuda" | "cpu" | "mps"
    inference_steps : DDPM inference steps (5 is fast; 10 is higher quality).

    Usage::

        svc = VibeVoiceTTSService(
            "microsoft/VibeVoice-Realtime-0.5B",
            voices_dir="/path/to/VibeVoice/voices/streaming_model",
        )
        svc.load()
        for chunk in svc.stream("Obstacle detected ahead."):
            sounddevice.play(chunk, samplerate=24000, blocking=True)
    """

    def __init__(
        self,
        model_path: str,
        voices_dir: Optional[str] = None,
        device: str = "cuda",
        inference_steps: int = 5,
    ) -> None:
        if not _VIBEVOICE_AVAILABLE:
            raise ImportError(
                "vibevoice package not found.  "
                "Install it from: https://github.com/microsoft/VibeVoice  "
                "or run: pip install -e /path/to/VibeVoice"
            )
        self.model_path = model_path
        self.voices_dir = Path(voices_dir) if voices_dir else None
        self.inference_steps = inference_steps
        self.sample_rate = SAMPLE_RATE

        self.processor: Optional[VibeVoiceStreamingProcessor] = None
        self.model: Optional[VibeVoiceStreamingForConditionalGenerationInference] = None
        self.voice_presets: Dict[str, Path] = {}
        self.default_voice_key: Optional[str] = None
        self._voice_cache: Dict[str, object] = {}

        if device == "mps" and not torch.backends.mps.is_available():
            print("[TTS] MPS not available; falling back to CPU")
            device = "cpu"
        self.device = device
        self._torch_device = torch.device(device)

    # ------------------------------------------------------------------

    def load(self) -> None:
        """Download / load model weights and voice presets into memory."""
        print(f"[TTS] Loading processor from {self.model_path}")
        self.processor = VibeVoiceStreamingProcessor.from_pretrained(self.model_path)

        if self.device == "mps":
            load_dtype, device_map, attn_impl = torch.float32, None, "sdpa"
        elif self.device == "cuda":
            load_dtype, device_map, attn_impl = torch.bfloat16, "cuda", "flash_attention_2"
        else:
            load_dtype, device_map, attn_impl = torch.float32, "cpu", "sdpa"

        try:
            self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                self.model_path,
                torch_dtype=load_dtype,
                device_map=device_map,
                attn_implementation=attn_impl,
            )
            if self.device == "mps":
                self.model.to("mps")
        except Exception:
            if attn_impl == "flash_attention_2":
                print("[TTS] flash_attention_2 unavailable; retrying with SDPA")
                self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map=device_map,
                    attn_implementation="sdpa",
                )
            else:
                raise

        self.model.eval()
        self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
            self.model.model.noise_scheduler.config,
            algorithm_type="sde-dpmsolver++",
            beta_schedule="squaredcos_cap_v2",
        )
        self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)

        # Apply compat patch 4 & 5: filter unknown kwargs from model.forward
        apply_forward_filters(self.model)

        self.voice_presets = self._load_voice_presets()
        first_key = next(iter(self.voice_presets))
        self.default_voice_key = (
            "en-Carter_man" if "en-Carter_man" in self.voice_presets else first_key
        )
        self._ensure_voice_cached(self.default_voice_key)
        print("[TTS] Model ready.")

    # ------------------------------------------------------------------

    def _load_voice_presets(self) -> Dict[str, Path]:
        if self.voices_dir is None:
            raise RuntimeError(
                "voices_dir not set.  Pass it to VibeVoiceTTSService(..., voices_dir=...)"
            )
        if not self.voices_dir.exists():
            raise RuntimeError(f"Voices directory not found: {self.voices_dir}")
        presets: Dict[str, Path] = {}
        for pt in self.voices_dir.rglob("*.pt"):
            presets[pt.stem] = pt
        if not presets:
            raise RuntimeError(f"No .pt voice presets found in {self.voices_dir}")
        print(f"[TTS] Found {len(presets)} voice preset(s)")
        return dict(sorted(presets.items()))

    def _ensure_voice_cached(self, key: str) -> object:
        if key not in self.voice_presets:
            raise RuntimeError(f"Voice preset {key!r} not found")
        if key not in self._voice_cache:
            pt = self.voice_presets[key]
            print(f"[TTS] Loading voice preset {key!r} from {pt}")
            self._voice_cache[key] = torch.load(
                pt, map_location=self._torch_device, weights_only=False
            )
        return self._voice_cache[key]

    # ------------------------------------------------------------------

    def stream(
        self,
        text: str,
        voice_key: Optional[str] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> Iterator[np.ndarray]:
        """
        Yield float32 mono PCM audio chunks (24 kHz) for `text`.

        Audio generation runs in a background thread; this method yields
        chunks as they become available (streaming / low-latency).
        """
        if not text.strip():
            # For empty text, yield nothing (valid empty generator)
            return iter([])
        if not self.processor or not self.model:
            raise RuntimeError("Call load() first")

        text = text.replace("\u2019", "'")  # smart apostrophe → straight
        key = (
            voice_key
            if voice_key and voice_key in self.voice_presets
            else self.default_voice_key
        )
        prefilled = self._ensure_voice_cached(key)

        processed = self.processor.process_input_with_cached_prompt(
            text=text.strip(),
            cached_prompt=prefilled,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        inputs = {
            k: (v.to(self._torch_device) if hasattr(v, "to") else v)
            for k, v in processed.items()
        }

        audio_streamer = AudioStreamer(batch_size=1, stop_signal=None, timeout=None)
        errors: List[Exception] = []
        stop_signal = stop_event or threading.Event()

        def _generate() -> None:
            try:
                self.model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=1.5,
                    tokenizer=self.processor.tokenizer,
                    generation_config={
                        "do_sample": False,
                        "temperature": 1.0,
                        "top_p": 1.0,
                    },
                    audio_streamer=audio_streamer,
                    stop_check_fn=stop_signal.is_set,
                    verbose=False,
                    refresh_negative=True,
                    all_prefilled_outputs=copy.deepcopy(prefilled),
                )
            except Exception as exc:
                errors.append(exc)
                audio_streamer.end()

        t = threading.Thread(target=_generate, daemon=True)
        t.start()
        try:
            for chunk in audio_streamer.get_stream(0):
                if torch.is_tensor(chunk):
                    chunk = chunk.detach().cpu().to(torch.float32).numpy()
                else:
                    chunk = np.asarray(chunk, dtype=np.float32)
                if chunk.ndim > 1:
                    chunk = chunk.reshape(-1)
                peak = float(np.max(np.abs(chunk))) if chunk.size else 0.0
                if peak > 1.0:
                    chunk = chunk / peak
                yield chunk.astype(np.float32, copy=False)
        finally:
            stop_signal.set()
            audio_streamer.end()
            t.join()
            if errors:
                raise errors[0]


# ─────────────────────────────────────────────────────────────────────────────
# Timing events
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BridgeEvents:
    """
    Timestamps recorded by WordBufferedTTSBridge (all in perf_counter seconds).
    Call the *_ms properties to get millisecond deltas suitable for a
    latency breakdown table.
    """
    t_start: float = 0.0           # bridge.start() called
    t_first_token: float = 0.0     # first VLM chunk received via feed()
    t_tts_triggered: float = 0.0   # TTS stream() first called (after word threshold)
    t_first_audio: float = 0.0     # first audio chunk yielded by TTS
    t_playback_done: float = 0.0   # wait() returned

    @property
    def ttft_ms(self) -> float:
        """Time-to-first-token from bridge.start()."""
        if not self.t_first_token:
            return 0.0
        return (self.t_first_token - self.t_start) * 1000.0

    @property
    def tts_trigger_ms(self) -> float:
        """Elapsed from start to TTS trigger."""
        if not self.t_tts_triggered:
            return 0.0
        return (self.t_tts_triggered - self.t_start) * 1000.0

    @property
    def tts_first_audio_ms(self) -> float:
        """Latency from TTS trigger to first audio chunk (TTS model latency)."""
        if not self.t_first_audio or not self.t_tts_triggered:
            return 0.0
        return (self.t_first_audio - self.t_tts_triggered) * 1000.0

    @property
    def e2e_first_audio_ms(self) -> float:
        """End-to-end from bridge.start() to first audio."""
        if not self.t_first_audio:
            return 0.0
        return (self.t_first_audio - self.t_start) * 1000.0

    @property
    def e2e_total_ms(self) -> float:
        """End-to-end from bridge.start() to playback complete."""
        if not self.t_playback_done:
            return 0.0
        return (self.t_playback_done - self.t_start) * 1000.0


# ─────────────────────────────────────────────────────────────────────────────
# WordBufferedTTSBridge
# ─────────────────────────────────────────────────────────────────────────────

class WordBufferedTTSBridge:
    """
    Connect VLM streaming output to VibeVoice TTS with word-count buffering.

    The bridge fires TTS as soon as `word_threshold` complete words have been
    received from the VLM, so the user hears the beginning of the response
    while the VLM is still generating the rest.

    Usage::

        bridge = WordBufferedTTSBridge(tts_service, word_threshold=3)
        bridge.start()                        # starts background threads
        for chunk, accepted in vlm.generate_streaming(...):
            bridge.feed(chunk)                # hand each token to the bridge
        bridge.flush()                        # send any trailing text to TTS
        bridge.wait(timeout=30.0)             # block until audio done
        print(bridge.events.e2e_first_audio_ms)  # latency in ms

    Parameters
    ----------
    tts_service     : a loaded VibeVoiceTTSService
    word_threshold  : number of words to buffer before firing TTS (default 3)
    play_audio      : if True, play chunks via sounddevice; otherwise just
                      collect them in self.audio_chunks for later use
    """

    def __init__(
        self,
        tts_service: VibeVoiceTTSService,
        word_threshold: int = 3,
        play_audio: bool = True,
    ) -> None:
        self.tts = tts_service
        self.word_threshold = word_threshold
        self.play_audio = play_audio

        self._buffer: List[str] = []
        self._word_count: int = 0
        self._tts_fired: bool = False
        self._tts_threads: List[threading.Thread] = []
        self._audio_queue: Queue = Queue()
        self._playback_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._playback_sentinel_sent: bool = False

        self.events = BridgeEvents()
        self.audio_chunks: List[np.ndarray] = []

    # ------------------------------------------------------------------

    def start(self) -> None:
        """Call once before feeding tokens."""
        self.events.t_start = time.perf_counter()
        if self.play_audio:
            self._playback_thread = threading.Thread(
                target=self._playback_worker, daemon=True
            )
            self._playback_thread.start()

    def feed(self, text_chunk: str) -> None:
        """
        Feed one VLM output chunk (token / subword string).

        Counts whitespace-delimited words.  When word_threshold is reached for
        the first time, fires TTS in a background thread.
        """
        if not self.events.t_first_token:
            self.events.t_first_token = time.perf_counter()

        self._buffer.append(text_chunk)

        # Count word boundaries introduced by this chunk (each space or newline
        # separates two words when preceded by non-whitespace content).
        self._word_count += text_chunk.count(" ") + text_chunk.count("\n")

        if not self._tts_fired and self._word_count >= self.word_threshold:
            self._fire_tts("".join(self._buffer))
            self._buffer.clear()

    def flush(self) -> None:
        """
        Flush any remaining buffered text to TTS.

        Call this after the VLM generator is exhausted.  Safe to call even if
        word_threshold was already reached (fires a second TTS segment for the
        tail of the response).
        """
        remaining = "".join(self._buffer).strip()
        self._buffer.clear()
        if remaining:
            self._fire_tts(remaining)

    def wait(self, timeout: Optional[float] = None) -> None:
        """
        Block until all TTS threads have finished and the audio queue is drained.
        Records t_playback_done on the events object.
        Thread-safe path: joins all worker threads before touching shared state.
        """
        # Wait for all TTS worker threads to finish
        for t in self._tts_threads:
            if t.is_alive():
                t.join(timeout=timeout)

        # Signal playback worker to stop (if not already sent)
        if not self._playback_sentinel_sent:
            self._audio_queue.put(_SENTINEL)
            self._playback_sentinel_sent = True

        # Wait for playback thread to drain queue
        if self._playback_thread and self._playback_thread.is_alive():
            self._playback_thread.join(timeout=timeout)

        # Record completion time
        if not self.events.t_playback_done:
            self.events.t_playback_done = time.perf_counter()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fire_tts(self, text: str) -> None:
        """Start a background thread that streams TTS for `text`."""
        text = text.strip()
        if not text:
            return
        if not self.events.t_tts_triggered:
            self.events.t_tts_triggered = time.perf_counter()
        self._tts_fired = True

        first_audio_recorded = threading.Event()

        def _worker(t: str) -> None:
            first = True
            try:
                for chunk in self.tts.stream(t, stop_event=self._stop_event):
                    if first:
                        if not self.events.t_first_audio:
                            self.events.t_first_audio = time.perf_counter()
                        first_audio_recorded.set()
                        first = False
                    self._audio_queue.put(chunk)
            except Exception as exc:
                print(f"[TTS bridge] generation error: {exc}")

        thread = threading.Thread(target=_worker, args=(text,), daemon=True)
        self._tts_threads.append(thread)
        thread.start()

    def _playback_worker(self) -> None:
        """Drain the audio queue and play each chunk via sounddevice."""
        try:
            import sounddevice as sd  # type: ignore[import]
            have_sd = True
        except ImportError:
            print("[TTS bridge] sounddevice not installed; collecting audio without playback")
            have_sd = False

        while True:
            item = self._audio_queue.get()
            if item is _SENTINEL:
                break
            chunk: np.ndarray = item  # type: ignore[assignment]
            self.audio_chunks.append(chunk)
            if have_sd:
                try:
                    import sounddevice as sd  # type: ignore[import]
                    sd.play(chunk, samplerate=SAMPLE_RATE, blocking=True)
                except Exception as exc:
                    print(f"[TTS bridge] playback error: {exc}")
