"""
Fallback TTS backends for Jetson-compatible runs.

Priority for quality/runtime:
1. Piper (fast, offline, excellent Jetson fit)
2. Silero (torch hub, good naturalness, offline after first download)
3. pyttsx3 (system TTS fallback)
"""
from __future__ import annotations

import shutil
import subprocess
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.audio_utils import save_wav


class PiperTTSBackend:
    """Piper CLI backend (preferred on Jetson for stability and speed)."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        piper_executable: str = "piper",
    ) -> None:
        default_model = str(Path.home() / "piper" / "models" / "en_US-lessac-medium.onnx")
        default_exe = str(Path.home() / "piper" / "piper" / "piper")
        env_model = str(Path(os.environ.get("PIPER_MODEL_PATH", default_model)).expanduser())
        env_cfg = os.environ.get("PIPER_CONFIG_PATH")
        env_exe = os.environ.get("PIPER_EXECUTABLE", default_exe)

        self.model_path = model_path or env_model
        self.config_path = config_path or (env_cfg if env_cfg else (self.model_path + ".json"))
        self.piper_executable = piper_executable if piper_executable != "piper" else env_exe
        self.sample_rate = 22050
        self._last_error: Optional[str] = None

    @property
    def available(self) -> bool:
        exe_path = Path(self.piper_executable).expanduser()
        if exe_path.is_file():
            resolved_exe = str(exe_path)
        else:
            resolved_exe = shutil.which(self.piper_executable) or ""
        if not resolved_exe:
            self._last_error = f"piper executable not found: {self.piper_executable}"
            return False
        self.piper_executable = resolved_exe
        if not Path(self.model_path).exists():
            self._last_error = f"piper model missing: {self.model_path}"
            return False
        if self.config_path and not Path(self.config_path).exists():
            self._last_error = f"piper config missing: {self.config_path}"
            return False
        self._last_error = None
        return True

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def synthesize_to_wav(self, text: str, output_path: str) -> bool:
        if not text.strip() or not self.available:
            return False

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        cmd = [self.piper_executable, "--model", self.model_path, "--output_file", str(out)]
        if self.config_path:
            cmd.extend(["--config", self.config_path])

        try:
            proc = subprocess.run(
                cmd,
                input=text.strip().encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if proc.returncode != 0:
                stderr = proc.stderr.decode("utf-8", errors="ignore").strip()
                self._last_error = stderr or f"piper exited with code {proc.returncode}"
                return False
            return out.exists() and out.stat().st_size > 0
        except Exception as exc:
            self._last_error = str(exc)
            return False


class SileroTTSBackend:
    """Silero TTS backend with lazy model load."""

    def __init__(
        self,
        device: str = "cpu",
        language: str = "en",
        speaker_model: str = "v3_en",
        voice: str = "en_0",
        sample_rate: int = 24000,
    ) -> None:
        self.device = device
        self.language = language
        self.speaker_model = speaker_model
        self.voice = voice
        self.sample_rate = sample_rate
        self._model = None
        self._available: Optional[bool] = None
        self._last_error: Optional[str] = None

    @property
    def available(self) -> bool:
        if self._available is None:
            self._available = self._try_load()
        return bool(self._available)

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def _try_load(self) -> bool:
        try:
            model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-models",
                model="silero_tts",
                language=self.language,
                speaker=self.speaker_model,
            )
            model = model.to(self.device)
            self._model = model
            self._last_error = None
            return True
        except Exception as exc:
            self._last_error = str(exc)
            self._model = None
            return False

    def synthesize_to_wav(self, text: str, output_path: str) -> bool:
        if not text.strip():
            return False
        if not self.available or self._model is None:
            return False
        try:
            audio = self._model.apply_tts(
                text=text.strip(),
                speaker=self.voice,
                sample_rate=self.sample_rate,
            )
            if torch.is_tensor(audio):
                audio = audio.detach().cpu().to(torch.float32).numpy()
            audio_np = np.asarray(audio, dtype=np.float32)
            if audio_np.ndim > 1:
                audio_np = audio_np.reshape(-1)
            return save_wav(audio_np, output_path, sample_rate=self.sample_rate, verbose=False)
        except Exception as exc:
            self._last_error = str(exc)
            return False


class Pyttsx3TTSBackend:
    """pyttsx3 fallback backend (works well on constrained systems)."""

    def __init__(self, rate: int = 155) -> None:
        self.sample_rate = 24000
        self._engine = None
        self._available = False
        self._last_error: Optional[str] = None
        try:
            import pyttsx3  # type: ignore[import]

            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", rate)
            self._available = True
        except Exception as exc:
            self._last_error = str(exc)

    @property
    def available(self) -> bool:
        return self._available

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def synthesize_to_wav(self, text: str, output_path: str) -> bool:
        if not text.strip() or not self._available or self._engine is None:
            return False
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._engine.save_to_file(text.strip(), str(out))
            self._engine.runAndWait()
            return out.exists()
        except Exception as exc:
            self._last_error = str(exc)
            return False
