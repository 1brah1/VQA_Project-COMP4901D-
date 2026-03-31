"""
src/audio_utils.py
==================
Audio utilities for WAV file saving with multiple backend support.

Tries in order:
1. scipy.io.wavfile (most common)
2. soundfile (fast, good quality)
3. wave module (Python stdlib fallback)
"""
from pathlib import Path
from typing import Optional
import numpy as np


def save_wav(
    audio: np.ndarray,
    output_path: str,
    sample_rate: int = 24000,
    verbose: bool = False,
) -> bool:
    """
    Save numpy array as WAV file using available backends.
    
    Parameters
    ----------
    audio : np.ndarray
        PCM audio, shape (n_samples,) or (n_samples, n_channels)
    output_path : str
        Output file path
    sample_rate : int
        Sample rate in Hz
    verbose : bool
        Print backend info
        
    Returns
    -------
    bool
        True if save succeeded, False otherwise
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure audio is float32 and finite.
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        if verbose:
            print("[audio] Empty audio buffer; skipping WAV save")
        return False

    finite_mask = np.isfinite(audio)
    if not bool(finite_mask.any()):
        if verbose:
            print("[audio] Audio buffer is non-finite (all NaN/Inf); skipping WAV save")
        return False
    if not bool(finite_mask.all()):
        audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak

    # Always write PCM16 for broad player compatibility.
    audio_int16 = np.clip(audio * 32767.0, -32768.0, 32767.0).astype(np.int16)

    # Treat all-zero audio as synthesis failure, not a valid artifact.
    if not np.any(audio_int16):
        if verbose:
            print("[audio] Audio buffer is fully silent (all zeros); skipping WAV save")
        return False

    # Use stdlib wave for deterministic PCM output.
    try:
        import wave

        n_channels = int(audio_int16.shape[1]) if audio_int16.ndim > 1 else 1
        sample_width = 2  # 16-bit PCM

        with wave.open(str(output_path), "w") as wav:
            wav.setnchannels(n_channels)
            wav.setsampwidth(sample_width)
            wav.setframerate(sample_rate)
            wav.writeframes(audio_int16.tobytes())

        if verbose:
            print(f"[audio] Saved WAV using wave module: {output_path}")
        return True
    except Exception as e:
        if verbose:
            print(f"[audio] wave module failed: {e}")

    # Fallback: attempt scipy with int16.
    try:
        import scipy.io.wavfile
        scipy.io.wavfile.write(str(output_path), sample_rate, audio_int16)
        if verbose:
            print(f"[audio] Saved WAV using scipy.io.wavfile fallback: {output_path}")
        return True
    except Exception as e:
        if verbose:
            print(f"[audio] scipy fallback failed: {e}")

    return False
