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
    
    # Ensure audio is float32
    audio = np.asarray(audio, dtype=np.float32)
    
    # Try scipy first (most common)
    try:
        import scipy.io.wavfile
        scipy.io.wavfile.write(str(output_path), sample_rate, audio)
        if verbose:
            print(f"[audio] Saved WAV using scipy.io.wavfile: {output_path}")
        return True
    except ImportError:
        pass
    except Exception as e:
        if verbose:
            print(f"[audio] scipy.io.wavfile failed: {e}")
    
    # Try soundfile
    try:
        import soundfile
        soundfile.write(str(output_path), audio, sample_rate)
        if verbose:
            print(f"[audio] Saved WAV using soundfile: {output_path}")
        return True
    except ImportError:
        pass
    except Exception as e:
        if verbose:
            print(f"[audio] soundfile failed: {e}")
    
    # Fallback to wave module (stdlib)
    try:
        import wave
        
        # wave module expects int16
        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        
        n_channels = audio_int16.ndim if audio_int16.ndim > 1 else 1
        n_frames = len(audio_int16)
        sample_width = 2  # 16-bit = 2 bytes
        
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
    
    return False
