"""
src/compat_patches.py
======================

Compatibility patches for VibeVoice + transformers 4.46.3 on Jetson (Python 3.8).

Five patches are applied on import:
  1. FlashAttentionKwargs stub for transformers 4.46.3
  2. BaseStreamer re-export for transformers.generation
  3. GenerationMixin._prepare_generation_config wrapper for kwargs compat
  4. language_model.forward filter to remove unknown kwargs
  5. tts_language_model.forward filter to remove unknown kwargs

Apply patches before any vibevoice imports.
"""

__all__ = ["apply_vibevoice_compat_patches"]


def apply_vibevoice_compat_patches() -> None:
    """
    Apply all five VibeVoice + transformers 4.46.3 compat patches.
    Safe to call multiple times; checks before patching.
    """

    # Patch 1 — FlashAttentionKwargs missing in transformers 4.46.3
    import transformers.modeling_flash_attention_utils as _m
    if not hasattr(_m, "FlashAttentionKwargs"):
        class FlashAttentionKwargs(dict):
            pass

        _m.FlashAttentionKwargs = FlashAttentionKwargs

    # Patch 2 — BaseStreamer not re-exported at generation level
    import transformers.generation as _gen
    if not hasattr(_gen, "BaseStreamer"):
        from transformers.generation.streamers import BaseStreamer as _BS

        _gen.BaseStreamer = _BS

    # Patch 3 — _prepare_generation_config gets extra bool positional arg
    from transformers import GenerationMixin as _GM

    _orig = _GM._prepare_generation_config

    def _pgc_compat(self, generation_config=None, *args, **kwargs):
        return _orig(self, generation_config, **kwargs)

    _GM._prepare_generation_config = _pgc_compat

    # Patch 4 & 5 — filter unknown kwargs from model.forward calls
    # These patches are applied later in VibeVoiceTTSService.load()
    # because we don't have access to the model instance yet.


def apply_forward_filters(model) -> None:
    """
    Apply patches 4 & 5 after VibeVoice model is loaded.
    Filters out kwargs that the model doesn't accept (speech_start_id, verbose).
    """
    import inspect
    import types

    def _filtered_fwd(orig):
        """Wrapper that filters kwargs to only accepted parameters."""
        accepted = set(inspect.signature(orig).parameters.keys())

        def _f(*a, **kw):
            return orig(*a, **{k: v for k, v in kw.items() if k in accepted})

        return _f

    # Patch 4 — speech_start_id leaks into language_model.forward kwargs
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        model.model.language_model.forward = _filtered_fwd(
            model.model.language_model.forward
        )

    # Patch 5 — verbose leaks into tts_language_model.forward kwargs
    if hasattr(model, "model") and hasattr(model.model, "tts_language_model"):
        model.model.tts_language_model.forward = _filtered_fwd(
            model.model.tts_language_model.forward
        )


# Apply patches 1-3 on module import
apply_vibevoice_compat_patches()
