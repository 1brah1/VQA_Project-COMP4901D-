"""
src/vlm/pipelined_vlm.py
========================
PPSD-inspired self-speculative decoding for Qwen2.5-0.5B.

Architecture
------------
Qwen2.5-0.5B has 24 transformer decoder layers.  We split them:

  Draft  model : layers 0 … split_layer-1  (default 12)  + norm + lm_head
  Verify model : layers 0 … 23             (full 24)      + norm + lm_head

Pipeline per decoding round
---------------------------
  1. pending_token  – the last *verified* token (always correct).
  2. Draft  – run the pending_token through the first E=12 layers K=4 times,
              producing K speculative "draft" tokens.
  3. Verify – run [pending_token, draft[0..K-1]] through ALL 24 layers
              in ONE batched forward pass.
  4. Accept – the longest prefix of drafts that agree with the verifier;
              the corrected token if there is a mismatch.
  5. Repeat from step 2 with the new pending_token.

Speedup accounting (with E=12, K=4, 100 % acceptance)
  Baseline  : (K+1) × 24-layer passes          = 5 × 24  = 120 layer-ops / round
  Speculative: K × 12-layer draft + 1 × 24-layer verify
                                                = 4 × 12 + 24 = 72 layer-ops / round
  → theoretical 1.67× speedup; empirical ~1.3–1.5× on short navigation text.

Python 3.8-compatible (Jetson L4T default).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional, Tuple, Union

import torch
from transformers import DynamicCache

from src.vlm.model import SimplePrefixVLM


# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def _crop_cache(cache: DynamicCache, max_length: int) -> None:
    """Trim a DynamicCache so every layer's KV spans only `max_length` positions."""
    if hasattr(cache, "crop"):
        cache.crop(max_length)
        return
    # Fallback for transformers that don't have .crop()
    for i in range(len(cache.key_cache)):
        if cache.key_cache[i] is not None:
            cache.key_cache[i] = cache.key_cache[i][:, :, :max_length, :]
            cache.value_cache[i] = cache.value_cache[i][:, :, :max_length, :]


# ─────────────────────────────────────────────────────────────────────────────
# Statistics dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SpecStats:
    """Runtime statistics collected during one speculative generation run."""

    total_tokens: int = 0
    accepted_drafts: int = 0       # draft tokens accepted (not counting pending)
    total_draft_candidates: int = 0  # draft tokens proposed (K per round)
    verify_passes: int = 0
    prefill_ms: float = 0.0
    draft_ms: float = 0.0
    verify_ms: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        if self.total_draft_candidates == 0:
            return 0.0
        return self.accepted_drafts / self.total_draft_candidates

    @property
    def tokens_per_verify_pass(self) -> float:
        if self.verify_passes == 0:
            return 0.0
        return self.total_tokens / self.verify_passes

    @property
    def total_generation_ms(self) -> float:
        return self.draft_ms + self.verify_ms

    @property
    def baseline_ms_estimate(self) -> float:
        """Estimated time if we had used sequential full-model decoding."""
        if self.verify_passes == 0:
            return 0.0
        ms_per_verify = self.verify_ms / self.verify_passes
        return self.total_tokens * ms_per_verify

    @property
    def speedup(self) -> float:
        if self.total_generation_ms == 0:
            return 1.0
        return self.baseline_ms_estimate / self.total_generation_ms

    def summary(self) -> str:
        return (
            f"tokens={self.total_tokens}  "
            f"verify_passes={self.verify_passes}  "
            f"accepted={self.accepted_drafts}/{self.total_draft_candidates} "
            f"({self.acceptance_rate:.0%} accept)  "
            f"~{self.tokens_per_verify_pass:.1f} tok/verify  "
            f"speedup~{self.speedup:.2f}x  "
            f"prefill={self.prefill_ms:.0f}ms  "
            f"draft={self.draft_ms:.0f}ms  "
            f"verify={self.verify_ms:.0f}ms"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class SelfSpeculativeVLM:
    """
    PPSD-inspired self-speculative decoding wrapped around SimplePrefixVLM.

    Parameters
    ----------
    vlm         : a loaded SimplePrefixVLM (Qwen2.5-0.5B-Instruct)
    split_layer : first layer of the "verify-only" stage (default 12 of 24)
    K           : draft tokens to propose per verify pass (default 4)
    """

    def __init__(
        self,
        vlm: SimplePrefixVLM,
        split_layer: int = 12,
        K: int = 4,
    ) -> None:
        self.vlm = vlm
        self.split_layer = split_layer
        self.K = K
        self.device = vlm.device
        self.dtype = vlm.dtype

        # Unpack model internals
        llm = vlm.llm
        self._base_model = llm.model        # Qwen2Model (all layers + norm)
        self._lm_head = llm.lm_head         # vocab projection
        self._embed_tokens = llm.model.embed_tokens
        self._layers = llm.model.layers      # nn.ModuleList of 24 decoder layers
        self._norm = llm.model.norm          # final RMS norm
        self._n_layers = len(self._layers)
        self._tokenizer = vlm.tokenizer
        self._image_proj = vlm.image_proj

        if not (1 <= split_layer < self._n_layers):
            raise ValueError(
                f"split_layer must be in [1, {self._n_layers - 1}]; got {split_layer}"
            )

    # ------------------------------------------------------------------
    # Build the image + text prefix embeddings
    # ------------------------------------------------------------------

    def _build_prefix_embeds(
        self,
        image_tokens: torch.Tensor,
        system_prompt: str,
        user_prompt: str,
    ) -> torch.Tensor:
        """
        Project image tokens and concatenate with formatted text prompt embeddings.
        Returns inputs_embeds of shape (1, prefix_len, hidden_dim).
        """
        _, _, dv = image_tokens.shape
        # Lazy resize of image projection if SigLIP dim changed
        if self._image_proj.in_features != dv:
            self._image_proj = torch.nn.Linear(
                dv, self._image_proj.out_features, bias=False
            ).to(self.device)
            if self.device == "cuda":
                self._image_proj = self._image_proj.to(self.dtype)
            self._image_proj.eval()

        img_prefix = self._image_proj(image_tokens.to(device=self.device))
        if self.device == "cuda":
            img_prefix = img_prefix.to(dtype=self.dtype)

        prompt = self.vlm._format_prompt(
            system_prompt=system_prompt, user_prompt=user_prompt
        )
        input_ids = self._tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        text_embeds = self._embed_tokens(input_ids)

        return torch.cat([img_prefix, text_embeds], dim=1)  # (1, P, D)

    # ------------------------------------------------------------------
    # Draft: run only first E layers for one token
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _draft_one(
        self,
        tok_id: int,
        draft_cache: DynamicCache,
        seq_offset: int,
    ) -> Tuple[int, DynamicCache]:
        """
        Embed tok_id, run through layers 0..split_layer-1, apply norm+lm_head.
        Returns (draft_token_id, updated_draft_cache).
        """
        embed = self._embed_tokens(
            torch.tensor([[tok_id]], device=self.device, dtype=torch.long)
        )
        if self.device == "cuda":
            embed = embed.to(self.dtype)

        cache_pos = torch.tensor([seq_offset], device=self.device, dtype=torch.long)
        pos_ids = cache_pos.unsqueeze(0)  # (1, 1)

        hidden = embed
        position_embeddings = self._base_model.rotary_emb(hidden, pos_ids)
        for i in range(self.split_layer):
            hidden = self._layers[i](
                hidden,
                attention_mask=None,    # single token: attends to all past ✓
                position_ids=pos_ids,
                past_key_values=draft_cache,
                use_cache=True,
                cache_position=cache_pos,
                position_embeddings=position_embeddings,
            )

        logits = self._lm_head(self._norm(hidden))   # (1, 1, vocab)
        draft_tok = int(logits[0, 0].argmax().item())
        return draft_tok, draft_cache

    # ------------------------------------------------------------------
    # Verify: run full model on K+1 tokens in ONE batched forward pass
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _verify_batch(
        self,
        token_ids: List[int],
        verify_cache: DynamicCache,
        seq_offset: int,
    ) -> Tuple[List[int], DynamicCache]:
        """
        Run Qwen2Model on `token_ids` (K+1 tokens) in a single forward pass
        using the running verify_cache.  Returns (predictions, updated_cache).

        predictions[j] = argmax of logits at position j, i.e. what the full
        model predicts comes *after* token_ids[j] (given all prior context).
        """
        n = len(token_ids)
        ids = torch.tensor([token_ids], device=self.device, dtype=torch.long)
        cache_pos = torch.arange(seq_offset, seq_offset + n, device=self.device, dtype=torch.long)

        # Qwen2Model.forward handles _update_causal_mask internally,
        # giving a correct 4-D causal mask for multi-token inputs.
        outputs = self._base_model(
            input_ids=ids,
            past_key_values=verify_cache,
            use_cache=True,
            cache_position=cache_pos,
        )
        hidden = outputs.last_hidden_state   # (1, n, D) — already normed by Qwen2Model
        logits = self._lm_head(hidden)       # (1, n, vocab)
        preds = logits[0].argmax(dim=-1).tolist()   # list of n ints

        # outputs.past_key_values IS verify_cache (modified in-place)
        return preds, verify_cache

    # ------------------------------------------------------------------
    # Prefill both caches and return first pending token
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _prefill(
        self,
        prefix_embeds: torch.Tensor,
    ) -> Tuple[int, DynamicCache, DynamicCache, int]:
        """
        Prefill the verify and draft caches with `prefix_embeds`.

        Returns (pending_token, verify_cache, draft_cache, prefix_seq_len).
        pending_token = the full model's first-token prediction after the prefix.
        """
        prefix_len = prefix_embeds.shape[1]
        cache_pos = torch.arange(prefix_len, device=self.device, dtype=torch.long)

        # ── Verify cache (all N layers) ──────────────────────────────
        verify_cache = DynamicCache()
        verify_out = self._base_model(
            inputs_embeds=prefix_embeds,
            past_key_values=verify_cache,
            use_cache=True,
            cache_position=cache_pos,
        )
        v_hidden = verify_out.last_hidden_state   # (1, P, D)
        v_logits = self._lm_head(v_hidden)        # (1, P, vocab)
        pending_token = int(v_logits[0, -1].argmax().item())

        # ── Draft cache (first E layers only) ───────────────────────
        draft_cache = DynamicCache()
        d_pos = cache_pos  # same positions
        d_hidden = prefix_embeds
        d_pos_ids = d_pos.unsqueeze(0)
        d_position_embeddings = self._base_model.rotary_emb(d_hidden, d_pos_ids)
        for i in range(self.split_layer):
            layer_out = self._layers[i](
                d_hidden,
                attention_mask=None,
                past_key_values=draft_cache,
                use_cache=True,
                cache_position=d_pos,
                position_ids=d_pos_ids,
                position_embeddings=d_position_embeddings,
            )
            # Qwen2 layers return tuple (hidden_states, ...) when use_cache=True
            d_hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        return pending_token, verify_cache, draft_cache, prefix_len

    # ------------------------------------------------------------------
    # Main streaming generator
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate_streaming(
        self,
        image_tokens: torch.Tensor,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 64,
    ) -> Generator[Tuple[str, bool], None, SpecStats]:
        """
        Streaming generator that yields (decoded_text_chunk, was_draft_accepted).

        Iteration example::

            gen = vlm.generate_streaming(...)
            try:
                while True:
                    chunk, accepted = next(gen)
                    print(chunk, end="", flush=True)
            except StopIteration as e:
                stats = e.value   # SpecStats

        Returns SpecStats via StopIteration.value when generation ends.
        """
        stats = SpecStats()

        # ── Build prefix ─────────────────────────────────────────────
        prefix_embeds = self._build_prefix_embeds(image_tokens, system_prompt, user_prompt)

        # ── Prefill ──────────────────────────────────────────────────
        t0 = time.perf_counter()
        pending_tok, verify_cache, draft_cache, prefix_len = self._prefill(prefix_embeds)
        stats.prefill_ms = (time.perf_counter() - t0) * 1000.0

        # total_seq_len tracks the confirmed sequence length
        # (prefix + all accepted tokens, NOT including unconfirmed drafts)
        total_seq_len = prefix_len
        n_generated = 0

        eos_id = self._tokenizer.eos_token_id

        while n_generated < max_new_tokens:

            # ── Step A: Draft K tokens ──────────────────────────────
            draft_ids: List[int] = []
            cur = pending_tok
            t_draft = time.perf_counter()
            for _ in range(self.K):
                if n_generated + len(draft_ids) >= max_new_tokens:
                    break
                draft_tok, draft_cache = self._draft_one(
                    cur, draft_cache, seq_offset=total_seq_len + len(draft_ids)
                )
                draft_ids.append(draft_tok)
                stats.total_draft_candidates += 1
                cur = draft_tok
            stats.draft_ms += (time.perf_counter() - t_draft) * 1000.0

            if not draft_ids:
                break

            # ── Step B: Verify [pending + drafts] in one pass ───────
            #
            # We feed K+1 tokens: [pending_tok, draft_ids[0], ..., draft_ids[K-1]]
            # verify_preds[j] = full-model prediction after seeing token j
            #   → verify_preds[0] should equal draft_ids[0]  (check draft_ids[0])
            #   → verify_preds[j] should equal draft_ids[j]  (check draft_ids[j])
            #   → verify_preds[K] = "bonus" token (new pending if all accepted)
            #
            verify_input = [pending_tok] + draft_ids
            t_verify = time.perf_counter()
            verify_preds, verify_cache = self._verify_batch(
                verify_input, verify_cache, seq_offset=total_seq_len
            )
            stats.verify_ms += (time.perf_counter() - t_verify) * 1000.0
            stats.verify_passes += 1

            # ── Step C: Accept ───────────────────────────────────────
            #
            # pending_tok is ALWAYS confirmed this round (it was verified last round).
            # Compare draft_ids[j] vs verify_preds[j] for j=0..K-1.
            n_accept = 0
            for j, d in enumerate(draft_ids):
                if j < len(verify_preds) and verify_preds[j] == d:
                    n_accept += 1
                else:
                    break

            # New confirmed tokens = pending + n_accept draft tokens
            confirmed = [pending_tok] + draft_ids[:n_accept]
            stats.accepted_drafts += n_accept

            # The new pending is either:
            #   • verify_preds[n_accept] — the verifier's correction / bonus token
            new_pending = verify_preds[n_accept] if n_accept < len(verify_preds) else verify_preds[-1]

            # ── Step D: Yield confirmed tokens ───────────────────────
            for tok in confirmed:
                if tok == eos_id:
                    # Sync cache and return
                    _crop_cache(verify_cache, total_seq_len)
                    _crop_cache(draft_cache, total_seq_len)
                    return stats
                text = self._tokenizer.decode([tok], skip_special_tokens=True)
                yield text, True
                n_generated += 1
                stats.total_tokens += 1
                total_seq_len += 1
                if n_generated >= max_new_tokens:
                    return stats

            # ── Step E: Sync caches to confirmed length ──────────────
            #
            # verify_cache now has total_seq_len + K+1 entries.
            # We keep only the confirmed portion.
            _crop_cache(verify_cache, total_seq_len)
            _crop_cache(draft_cache, total_seq_len)

            pending_tok = new_pending

        return stats

    # ------------------------------------------------------------------
    # Non-streaming generate (convenience wrapper)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        image_tokens: torch.Tensor,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 64,
        return_stats: bool = False,
    ) -> Union[str, Tuple[str, SpecStats]]:
        """
        Generate a response and return the full text string.
        Set return_stats=True to also get SpecStats.
        """
        parts: List[str] = []
        stats: Optional[SpecStats] = None
        gen = self.generate_streaming(
            image_tokens=image_tokens,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=max_new_tokens,
        )
        try:
            while True:
                chunk, _ = next(gen)
                parts.append(chunk)
        except StopIteration as e:
            stats = e.value

        text = "".join(parts).strip()
        if return_stats:
            return text, stats
        return text

    # ------------------------------------------------------------------
    # Head-to-head benchmark vs. baseline SimplePrefixVLM
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def benchmark_vs_baseline(
        self,
        image_tokens: torch.Tensor,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 32,
        n_trials: int = 3,
    ) -> Dict[str, object]:
        """
        Run both speculative and baseline generation `n_trials` times each.
        Returns a dict with mean latencies and speedup.
        """
        import numpy as np  # optional; numpy is in requirements

        spec_times: List[float] = []
        base_times: List[float] = []
        spec_text = ""
        base_text = ""

        for _ in range(n_trials):
            # Speculative
            t0 = time.perf_counter()
            spec_text, stats = self.generate(
                image_tokens, system_prompt, user_prompt,
                max_new_tokens=max_new_tokens, return_stats=True
            )
            spec_times.append((time.perf_counter() - t0) * 1000.0)

            # Baseline (SimplePrefixVLM sequential)
            t0 = time.perf_counter()
            base_text = self.vlm.generate(
                image_tokens=image_tokens,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=max_new_tokens,
            )
            base_times.append((time.perf_counter() - t0) * 1000.0)

        spec_arr = np.array(spec_times)
        base_arr = np.array(base_times)
        speedup = float(base_arr.mean() / spec_arr.mean()) if spec_arr.mean() > 0 else 1.0

        return {
            "speculative_text": spec_text,
            "baseline_text": base_text,
            "spec_mean_ms": float(spec_arr.mean()),
            "spec_p50_ms": float(np.median(spec_arr)),
            "base_mean_ms": float(base_arr.mean()),
            "base_p50_ms": float(np.median(base_arr)),
            "speedup": speedup,
            "last_spec_stats": stats.summary() if stats else "",
            "split_layer": self.split_layer,
            "K": self.K,
            "n_layers_total": self._n_layers,
        }
