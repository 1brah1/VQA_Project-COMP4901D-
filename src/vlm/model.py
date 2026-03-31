from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _extract_model_identity(llm_name_or_path: str, llm: Any, mode: str) -> Dict[str, Any]:
    cfg = getattr(llm, "config", None)
    hidden_size = getattr(cfg, "hidden_size", None)
    num_layers = getattr(cfg, "num_hidden_layers", None)
    vocab_size = getattr(cfg, "vocab_size", None)
    return {
        "model_id": llm_name_or_path,
        "mode": mode,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "vocab_size": vocab_size,
    }


@dataclass
class SimplePrefixVLM:
    """
    Minimal VLM wrapper: project image tokens into the LLM embedding space and
    prepend them to the text prompt as a learned continuous prefix.

    This is intentionally simple so we can benchmark token-compression speed/quality.
    """

    tokenizer: any
    llm: any
    image_proj: torch.nn.Linear
    device: str
    dtype: torch.dtype
    model_identity: Dict[str, Any]

    @classmethod
    def from_pretrained(
        cls,
        llm_name_or_path: str,
        *,
        device: Union[Literal["cpu", "cuda"], str] = "cpu",
        dtype: torch.dtype = torch.float32,
        image_token_dim: Optional[int] = None,
    ) -> "SimplePrefixVLM":
        tokenizer = AutoTokenizer.from_pretrained(llm_name_or_path, use_fast=True)
        llm = AutoModelForCausalLM.from_pretrained(llm_name_or_path)

        llm.eval()
        llm.to(device=device)
        if device == "cuda":
            llm.to(dtype=dtype)

        embed_dim = llm.get_input_embeddings().weight.shape[1]
        if image_token_dim is None:
            image_token_dim = embed_dim
        image_proj = torch.nn.Linear(image_token_dim, embed_dim, bias=False)
        image_proj.to(device=device)
        if device == "cuda":
            image_proj.to(dtype=dtype)
        image_proj.eval()

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        identity = _extract_model_identity(llm_name_or_path, llm, "fp16")
        print(
            "[SimplePrefixVLM] Active model: "
            f"id={identity['model_id']}, hidden_size={identity['hidden_size']}, "
            f"layers={identity['num_layers']}, image_proj={image_token_dim}->{embed_dim}"
        )

        return cls(
            tokenizer=tokenizer,
            llm=llm,
            image_proj=image_proj,
            device=str(device),
            dtype=dtype,
            model_identity=identity,
        )

    @classmethod
    def from_loaded_llm(
        cls,
        *,
        tokenizer: any,
        llm: any,
        device: Union[Literal["cpu", "cuda"], str],
        dtype: torch.dtype,
        image_token_dim: int,
    ) -> "SimplePrefixVLM":
        embed_dim = llm.get_input_embeddings().weight.shape[1]
        image_proj = torch.nn.Linear(image_token_dim, embed_dim, bias=False).to(device=device)
        if str(device) == "cuda":
            image_proj = image_proj.to(dtype)
        image_proj.eval()

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        llm_name_or_path = getattr(llm, "name_or_path", "unknown")
        identity = _extract_model_identity(llm_name_or_path, llm, "loaded")
        print(
            "[SimplePrefixVLM] Active model: "
            f"id={identity['model_id']}, hidden_size={identity['hidden_size']}, "
            f"layers={identity['num_layers']}, image_proj={image_token_dim}->{embed_dim}"
        )

        return cls(
            tokenizer=tokenizer,
            llm=llm,
            image_proj=image_proj,
            device=str(device),
            dtype=dtype,
            model_identity=identity,
        )

    @torch.inference_mode()
    def generate(
        self,
        *,
        image_tokens: torch.Tensor,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.2,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = False,
        return_num_new_tokens: bool = False,
    ) -> Union[str, Tuple[str, int]]:
        if image_tokens.ndim != 3:
            raise ValueError(f"Expected image_tokens (B,N,D); got {tuple(image_tokens.shape)}")

        b, _n, dv = image_tokens.shape
        if b != 1:
            raise ValueError("This minimal wrapper currently supports batch_size=1 for benchmarking simplicity.")

        if self.image_proj.in_features != dv:
            self.image_proj = torch.nn.Linear(dv, self.image_proj.out_features, bias=False).to(self.device)
            if self.device == "cuda":
                self.image_proj = self.image_proj.to(self.dtype)
            self.image_proj.eval()

        proj_dtype = self.image_proj.weight.dtype
        img_prefix = self.image_proj(image_tokens.to(device=self.device, dtype=proj_dtype))
        if self.device == "cuda":
            img_prefix = img_prefix.to(dtype=self.dtype)

        prompt = self._format_prompt(system_prompt=system_prompt, user_prompt=user_prompt)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([img_prefix, text_embeds], dim=1)
        attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=self.device)

        gen_kwargs = dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            synced_gpus=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            num_beams=1,
            repetition_penalty=1.05,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_k"] = top_k
            gen_kwargs["top_p"] = top_p
        else:
            gen_kwargs["temperature"] = 1.0
            gen_kwargs["top_p"] = 1.0
            gen_kwargs["top_k"] = 50

        out_ids = self.llm.generate(**gen_kwargs)
        if out_ids.shape[1] > max_new_tokens:
            gen = out_ids[0, -max_new_tokens:]
        else:
            gen = out_ids[0]
        n_new = int(gen.shape[0])
        text = self.tokenizer.decode(gen, skip_special_tokens=True).strip()
        if return_num_new_tokens:
            return text, n_new
        return text

    def _format_prompt(self, *, system_prompt: str, user_prompt: str) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ]
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return f"System: {system_prompt.strip()}\nUser: {user_prompt.strip()}\nAssistant:"

