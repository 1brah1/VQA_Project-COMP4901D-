from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union

import torch
from PIL import Image
from transformers import AutoImageProcessor, SiglipVisionModel


@dataclass
class SiglipPatchEncoder:
    """
    Encodes an image into SigLIP patch tokens.

    Expected output for 384px + 27x27 grid: (B, 729, D)
    Some checkpoints include a CLS token -> (B, 730, D); we drop it.
    """

    processor: any
    model: torch.nn.Module
    device: str
    dtype: torch.dtype

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "google/siglip-base-patch16-384",
        *,
        device: Union[Literal["cpu", "cuda"], str] = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "SiglipPatchEncoder":
        # Using SiglipVisionModel directly is cleaner and uses less RAM 
        # than loading the full Text+Vision model.
        model = SiglipVisionModel.from_pretrained(
            model_name_or_path, 
            torch_dtype=dtype if device == "cuda" else torch.float32
        )
        processor = AutoImageProcessor.from_pretrained(model_name_or_path)

        model.eval()
        model.to(device=device)

        return cls(processor=processor, model=model, device=str(device), dtype=dtype)

    @torch.inference_mode()
    def encode(self, image: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=self.device)
        
        if self.device == "cuda":
            pixel_values = pixel_values.to(dtype=self.dtype)

        # Now calling the vision model directly
        outputs = self.model(pixel_values=pixel_values)

        # SiglipVisionModel outputs have last_hidden_state
        if not hasattr(outputs, "last_hidden_state"):
            raise RuntimeError("Vision model output missing last_hidden_state.")

        x = outputs.last_hidden_state  # (B, seq, D)
        
        if x.ndim != 3:
            raise RuntimeError(f"Expected (B, seq, D) but got shape {tuple(x.shape)}")

        seq_len = x.shape[1]
        
        # SigLIP 384-16 usually results in 576 or 729 tokens depending on config.
        # We drop the CLS token if present (usually at index 0).
        if seq_len == 730 or seq_len == 577:
            x = x[:, 1:, :]  # drop CLS
            seq_len = x.shape[1]

        # Known SigLIP outputs we support in compression.
        if seq_len not in (576, 729):
            raise RuntimeError(
                f"Unsupported number of patch tokens: {seq_len}. "
                "Expected one of {576, 729} after optional CLS removal."
            )

        return x