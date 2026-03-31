#!/usr/bin/env python3
"""LoRA + Image Projection fine-tuning for accessibility VQA on Qwen2.5-0.5B/1.5B."""

import json
import sys
from pathlib import Path
import argparse
import random

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.vision.siglip_encoder import SiglipPatchEncoder
from src.vision.token_compression import compress_27x27_tokens
from src.prompts.load_prompt import load_system_prompt
from src.vlm.model import SimplePrefixVLM

_TASK_PROMPTS = {
    "crosswalk_signal": (
        "Is the crosswalk walk signal red or green? "
        "Start your response with exactly one word: red|green|unknown. "
        "Then give one short action clause."
    ),
    "stairs": (
        "Are there stairs or steps visible? "
        "Start your response with exactly one word: yes|no|unknown. "
        "Then give one short action clause."
    ),
    "obstacles": (
        "Is there an obstacle ahead? "
        "Start your response with exactly one word: yes|no|unknown. "
        "Then give one short action clause."
    ),
}

def _set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _label_for_item(task: str, labels: dict) -> str:
    if task == "crosswalk_signal":
        return str(labels.get("walk_signal", "unknown")).lower()
    if task == "stairs":
        return str(labels.get("stairs_present", "unknown")).lower()
    if task == "obstacles":
        return str(labels.get("obstacle_present", "unknown")).lower()
    return "unknown"

def _load_items(labels_path: Path):
    with open(labels_path, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    items = []
    for row in obj.get('items', []):
        sample_id = str(row['id'])
        task = str(row['task'])
        label = _label_for_item(task, row.get('labels', {}))
        image_path = (_ROOT / row['path']).resolve()
        if image_path.exists():
            items.append({
                'id': sample_id,
                'task': task,
                'label': label,
                'image_path': image_path
            })
    return items

class AccessibilityVQADataset(Dataset):
    def __init__(self, labels_path: Path, model_name: str, max_samples: int = None):
        self.model_name = model_name
        self.items = _load_items(labels_path)
        if max_samples and len(self.items) > max_samples:
            self.items = self.items[:max_samples]
        self.encoder = None
        self.tokenizer = None
        self.system_prompt = load_system_prompt()
        print(f"[Dataset] Loaded {len(self.items)} samples from {labels_path}")
    
    def _ensure_encoder(self):
        if self.encoder is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.encoder = SiglipPatchEncoder.from_pretrained('google/siglip-base-patch16-384', device=device, dtype=torch.float32)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
    def __len__(self): return len(self.items)
    
    def __getitem__(self, idx):
        self._ensure_encoder()
        item = self.items[idx]
        image_path, task, label = item['image_path'], item['task'], item['label']
        try:
            img = Image.open(image_path).convert('RGB')
            img_tokens = self.encoder.encode(img)
            img_tokens_compressed = compress_27x27_tokens(img_tokens, target_tokens=192)
            img_tokens_tensor = img_tokens_compressed.squeeze(0).detach().to(device="cpu", dtype=torch.float32)
        except Exception as e:
            print(f"[WARN] Failed to encode {image_path}: {e}")
            img_tokens_tensor = torch.zeros((192, 768), dtype=torch.float32)
        
        task_prompt = _TASK_PROMPTS.get(task, "Answer the question.")
        messages = [
            {"role": "system", "content": self.system_prompt.strip()},
            {"role": "user", "content": task_prompt},
        ]
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # dynamic length tokenization
        prompt_ids = self.tokenizer(prompt_text, return_tensors='pt').input_ids.squeeze(0)
        label_ids = self.tokenizer(f" {label}", return_tensors='pt').input_ids.squeeze(0)
        
        return {
            'image_tokens': img_tokens_tensor,
            'prompt_ids': prompt_ids,
            'label_ids': label_ids,
        }

def collate_fn(batch):
    return batch  # handle batch size 1 correctly, or pad

def train_lora(train_labels_path: Path, val_labels_path: Path, output_dir: Path, 
               num_epochs: int = 2, learning_rate: float = 5e-4, model_name: str = 'Qwen/Qwen2.5-0.5B-Instruct'):
    _set_seed(42)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"[Training] Loading SimplePrefixVLM ({model_name})...")
    vlm = SimplePrefixVLM.from_pretrained(model_name, device=device, dtype=torch.float16, image_token_dim=768)
    
    lora_config = LoraConfig(
        r=8, lora_alpha=16, target_modules=['q_proj', 'v_proj'],
        lora_dropout=0.05, bias='none', task_type=TaskType.CAUSAL_LM
    )
    vlm.llm = get_peft_model(vlm.llm, lora_config)
    vlm.llm.print_trainable_parameters()
    
    # Train both image projection and LoRA adapter
    vlm.image_proj.to(device)
    vlm.image_proj.train()
    vlm.image_proj.weight.requires_grad = True
    
    train_dataset = AccessibilityVQADataset(train_labels_path, model_name=model_name)
    val_dataset = AccessibilityVQADataset(val_labels_path, model_name=model_name)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    optimizer = torch.optim.AdamW(list(vlm.llm.parameters()) + list(vlm.image_proj.parameters()), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_dataset))
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        vlm.llm.train()
        vlm.image_proj.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            item = batch[0]
            img_tok = item['image_tokens'].unsqueeze(0).to(device) # (1, 192, 768)
            prompt_ids = item['prompt_ids'].unsqueeze(0).to(device)
            label_ids = item['label_ids'].unsqueeze(0).to(device)
            
            optimizer.zero_grad()
            
            text_embeds = vlm.llm.get_input_embeddings()(prompt_ids)
            img_prefix = vlm.image_proj(img_tok).to(text_embeds.dtype)
            inputs_embeds = torch.cat([img_prefix, text_embeds], dim=1)
            
            attention_mask = torch.ones(1, inputs_embeds.shape[1] + label_ids.shape[1], dtype=torch.long, device=device)
            target_ids = label_ids
            labels = torch.full((1, inputs_embeds.shape[1] + target_ids.shape[1]), -100, dtype=torch.long, device=device)
            
            image_len = img_prefix.shape[1]
            prompt_len = prompt_ids.shape[1]
            labels[0, image_len + prompt_len : image_len + prompt_len + target_ids.shape[1]] = target_ids[0]
            
            target_embeds = vlm.llm.get_input_embeddings()(target_ids)
            full_inputs = torch.cat([inputs_embeds, target_embeds], dim=1)
            
            outputs = vlm.llm(inputs_embeds=full_inputs, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(list(vlm.llm.parameters()) + list(vlm.image_proj.parameters()), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            
            if (batch_idx + 1) % max(1, len(train_loader) // 4) == 0:
                print(f"[Training] Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {train_loss/(batch_idx + 1):.4f}")
        
        # omitted val for brevity, just evaluate train loss...
        print(f"[Training] OK Epoch {epoch+1} complete: train_loss={train_loss/len(train_loader):.4f}")
    
    vlm.llm.save_pretrained(output_dir)
    torch.save(vlm.image_proj.state_dict(), str(output_dir / "image_proj.pt"))
    print(f"[Training] OK LoRA adapter and image_proj saved to {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-labels', type=Path, default=_ROOT / 'data' / 'train' / 'labels.json')
    parser.add_argument('--val-labels', type=Path, default=_ROOT / 'data' / 'val' / 'labels.json')
    parser.add_argument('--output-dir', type=Path, default=_ROOT / 'models' / 'lora_accessibility_vqa')
    parser.add_argument('--epochs', type=int, default=3) # bumped to 3
    parser.add_argument('--lr', type=float, default=5e-4) # learning rate
    parser.add_argument('--model-name', type=str, default='Qwen/Qwen2.5-0.5B-Instruct')
    args = parser.parse_args()
    
    train_lora(args.train_labels, args.val_labels, args.output_dir, num_epochs=args.epochs, learning_rate=args.lr, model_name=args.model_name)
