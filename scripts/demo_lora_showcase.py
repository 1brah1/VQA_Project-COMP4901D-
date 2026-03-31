#!/usr/bin/env python3
"""Interactive demo: baseline vs LoRA side-by-side comparison."""

import sys
from pathlib import Path
import json

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.vision.siglip_encoder import SiglipPatchEncoder
from src.vision.token_compression import compress_27x27_tokens
from src.prompts.load_prompt import load_system_prompt

_TASK_PROMPTS = {
    "crosswalk_signal": "Is the crosswalk walk signal red or green? Respond with one word only.",
    "stairs": "Are there stairs visible? Respond with one word only.",
    "obstacles": "Is there an obstacle ahead? Respond with one word only.",
}


class VQADemo:
    def __init__(self,lora_adapter_path=None, compression=192):
        print("[Demo] Loading models...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.compression = compression
        
        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            'Qwen/Qwen2.5-1.5B-Instruct',
            torch_dtype=torch.float16,
            device_map='auto'
        )
        
        # Load LoRA model if provided
        self.lora_model = None
        if lora_adapter_path and Path(lora_adapter_path).exists():
            print(f"[Demo] Loading LoRA from {lora_adapter_path}...")
            self.lora_model = PeftModel.from_pretrained(self.base_model, str(lora_adapter_path))
        
        self.tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
        self.encoder = SiglipPatchEncoder.from_pretrained(
            'google/siglip-base-patch16-384',
            device=self.device,
            dtype=torch.float32
        )
        self.system_prompt = load_system_prompt()
        self.base_model.eval()
        if self.lora_model:
            self.lora_model.eval()
        
        print("[Demo] ✓ Models ready")
    
    def infer(self, image_path, task):
        """Run inference on both baseline and LoRA."""
        image_path = Path(image_path)
        if not image_path.exists():
            print(f"[ERROR] Image not found: {image_path}")
            return None
        
        # Encode image
        img = Image.open(image_path).convert('RGB')
        tokens = self.encoder.encode(img)  # (1, H, D)
        tokens_compressed = compress_27x27_tokens(tokens, target_tokens=self.compression)
        
        # Build prompt
        task_prompt = _TASK_PROMPTS.get(task, "Answer the question.")
        messages = [
            {"role": "system", "content": self.system_prompt.strip()},
            {"role": "user", "content": task_prompt},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = self.tokenizer(prompt, return_tensors='pt')['input_ids'].to(self.device)
        
        results = {'image': str(image_path), 'task': task}
        
        # Baseline
        with torch.no_grad():
            outputs_base = self.base_model.generate(
                input_ids, max_new_tokens=4, do_sample=False, temperature=1.0, top_p=1.0
            )
            text_base = self.tokenizer.decode(outputs_base[0], skip_special_tokens=True)
            pred_base = text_base[len(prompt):].strip().split()[0].lower()
            results['baseline_full'] = text_base[len(prompt):].strip()
            results['baseline_pred'] = pred_base
        
        # LoRA
        if self.lora_model:
            with torch.no_grad():
                outputs_lora = self.lora_model.generate(
                    input_ids, max_new_tokens=4, do_sample=False, temperature=1.0, top_p=1.0
                )
                text_lora = self.tokenizer.decode(outputs_lora[0], skip_special_tokens=True)
                pred_lora = text_lora[len(prompt):].strip().split()[0].lower()
                results['lora_full'] = text_lora[len(prompt):].strip()
                results['lora_pred'] = pred_lora
        
        return results
    
    def demo_batch(self, image_dir, task_filter=None, max_samples=5):
        """Run demo on samples from a directory."""
        image_dir = Path(image_dir)
        images = sorted(image_dir.glob('*.png')) + sorted(image_dir.glob('*.jpg'))
        images = images[:max_samples]
        
        results = []
        for img_path in images:
            # Infer task from parent directory
            task = img_path.parent.name
            if 'crosswalk' in task.lower():
                task = 'crosswalk_signal'
            elif 'stair' in task.lower():
                task = 'stairs'
            elif 'obstacle' in task.lower():
                task = 'obstacles'
            
            if task_filter and task != task_filter:
                continue
            
            result = self.infer(img_path, task)
            if result:
                results.append(result)
                
                # Print result
                print(f"\n📷 {img_path.name} (task={task})")
                print(f"  Baseline: pred={result.get('baseline_pred', '?'):<10} | {result.get('baseline_full', '')[:60]}")
                if self.lora_model:
                    print(f"  LoRA:     pred={result.get('lora_pred', '?'):<10} | {result.get('lora_full', '')[:60]}")
                    if result.get('baseline_pred') != result.get('lora_pred'):
                        print(f"  → CHANGE: {result['baseline_pred']} → {result['lora_pred']}")
        
        return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', type=Path, default=_ROOT / 'data' / 'eval' / 'images')
    parser.add_argument('--lora-adapter', type=Path, default=_ROOT / 'models' / 'lora_accessibility_vqa')
    parser.add_argument('--task', default=None, help='Filter by task type')
    parser.add_argument('--samples', type=int, default=5, help='Max samples to demo')
    parser.add_argument('--compression', type=int, default=192)
    
    args = parser.parse_args()
    
    demo = VQADemo(args.lora_adapter, compression=args.compression)
    
    # Run on random subset
    results = demo.demo_batch(
        args.image_dir / 'crosswalk',
        task_filter=args.task,
        max_samples=args.samples
    )
    results += demo.demo_batch(
        args.image_dir / 'stairs',
        task_filter=args.task,
        max_samples=args.samples
    )
    results += demo.demo_batch(
        args.image_dir / 'obstacles',
        task_filter=args.task,
        max_samples=args.samples
    )
    
    print(f"\n✅ Demo complete ({len(results)} samples)")
