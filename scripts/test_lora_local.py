#!/usr/bin/env python3
"""Local validation of LoRA-tuned VLM on eval set."""

import json
import sys
from pathlib import Path
from datetime import datetime

import torch
from PIL import Image
from transformers import AutoTokenizer
from peft import PeftModel

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.vision.siglip_encoder import SiglipPatchEncoder
from src.vision.token_compression import compress_27x27_tokens
from src.prompts.load_prompt import load_system_prompt
from src.vlm.model import SimplePrefixVLM

def _label_for_item(task: str, labels: dict) -> str:
    if task == "crosswalk_signal":
        return str(labels.get("walk_signal", "unknown")).lower()
    if task in ["stairs", "obstacles"]:
        key = f"{task[:-1]}_present" if task == "obstacles" else "stairs_present"
        return str(labels.get(key, "unknown")).lower()
    return "unknown"

def validate_lora(
    eval_labels_path: Path = _ROOT / 'data' / 'eval' / 'labels.json',
    lora_adapter_dir: Path = _ROOT / 'models' / 'lora_accessibility_vqa',
    output_report: Path = _ROOT / 'reports' / 'lora_validation_local.json',
    model_name: str = 'Qwen/Qwen2.5-0.5B-Instruct'
):
    output_report.parent.mkdir(parents=True, exist_ok=True)
    
    with open(eval_labels_path, 'r', encoding='utf-8') as f:
        items = json.load(f).get('items', [])
    print(f"[Validation] Loaded {len(items)} eval samples")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Validation] Loading base VLM ({model_name})...")
    
    vlm = SimplePrefixVLM.from_pretrained(model_name, device=device, dtype=torch.float16, image_token_dim=768)
    
    adapter_path = lora_adapter_dir
    if adapter_path.exists():
        print(f"[Validation] Loading LoRA adapter from {adapter_path}...")
        vlm.llm = PeftModel.from_pretrained(vlm.llm, str(adapter_path))
        proj_path = adapter_path / "image_proj.pt"
        if proj_path.exists():
            print("[Validation] Loading trained image projection weights...")
            vlm.image_proj.load_state_dict(torch.load(proj_path, map_location=device))
        else:
            print("[WARN] image_proj.pt not found! Model will use random projection.")
    else:
        print("[WARN] LoRA adapter not found. Validating baseline.")
    
    vlm.llm.eval()
    vlm.image_proj.eval()
    
    encoder = SiglipPatchEncoder.from_pretrained('google/siglip-base-patch16-384', device=device, dtype=torch.float32)
    system_prompt = load_system_prompt()
    
    metrics = {
        'n_items': len(items), 'n_correct': 0, 'n_gt_known': 0,
        'accuracy': 0.0, 'unknown_rate': 0.0, 'bang_rate': 0.0,
        'per_task': {}, 'timings': []
    }
    results = []
    print(f"[Validation] Running inference on {len(items)} samples...")
    
    _TASK_PROMPTS = {
        "crosswalk_signal": "Is the crosswalk walk signal red or green? Start your response with exactly one word: red|green|unknown.",
        "stairs": "Are there stairs or steps visible? Start your response with exactly one word: yes|no|unknown.",
        "obstacles": "Is there an obstacle ahead? Start your response with exactly one word: yes|no|unknown.",
    }
    
    with torch.no_grad():
        for idx, item in enumerate(items):
            task = item['task']
            gt_label = _label_for_item(task, item.get('labels', {}))
            
            img = Image.open(_ROOT / item['path']).convert('RGB')
            img_tokens = encoder.encode(img)
            img_tokens_compressed = compress_27x27_tokens(img_tokens, target_tokens=192)
            
            task_prompt = _TASK_PROMPTS.get(task, "Answer the question.")
            
            response_text = vlm.generate(
                image_tokens=img_tokens_compressed,
                system_prompt=system_prompt,
                user_prompt=task_prompt,
                max_new_tokens=4,
                temperature=1.0,
                top_p=1.0,
                do_sample=False
            )
            
            first_word = response_text.split()[0].lower() if response_text else "unknown"
            pred_label = first_word if first_word in ['red', 'green', 'yes', 'no', 'unknown'] else 'unknown'
            
            is_correct = (pred_label == gt_label.lower())
            if gt_label.lower() != 'unknown':
                metrics['n_gt_known'] += 1
                if is_correct: metrics['n_correct'] += 1
            
            if task not in metrics['per_task']: metrics['per_task'][task] = {'n': 0, 'correct': 0}
            metrics['per_task'][task]['n'] += 1
            if is_correct and gt_label.lower() != 'unknown':
                metrics['per_task'][task]['correct'] += 1
            
            results.append({
                'sample_id': item['id'], 'task': task,
                'gt_label': gt_label, 'pred_label': pred_label,
                'response': response_text[:50]
            })
            print(f"[{idx+1}/{len(items)}] {item['id']}: gt={gt_label}, pred={pred_label}, {'✓' if is_correct else '✗'}")
    
    metrics['accuracy'] = metrics['n_correct'] / metrics['n_gt_known'] if metrics['n_gt_known'] > 0 else 0.0
    metrics['unknown_rate'] = sum(1 for r in results if r['pred_label'] == 'unknown') / len(results)
    metrics['bang_rate'] = sum(1 for r in results if '!!!!' in r['response']) / len(results)
    
    for _, counts in metrics['per_task'].items():
        counts['accuracy'] = counts['correct'] / counts['n'] if counts['n'] > 0 else 0.0
    
    with open(output_report, 'w') as f:
        json.dump({'metrics': metrics, 'results': results}, f, indent=2)
    
    print(f"\n[Validation] Accuracy: {metrics['accuracy']:.4f}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default='Qwen/Qwen2.5-0.5B-Instruct')
    args = parser.parse_args()
    validate_lora(model_name=args.model_name)
