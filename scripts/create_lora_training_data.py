#!/usr/bin/env python3
"""Create augmented training data for LoRA fine-tuning from eval set."""

import json
import shutil
from pathlib import Path
from PIL import Image, ImageEnhance
import argparse

_ROOT = Path(__file__).resolve().parent.parent


def create_train_val_split():
    """Create data/train and data/val directories with augmented images."""
    
    eval_labels_path = _ROOT / "data" / "eval" / "labels.json"
    train_dir = _ROOT / "data" / "train"
    train_images_dir = train_dir / "images"
    val_dir = _ROOT / "data" / "val"
    val_images_dir = val_dir / "images"
    
    # Create directories
    for d in [train_images_dir, val_images_dir]:
        d.mkdir(parents=True, exist_ok=True)
        (d / "crosswalk").mkdir(exist_ok=True)
        (d / "stairs").mkdir(exist_ok=True)
        (d / "obstacles").mkdir(exist_ok=True)
    
    # Load eval labels
    with open(eval_labels_path, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
    
    eval_items = eval_data.get('items', [])
    print(f"[Data Prep] Loaded {len(eval_items)} eval images")
    
    train_items = []
    val_items = []
    
    # For each eval image, create augmented variants
    for idx, item in enumerate(eval_items):
        sample_id = item['id']
        task = item['task']
        labels = item.get('labels', {})
        image_path_str = item['path']
        
        # Load original image
        orig_image_path = _ROOT / image_path_str
        
        if not orig_image_path.exists():
            print(f"[WARN] Image not found: {orig_image_path}, skipping")
            continue
        
        try:
            img = Image.open(orig_image_path)
        except Exception as e:
            print(f"[WARN] Failed to load {orig_image_path}: {e}")
            continue
        
        # Determine split: ~70/30 train/val
        is_train = (idx % 10) < 7
        target_dir = train_images_dir if is_train else val_images_dir
        target_items = train_items if is_train else val_items
        
        task_subdir = task if task != "crosswalk_signal" else "crosswalk"
        
        # Save base image
        base_filename = f"{sample_id}.png"
        base_save_path = target_dir / task_subdir / base_filename
        img.save(base_save_path)
        
        new_item = {
            'id': sample_id,
            'path': f"data/{'train' if is_train else 'val'}/images/{task_subdir}/{base_filename}",
            'task': task,
            'labels': labels
        }
        target_items.append(new_item)
        
        # Create augmented variants: rotation + brightness
        augmentations = [
            ('rot5', lambda im: im.rotate(5, expand=False)),
            ('rot_m5', lambda im: im.rotate(-5, expand=False)),
            ('bright_hi', lambda im: ImageEnhance.Brightness(im).enhance(1.15)),
            ('bright_lo', lambda im: ImageEnhance.Brightness(im).enhance(0.85)),
        ]
        
        for aug_name, aug_fn in augmentations:
            try:
                aug_img = aug_fn(img.copy())
            except Exception as e:
                print(f"[WARN] Augmentation {aug_name} failed for {sample_id}: {e}")
                continue
            
            aug_filename = f"{sample_id}_{aug_name}.png"
            aug_save_path = target_dir / task_subdir / aug_filename
            aug_img.save(aug_save_path)
            
            aug_item = {
                'id': f"{sample_id}_{aug_name}",
                'path': f"data/{'train' if is_train else 'val'}/images/{task_subdir}/{aug_filename}",
                'task': task,
                'labels': labels
            }
            target_items.append(aug_item)
    
    # Write train/val labels
    train_labels = {'version': 1, 'items': train_items}
    val_labels = {'version': 1, 'items': val_items}
    
    train_labels_path = train_dir / "labels.json"
    val_labels_path = val_dir / "labels.json"
    
    with open(train_labels_path, 'w', encoding='utf-8') as f:
        json.dump(train_labels, f, indent=2)
    
    with open(val_labels_path, 'w', encoding='utf-8') as f:
        json.dump(val_labels, f, indent=2)
    
    print(f"[Data Prep] ✓ Created {len(train_items)} training samples")
    print(f"[Data Prep] ✓ Created {len(val_items)} validation samples")
    print(f"[Data Prep] ✓ Training labels: {train_labels_path}")
    print(f"[Data Prep] ✓ Validation labels: {val_labels_path}")
    
    return train_labels_path, val_labels_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    train_labels, val_labels = create_train_val_split()
    print("\n[Data Prep] ✅ Dataset preparation complete!")
