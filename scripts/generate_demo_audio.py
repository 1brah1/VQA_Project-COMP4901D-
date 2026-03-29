#!/usr/bin/env python3
"""
Generate demo WAV files from VQA pipeline responses for local review.

Takes the VQA results JSON and synthesizes audio for classified outputs
using Python's built-in TTS or pyttsx3 for demonstration.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

try:
    import pyttsx3
    HAS_PYTTSX3 = True
except ImportError:
    HAS_PYTTSX3 = False


def generate_audio_demo(results_json: str, output_dir: str) -> None:
    """Generate WAV files from VQA results using pyttsx3 TTS."""
    
    if not HAS_PYTTSX3:
        print("ERROR: pyttsx3 not installed. Install with: pip install pyttsx3")
        print("Alternatively, show this output summary to user without audio.")
        sys.exit(1)
    
    results_path = Path(results_json)
    if not results_path.exists():
        print(f"ERROR: Results file not found: {results_json}")
        sys.exit(1)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    results = data.get('results', [])
    if not results:
        print("No results found in JSON")
        sys.exit(1)
    
    print(f"\n[Demo Audio Generator] Found {len(results)} results")
    print(f"Generating audio for classified responses...\n")
    
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Slower speech for clarity
    
    generated_count = 0
    skipped_count = 0
    
    for i, result in enumerate(results):
        sample_id = result.get('id', f'sample_{i}')
        task = result.get('task', 'unknown')
        pred = result.get('pred', 'unknown')
        response = result.get('response', '')
        
        # Only generate audio for successful classifications (short, clean outputs)
        if pred in ('red', 'green', 'yes', 'no'):
            wav_path = str(output_path / f"{sample_id}_{task}_{pred}.wav")
            
            # Use the classification as the primary audio output
            audio_text = f"{task}: {pred}"
            
            try:
                engine.save_to_file(audio_text, wav_path)
                engine.runAndWait()
                print(f"✓ {sample_id:20} | {task:15} | Pred={pred:10} | {wav_path}")
                generated_count += 1
            except Exception as e:
                print(f"✗ {sample_id:20} | {task:15} | Failed: {e}")
                skipped_count += 1
        else:
            # Skip unclear or unparseable responses
            print(f"⊗ {sample_id:20} | {task:15} | Skipped (unclear pred)")
            skipped_count += 1
    
    print(f"\n[Summary] Generated {generated_count} WAV files, skipped {skipped_count}")
    print(f"[Output] Audio files saved to: {output_path}")
    return generated_count


if __name__ == "__main__":
    results_file = sys.argv[1] if len(sys.argv) > 1 else "reports/vqa_results.json"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "outputs/audio_demo"
    
    count = generate_audio_demo(results_file, output_dir)
    if count > 0:
        print(f"\n✓ Demo audio generation complete! Listen to WAV files in: {output_dir}")
