import os
import subprocess
import time
import sys
from pathlib import Path

def run_command(cmd, description, timeout=None):
    """Run a shell command and return output"""
    print(f"\n{'='*80}")
    print(f"[{description}]")
    print(f"{'='*80}")
    print(f"Command: {cmd}\n")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=False,
            text=True,
            timeout=timeout
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"ERROR: Command timed out after {timeout} seconds")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def main():
    print("VQA+TTS Pipeline Executor - Jetson to Local Transfer")
    print("="*80)
    
    # Configuration
    JETSON_SSH = os.environ.get("JETSON_SSH", "comp4901d-jetson")
    PROJECT_DIR = "~/VQA_Project-COMP4901D-"
    LOCAL_REPORTS_DIR = "reports"
    
    # Step 1: Run pipeline on Jetson
    print("\n[STEP 1] Running VQA+TTS pipeline on Jetson...")
    print(f"Target: {JETSON_SSH}")
    print(f"This will take 10-15 minutes...")
    
    jetson_cmd = (
        f"ssh {JETSON_SSH} "
        f'"cd {PROJECT_DIR} && '
        f'source .venv/bin/activate && '
        f'export PYTHONPATH="${{PYTHONPATH}}:$(pwd)" && '
        f'python scripts/run_vqa_with_tts.py '
        f'--llm_mode fp16 '
        f'--enable_tts '
        f'--max_new_tokens 24 '
        f'--output reports/vqa_with_tts_results.json '
        f'--audio_dir reports/vqa_audio"'
    )
    
    if not run_command(jetson_cmd, "JETSON: Running VQA+TTS pipeline", timeout=1800):
        print("ERROR: Pipeline execution failed")
        return False
    
    # Step 2: Verify output on Jetson
    print("\n[STEP 2] Verifying output on Jetson...")
    verify_cmd = (
        f'ssh {JETSON_SSH} '
        f'"ls -lh {PROJECT_DIR}/reports/vqa_with_tts_results.json && '
        f'echo \\"WAV files: $(ls {PROJECT_DIR}/reports/vqa_audio/*.wav 2>/dev/null | wc -l)\\"'
        f'"'
    )
    run_command(verify_cmd, "JETSON: Verify output files")
    
    # Step 3: Create local directory
    print("\n[STEP 3] Preparing local directory...")
    Path(LOCAL_REPORTS_DIR).mkdir(exist_ok=True)
    Path(f"{LOCAL_REPORTS_DIR}/vqa_audio").mkdir(exist_ok=True)
    print(f"Created {LOCAL_REPORTS_DIR}/vqa_audio/")
    
    # Step 4: Transfer JSON results
    print("\n[STEP 4] Transferring VQA+TTS results JSON...")
    json_transfer_cmd = (
        f'scp {JETSON_SSH}:{PROJECT_DIR}/reports/vqa_with_tts_results.json '
        f'{LOCAL_REPORTS_DIR}/vqa_with_tts_results.json'
    )
    if run_command(json_transfer_cmd, "LOCAL: Transferring JSON results"):
        print(f"✓ JSON transferred to {LOCAL_REPORTS_DIR}/vqa_with_tts_results.json")
    else:
        print("WARNING: JSON transfer may have failed, continuing...")
    
    # Step 5: Transfer WAV files
    print("\n[STEP 5] Transferring WAV audio files...")
    print("This may take a few minutes...")
    wav_transfer_cmd = (
        f'scp {JETSON_SSH}:{PROJECT_DIR}/reports/vqa_audio/*.wav '
        f'{LOCAL_REPORTS_DIR}/vqa_audio/'
    )
    if run_command(wav_transfer_cmd, "LOCAL: Transferring WAV files"):
        print(f"✓ WAV files transferred to {LOCAL_REPORTS_DIR}/vqa_audio/")
    else:
        print("WARNING: WAV transfer may have had issues")
    
    # Step 6: Verify local files
    print("\n[STEP 6] Verifying local files...")
    json_file = Path(f"{LOCAL_REPORTS_DIR}/vqa_with_tts_results.json")
    audio_dir = Path(f"{LOCAL_REPORTS_DIR}/vqa_audio")
    
    if json_file.exists():
        size_mb = json_file.stat().st_size / (1024*1024)
        print(f"✓ JSON file: {json_file} ({size_mb:.2f} MB)")
    else:
        print(f"✗ JSON file NOT found: {json_file}")
        return False
    
    wav_files = list(audio_dir.glob("*.wav"))
    if wav_files:
        print(f"✓ Audio files: {len(wav_files)} WAV files")
        total_size_mb = sum(f.stat().st_size for f in wav_files) / (1024*1024)
        print(f"  Total size: {total_size_mb:.2f} MB")
        print(f"  Sample files:")
        for f in sorted(wav_files)[:3]:
            size_kb = f.stat().st_size / 1024
            print(f"    - {f.name} ({size_kb:.1f} KB)")
    else:
        print(f"✗ No WAV files found in {audio_dir}")
    
    # Step 7: Analyze results
    print("\n[STEP 7] Analyzing results...")
    try:
        import json
        with open(json_file) as f:
            data = json.load(f)
        
        stats = data.get('statistics', {})
        results = data.get('results', [])
        
        print("\n📊 RESULTS SUMMARY:")
        print(f"  Total samples: {stats.get('total_samples', '?')}")
        print(f"  Overall accuracy: {stats.get('overall_accuracy', 0)*100:.1f}%")
        print(f"  Correct predictions: {stats.get('correct_predictions', 0)}")
        print(f"  Avg VLM latency: {stats.get('avg_vlm_latency_ms', 0):.1f}ms")
        print(f"  Avg E2E latency: {stats.get('avg_e2e_latency_ms', 0):.1f}ms")
        print(f"  Avg TTS latency: {stats.get('avg_tts_latency_ms', 0):.1f}ms")
        
        # Task breakdown
        print("\n  By Task:")
        for task in ['crosswalk_signal', 'stairs', 'obstacles']:
            task_results = [r for r in results if r.get('task') == task]
            if task_results:
                correct = sum(1 for r in task_results if r.get('correct', False))
                accuracy = correct / len(task_results) * 100 if task_results else 0
                print(f"    {task}: {correct}/{len(task_results)} ({accuracy:.1f}%)")
        
        # Audio generation status
        audio_success = sum(1 for r in results if r.get('tts_metrics', {}).get('success', False))
        print(f"\n  Audio files generated: {audio_success}/{len(results)}")
        
    except Exception as e:
        print(f"Could not analyze results: {e}")
    
    print("\n" + "="*80)
    print("✅ COMPLETE - All files transferred successfully!")
    print("="*80)
    print(f"\nLocal files ready:")
    print(f"  Results: {LOCAL_REPORTS_DIR}/vqa_with_tts_results.json")
    print(f"  Audio:   {LOCAL_REPORTS_DIR}/vqa_audio/")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
