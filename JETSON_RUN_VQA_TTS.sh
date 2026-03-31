# COPY THIS AND RUN ON JETSON (via SSH)
# The script will process all 28 images and generate audio responses
# Expected runtime: 10-15 minutes

cd ~/VQA_Project-COMP4901D-
source .venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python scripts/run_vqa_with_tts.py \
  --llm_mode fp16 \
  --enable_tts \
  --max_new_tokens 24 \
  --output reports/vqa_with_tts_results.json \
  --audio_dir reports/vqa_audio

# After it completes, you'll see:
# [DONE] Results saved to reports/vqa_with_tts_results.json
# [SUMMARY] Accuracy: X/28 (X%)
# [SUMMARY] Audio files: 28/28 generated

# Then verify:
echo "=== RESULTS ===" 
ls -lh ~/VQA_Project-COMP4901D-/reports/vqa_with_tts_results.json
echo "Total audio files: $(ls ~/VQA_Project-COMP4901D-/reports/vqa_audio/*.wav | wc -l)"
