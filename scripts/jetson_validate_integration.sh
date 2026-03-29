#!/bin/bash
# scripts/jetson_validate_integration.sh
# Comprehensive validation suite for integrated VQA + VibeVoice pipeline

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VALIDATION_DIR="${REPO_ROOT}/reports/validation_$(date +%Y%m%d_%H%M%S)"
TEST_IMAGES_DIR="${REPO_ROOT}/data/eval/images"
LABELS_FILE="${REPO_ROOT}/data/eval/labels.json"

# Ensure Python path
export PYTHONPATH="${PYTHONPATH}:${REPO_ROOT}"

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $@"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $@"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $@"
}

pass_test() {
    echo -e "${GREEN}✓ PASS${NC}: $@"
}

fail_test() {
    echo -e "${RED}✗ FAIL${NC}: $@"
}

# Create validation directory
mkdir -p "$VALIDATION_DIR"
echo "Validation output directory: $VALIDATION_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Preflight Check
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 1: Jetson Preflight Check"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

log_info "Running preflight checks..."
if python3 "${REPO_ROOT}/scripts/jetson_preflight_check.py" > "$VALIDATION_DIR/preflight.log" 2>&1; then
    pass_test "Preflight checks passed"
    cat "$VALIDATION_DIR/preflight.log"
else
    fail_test "Preflight checks failed (see $VALIDATION_DIR/preflight.log)"
    cat "$VALIDATION_DIR/preflight.log"
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Single Image Smoke Test (Text-Only)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 2: Single Image Smoke Test (Text-Only, No TTS)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

SAMPLE_IMAGE="${TEST_IMAGES_DIR}/crosswalk/Crosswalk_1.png"
if [ ! -f "$SAMPLE_IMAGE" ]; then
    log_error "Sample image not found: $SAMPLE_IMAGE"
    exit 1
fi

log_info "Running single-image inference (text-only)..."
if python3 "${REPO_ROOT}/scripts/run_integrated.py" \
    --image-path "$SAMPLE_IMAGE" \
    --task crosswalk_signal \
    --compression 192 \
    --no-tts \
    --output-dir "$VALIDATION_DIR/smoke_test_textonly" \
    --verbose > "$VALIDATION_DIR/smoke_test_textonly.log" 2>&1; then
    
    # Check if report was generated
    if [ -f "$VALIDATION_DIR/smoke_test_textonly/report_"*.json ]; then
        pass_test "Single image text-only inference succeeded"
        cat "$VALIDATION_DIR/smoke_test_textonly.log" | tail -20
    else
        fail_test "No JSON report generated"
        cat "$VALIDATION_DIR/smoke_test_textonly.log"
    fi
else
    fail_test "Single image inference failed"
    cat "$VALIDATION_DIR/smoke_test_textonly.log"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Token Compression Unit Test
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 3: Token Compression Correctness"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

log_info "Running token compression tests..."
if python3 "${REPO_ROOT}/scripts/test_token_compression.py" > "$VALIDATION_DIR/token_compression.log" 2>&1; then
    pass_test "Token compression tests passed"
    cat "$VALIDATION_DIR/token_compression.log"
else
    fail_test "Token compression tests failed"
    cat "$VALIDATION_DIR/token_compression.log"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Full 16-Image Batch (Text-Only)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 4: Full Batch (16 Images, Text-Only)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ ! -f "$LABELS_FILE" ]; then
    log_error "Labels file not found: $LABELS_FILE"
    exit 1
fi

log_info "Running full 16-image batch (text-only)..."
if python3 "${REPO_ROOT}/scripts/run_integrated.py" \
    --labels "$LABELS_FILE" \
    --compression 192 \
    --no-tts \
    --output-dir "$VALIDATION_DIR/batch_textonly" > "$VALIDATION_DIR/batch_textonly.log" 2>&1; then
    
    # Count successful results
    REPORT_FILE=$(ls "$VALIDATION_DIR"/batch_textonly/report_*.json 2>/dev/null | head -1)
    if [ -f "$REPORT_FILE" ]; then
        SUCCESSFUL=$(python3 -c "
import json
with open('$REPORT_FILE') as f:
    data = json.load(f)
    count = sum(1 for r in data['results'] if r['error'] is None)
    print(count)
")
        pass_test "Batch inference completed ($SUCCESSFUL/16 images successful)"
        
        # Show sample latencies
        python3 -c "
import json
with open('$REPORT_FILE') as f:
    data = json.load(f)
    results = [r for r in data['results'] if r['error'] is None]
    if results:
        e2e_times = [r['e2e_total_ms'] for r in results]
        print(f'E2E latency (ms): min={min(e2e_times):.0f}, mean={sum(e2e_times)/len(e2e_times):.0f}, max={max(e2e_times):.0f}')
"
    else
        fail_test "No report generated"
        cat "$VALIDATION_DIR/batch_textonly.log" | tail -20
    fi
else
    fail_test "Batch inference failed"
    cat "$VALIDATION_DIR/batch_textonly.log" | tail -20
fi

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "VALIDATION SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Validation logs saved to: $VALIDATION_DIR"
echo ""
log_info "Validation suite completed successfully!"
echo "Next steps:"
echo "  1. Review logs in $VALIDATION_DIR"
echo "  2. Run full evaluation with TTS when VibeVoice is available:"
echo "     python scripts/run_integrated.py --labels data/eval/labels.json --tts microsoft/VibeVoice-Realtime-0.5B --voices-dir ~/vibevoice_test/voices"
echo ""
