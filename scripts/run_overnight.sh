#!/bin/bash
# ARIA Overnight Training Script
#
# Runs the brain and trainer together for autonomous learning.
# Use: ./scripts/run_overnight.sh
#
# Press Ctrl+C to stop both processes gracefully.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "═══════════════════════════════════════════════════════════"
echo "           ARIA Overnight Training Session"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Starting at: $(date)"
echo "Project dir: $PROJECT_DIR"
echo ""

# Create data directory for logs
mkdir -p data

# Build release version for performance
echo "📦 Building ARIA brain (release)..."
cargo build --release --package aria-brain

echo ""
echo "🧠 Starting ARIA brain..."
echo "   Logs will be written to data/brain.log"
echo ""

# Start brain in background
ARIA_BACKEND=gpu cargo run --release --package aria-brain > data/brain.log 2>&1 &
BRAIN_PID=$!

echo "   Brain PID: $BRAIN_PID"

# Wait for brain to start
echo "   Waiting for brain to initialize..."
sleep 10

# Check if brain is still running
if ! kill -0 $BRAIN_PID 2>/dev/null; then
    echo "❌ Brain failed to start! Check data/brain.log"
    exit 1
fi

echo "   ✅ Brain is running!"
echo ""

# Start trainer
echo "🎓 Starting autonomous trainer..."
echo "   Press Ctrl+C to stop training"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo ""

# Trap to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping ARIA..."
    kill $BRAIN_PID 2>/dev/null || true
    echo "   Brain stopped."
    echo ""
    echo "📊 Training session ended at: $(date)"
    echo "   Check data/trainer_log.txt for training history"
    echo "   Check data/brain.log for brain activity"
}

trap cleanup EXIT

# Run trainer (this will block until Ctrl+C)
# Use project venv if available, otherwise system python
if [ -f ".venv/bin/python" ]; then
    .venv/bin/python scripts/autonomous_trainer.py
else
    python3 scripts/autonomous_trainer.py
fi

# Wait for brain to finish
wait $BRAIN_PID
