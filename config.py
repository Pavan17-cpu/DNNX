"""Shared assignment configuration. Edit student identity fields before submission."""
from pathlib import Path
ROOT = Path(__file__).resolve().parent
STUDENT_NAME = "YOUR NAME"
STUDENT_ID = "YOUR STUDENT ID"
SECTION = "YOUR SECTION"
HIDDEN_SIZE = 32
MODEL_WIDTH = 64
NUM_HEADS = 4
NUM_LAYERS = 4
FF_WIDTH = 4 * MODEL_WIDTH
SEEDS = [42, 123, 2026]
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
HAR_EPOCHS = 12
ADDING_EPOCHS = 12
TRANSFORMER_EPOCHS = 10
LM_EPOCHS = 10
DEVICE = "cuda"  # automatically falls back to CPU
FIG1 = ROOT / "problem1" / "figures"
FIG2 = ROOT / "problem2" / "figures"
for p in (FIG1, FIG2): p.mkdir(parents=True, exist_ok=True)
