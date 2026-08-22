"""Single source of truth for paths and hyperparameters used across the pipeline."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../flower_state_ai
REPO_ROOT = PROJECT_ROOT.parent  # C:/flowerProject
PHOTO_ROOT = REPO_ROOT / "photo"

DATA_DIR = PROJECT_ROOT / "data"
LABELS_DIR = DATA_DIR / "labels"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = PROCESSED_DIR / "cache_800"
AUGMENTED_DIR = PROCESSED_DIR / "augmented"

MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
TINY_DIR = MODELS_DIR / "tiny"
LOGS_DIR = MODELS_DIR / "logs"

# Stage 1 outputs
LABELS_MASTER_CSV = LABELS_DIR / "labels_master.csv"
LABEL_CONFLICTS_CSV = LABELS_DIR / "label_conflicts.csv"
SKIPPED_FILES_CSV = LABELS_DIR / "skipped_files.csv"

# Stage 1.5 outputs (Ollama vision label verification, species 001-015 only)
VERIFY_SPECIES_IDS = list(range(1, 16))
OLLAMA_VISION_MODEL = "llava"
OLLAMA_VISION_HOST = "http://localhost:11434"
VERIFICATION_MAX_RETRIES = 2
LABEL_VERIFICATION_CSV = LABELS_DIR / "label_verification.csv"
LABELS_MASTER_BACKUP_CSV = LABELS_DIR / "labels_master_pre_verification_backup.csv"
VERIFICATION_REPORT_UNCLEAR_KEPT_CSV = LABELS_DIR / "verification_unclear_kept_prior.csv"
VERIFICATION_REPORT_REVIEW_UNCLEAR_CSV = LABELS_DIR / "verification_review_unclear.csv"

# Stage 2 outputs
LABELS_CACHED_CSV = PROCESSED_DIR / "labels_cached.csv"
CACHE_ERRORS_CSV = PROCESSED_DIR / "cache_errors.csv"

# Stage 3 outputs
LABELS_SPLIT_CSV = PROCESSED_DIR / "labels_split.csv"
SPLIT_SUMMARY_CSV = PROCESSED_DIR / "split_summary.csv"

# Stage 4 outputs
AUGMENTATION_REPORT_CSV = PROCESSED_DIR / "augmentation_report.csv"
LABELS_TRAIN_FINAL_CSV = PROCESSED_DIR / "labels_train_final.csv"

# Stage 5/6 outputs
BEST_CHECKPOINT = CHECKPOINTS_DIR / "mobilenet_v2_best.pt"
FINAL_CHECKPOINT = CHECKPOINTS_DIR / "mobilenet_v2_final.pt"
TRAIN_LOG_CSV = LOGS_DIR / "train_log.csv"
TEST_REPORT_CSV = LOGS_DIR / "test_report.csv"
CONFUSION_MATRIX_CSV = LOGS_DIR / "confusion_matrix.csv"
MISCLASSIFIED_CSV = LOGS_DIR / "misclassified_samples.csv"

# Stage 7 outputs
TORCH_QUANT_PATH = TINY_DIR / "mobilenet_v2_dynamic_quant.pt"
ONNX_FP32_PATH = TINY_DIR / "mobilenet_v2_fp32.onnx"
ONNX_INT8_PATH = TINY_DIR / "mobilenet_v2_int8.onnx"
MODEL_SIZE_COMPARISON_CSV = TINY_DIR / "model_size_comparison.csv"

# tiny_cnn architecture: a genuinely tiny from-scratch CNN (tens of thousands of params,
# not just a quantized MobileNetV2), for microcontroller-class ("ultra tiny") deployment.
TINY_CNN_IMG_SIZE = 96
TINY_CNN_WIDTH_MULT = 1.0
TINY_CNN_BATCH_SIZE = 64
TINY_CNN_EPOCHS = 40
TINY_CNN_LR = 1e-3
TINY_CNN_WEIGHT_DECAY = 1e-4
TINY_CNN_MEAN = [0.5, 0.5, 0.5]
TINY_CNN_STD = [0.5, 0.5, 0.5]
TINY_CNN_BEST_CHECKPOINT = CHECKPOINTS_DIR / "tiny_cnn_best.pt"
TINY_CNN_FINAL_CHECKPOINT = CHECKPOINTS_DIR / "tiny_cnn_final.pt"
TINY_CNN_TRAIN_LOG_CSV = LOGS_DIR / "tiny_cnn_train_log.csv"
TINY_CNN_TEST_REPORT_CSV = LOGS_DIR / "tiny_cnn_test_report.csv"
TINY_CNN_CONFUSION_MATRIX_CSV = LOGS_DIR / "tiny_cnn_confusion_matrix.csv"
TINY_CNN_MISCLASSIFIED_CSV = LOGS_DIR / "tiny_cnn_misclassified_samples.csv"
TINY_CNN_TORCH_QUANT_PATH = TINY_DIR / "tiny_cnn_dynamic_quant.pt"
TINY_CNN_ONNX_FP32_PATH = TINY_DIR / "tiny_cnn_fp32.onnx"
TINY_CNN_ONNX_INT8_PATH = TINY_DIR / "tiny_cnn_int8.onnx"
TINY_CNN_MODEL_SIZE_COMPARISON_CSV = TINY_DIR / "tiny_cnn_model_size_comparison.csv"

# Dataset rules
VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
STATE_DIRS = {"closed", "open"}
CLASS_TO_IDX = {"closed": 0, "open": 1}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

# Preprocessing
MAX_CACHE_SIDE = 800
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
MIN_PER_CELL = 3
SPLIT_SEED = 42

# Augmentation
ROTATION_ANGLES = [-20, -10, 10, 20, 90, 180, 270]
# <1.0 = zoom in (center-crop then upscale), >1.0 = zoom out (shrink then reflect-pad).
# Empty by default - measured to slightly hurt held-out accuracy vs. rotation-only
# (MobileNetV2 test acc 69.4%->67.8%, tiny_cnn 63.7%->63.2%) on this dataset. The
# scale_and_crop() code path in augmentation.py is still available - set this list
# non-empty to re-enable it if you want to retest with different factors.
SCALE_FACTORS: list[float] = []
AUGMENTATION_MAX_MULTIPLIER = 4
AUGMENTATION_TARGET_RATIO = 1.0
AUGMENTATION_SEED = 42

# Training
BATCH_SIZE = 32
NUM_WORKERS = 4
PHASE1_EPOCHS = 5
PHASE1_LR = 1e-3
PHASE2_EPOCHS = 15
PHASE2_LR = 1e-4
UNFREEZE_LAST_N_BLOCKS = 5

# Agent
OLLAMA_MODEL = "llama3.2:latest"

# Per-architecture lookup so train.py/evaluate.py/export_tiny.py/inference.py/cli.py share
# one dispatch table instead of each hand-rolling their own if/else per architecture.
ARCH_REGISTRY = {
    "mobilenet_v2": {
        "img_size": IMG_SIZE,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "best_checkpoint": BEST_CHECKPOINT,
        "final_checkpoint": FINAL_CHECKPOINT,
        "train_log_csv": TRAIN_LOG_CSV,
        "test_report_csv": TEST_REPORT_CSV,
        "confusion_matrix_csv": CONFUSION_MATRIX_CSV,
        "misclassified_csv": MISCLASSIFIED_CSV,
        "torch_quant_path": TORCH_QUANT_PATH,
        "onnx_fp32_path": ONNX_FP32_PATH,
        "onnx_int8_path": ONNX_INT8_PATH,
        "model_size_comparison_csv": MODEL_SIZE_COMPARISON_CSV,
    },
    "tiny_cnn": {
        "img_size": TINY_CNN_IMG_SIZE,
        "mean": TINY_CNN_MEAN,
        "std": TINY_CNN_STD,
        "best_checkpoint": TINY_CNN_BEST_CHECKPOINT,
        "final_checkpoint": TINY_CNN_FINAL_CHECKPOINT,
        "train_log_csv": TINY_CNN_TRAIN_LOG_CSV,
        "test_report_csv": TINY_CNN_TEST_REPORT_CSV,
        "confusion_matrix_csv": TINY_CNN_CONFUSION_MATRIX_CSV,
        "misclassified_csv": TINY_CNN_MISCLASSIFIED_CSV,
        "torch_quant_path": TINY_CNN_TORCH_QUANT_PATH,
        "onnx_fp32_path": TINY_CNN_ONNX_FP32_PATH,
        "onnx_int8_path": TINY_CNN_ONNX_INT8_PATH,
        "model_size_comparison_csv": TINY_CNN_MODEL_SIZE_COMPARISON_CSV,
    },
}

for _d in (LABELS_DIR, PROCESSED_DIR, CACHE_DIR, AUGMENTED_DIR, CHECKPOINTS_DIR, TINY_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
