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

# PaliGemma augmentation: unlike augment_closed_class above (which balances open:closed
# *within* a species), this boosts species with few total train images (Adenanthos n=15,
# Ammi_visnaga n=76, Astilbe n=88, Celosia n=101, Achillea n=128, Clarkia n=183, vs. a ~330
# median) - a species-identification task needs enough examples per species, not just a
# balanced state ratio. Rotation reuses ROTATION_ANGLES above; flip and color-jitter are
# PaliGemma-specific op types (see augmentation.py's FLIP/COLOR_JITTER handling) so this
# pulls from 3 distinct augmentation families rather than one, per the user's request for
# more variety than the classifier's rotation-only approach.
PALIGEMMA_AUG_TARGET_COUNT = 300
PALIGEMMA_AUG_MAX_MULTIPLIER = 6
PALIGEMMA_AUG_COLOR_JITTER_FACTORS = [0.8, 1.2]
# Species whose species-name accuracy (from the most recent evaluate run) falls below 0.80
# also get boosted, not just species with few raw images - a species with plenty of photos
# but poor recognition still benefits from more augmented variety. Severity scales with how
# bad the accuracy is: worse-performing species get both a bigger count multiplier AND a
# wider variety of augmentation op types (more distinct "views" of the same source photos),
# rather than just more copies of the same narrow transform.
PALIGEMMA_AUG_TIERS = [
    # (accuracy upper bound, exclusive; count multiplier; op-pool name)
    (0.40, 2.5, "heavy"),   # rotate (all angles) + flip + color_jitter (all factors)
    (0.60, 1.8, "medium"),  # rotate (4 small angles) + flip + color_jitter (1 factor)
    (0.80, 1.3, "light"),   # rotate (2 angles) + flip
]

# Training
BATCH_SIZE = 32
NUM_WORKERS = 4
PHASE1_EPOCHS = 5
PHASE1_LR = 1e-3
PHASE2_EPOCHS = 15
PHASE2_LR = 1e-4
UNFREEZE_LAST_N_BLOCKS = 5

# PaliGemma-3B fine-tuning: a joint species-name + open/closed captioning model over species
# 1-23, complementary to the ARCH_REGISTRY classifiers above (which predict only state, given
# species is already known from the folder). Kept as a flat sibling block like TINY_CNN_* above
# rather than folded into ARCH_REGISTRY - that registry's contract (img_size/mean/std for a
# hand-rolled Normalize, one .pt checkpoint, a 2-class confusion matrix) doesn't fit a
# generative model whose normalization lives inside its own processor and whose "checkpoint"
# is a PEFT adapter directory.
PALIGEMMA_MODEL_ID = "google/paligemma2-3b-pt-224"
PALIGEMMA_SPECIES_MIN_ID = 1
PALIGEMMA_SPECIES_MAX_ID = 24
PALIGEMMA_LORA_R = 8
PALIGEMMA_LORA_ALPHA = 16
PALIGEMMA_LORA_TARGET_REGEX = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
PALIGEMMA_BATCH_SIZE = 1
PALIGEMMA_GRAD_ACCUM_STEPS = 16
PALIGEMMA_EPOCHS = 3
PALIGEMMA_LR = 2e-4
PALIGEMMA_PROMPT = "answer en what flower species and bloom state is shown in this photo?"
# state_only mode: species name goes IN the prompt (as the classifiers already get it, e.g.
# from folder structure) - target is just "open"/"closed", not "<species> <state>". Tests
# whether PaliGemma's world knowledge of a *named* species' bloom morphology beats the
# classifiers at the one thing they're actually asked to do (state only), instead of also
# making it guess species, which is a strictly harder task than either classifier attempts.
PALIGEMMA_STATE_ONLY_PROMPT_TEMPLATE = "answer en is this {species} flower open or closed?"
PALIGEMMA_ADAPTER_DIR = CHECKPOINTS_DIR / "paligemma_lora_adapter"
PALIGEMMA_STATE_ONLY_ADAPTER_DIR = CHECKPOINTS_DIR / "paligemma_lora_adapter_state_only"
PALIGEMMA_AUGMENTED_DIR = PROCESSED_DIR / "paligemma_augmented"
PALIGEMMA_LABELS_TRAIN_FINAL_CSV = PROCESSED_DIR / "paligemma_labels_train_final.csv"
PALIGEMMA_AUGMENTATION_REPORT_CSV = PROCESSED_DIR / "paligemma_augmentation_report.csv"
PALIGEMMA_TRAIN_LOG_CSV = LOGS_DIR / "paligemma_train_log.csv"
PALIGEMMA_TEST_REPORT_CSV = LOGS_DIR / "paligemma_test_report.csv"
PALIGEMMA_STATE_CONFUSION_CSV = LOGS_DIR / "paligemma_state_confusion_matrix.csv"
PALIGEMMA_MISCLASSIFIED_CSV = LOGS_DIR / "paligemma_misclassified_samples.csv"
PALIGEMMA_STATE_ONLY_TRAIN_LOG_CSV = LOGS_DIR / "paligemma_state_only_train_log.csv"
PALIGEMMA_STATE_ONLY_TEST_REPORT_CSV = LOGS_DIR / "paligemma_state_only_test_report.csv"
PALIGEMMA_STATE_ONLY_STATE_CONFUSION_CSV = LOGS_DIR / "paligemma_state_only_state_confusion_matrix.csv"
PALIGEMMA_STATE_ONLY_MISCLASSIFIED_CSV = LOGS_DIR / "paligemma_state_only_misclassified_samples.csv"

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

for _d in (LABELS_DIR, PROCESSED_DIR, CACHE_DIR, AUGMENTED_DIR, PALIGEMMA_AUGMENTED_DIR, CHECKPOINTS_DIR, TINY_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
