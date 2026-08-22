"""Stage 6: evaluate the trained model on the held-out, never-augmented test split."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

from flower_state_ai import config
from flower_state_ai.dataset import get_transforms
from flower_state_ai.model import build_model


def load_checkpoint(checkpoint_path: Path = config.BEST_CHECKPOINT, arch: str = "mobilenet_v2") -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = build_model(arch=ckpt.get("arch", arch), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def evaluate(checkpoint_path: Path | None = None, arch: str = "mobilenet_v2") -> pd.DataFrame:
    reg = config.ARCH_REGISTRY[arch]
    checkpoint_path = checkpoint_path or reg["best_checkpoint"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(checkpoint_path, arch=arch).to(device)
    transform = get_transforms(False, reg["img_size"], reg["mean"], reg["std"])

    split_df = pd.read_csv(config.LABELS_SPLIT_CSV)
    test_df = split_df[split_df["split"] == "test"].reset_index(drop=True)

    idx_to_class = {v: k for k, v in config.CLASS_TO_IDX.items()}
    preds: list[int] = []
    confidences: list[float] = []
    labels: list[int] = []

    with torch.no_grad():
        for row in test_df.itertuples(index=False):
            img = Image.open(row.cached_abs_path).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            probs = torch.softmax(model(x), dim=1)[0]
            pred_idx = int(probs.argmax())
            preds.append(pred_idx)
            confidences.append(float(probs[pred_idx]))
            labels.append(config.CLASS_TO_IDX[row.resolved_state])

    test_df["pred_idx"] = preds
    test_df["pred_label"] = [idx_to_class[i] for i in preds]
    test_df["confidence"] = confidences
    test_df["true_idx"] = labels
    test_df["correct"] = test_df["pred_idx"] == test_df["true_idx"]

    class_names = [idx_to_class[i] for i in sorted(idx_to_class)]
    report = classification_report(labels, preds, target_names=class_names, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()

    per_species = (
        test_df.groupby(["species_id", "species_name"])["correct"]
        .mean()
        .reset_index()
        .rename(columns={"correct": "accuracy"})
    )

    conf = confusion_matrix(labels, preds, labels=list(config.CLASS_TO_IDX.values()))
    conf_df = pd.DataFrame(conf, index=list(config.CLASS_TO_IDX), columns=list(config.CLASS_TO_IDX))

    misclassified = (
        test_df[~test_df["correct"]]
        .sort_values("confidence", ascending=False)
        .head(20)[["image_path", "species_name", "resolved_state", "pred_label", "confidence"]]
    )

    test_report_csv = reg["test_report_csv"]
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(test_report_csv)
    with open(test_report_csv, "a", encoding="utf-8") as f:
        f.write("\nper_species_accuracy\n")
    per_species.to_csv(test_report_csv, mode="a", index=False)

    conf_df.to_csv(reg["confusion_matrix_csv"])
    misclassified.to_csv(reg["misclassified_csv"], index=False)

    print(f"[evaluate] arch={arch} overall accuracy={report['accuracy']:.4f}, macro f1={report['macro avg']['f1-score']:.4f}")
    print(f"[evaluate] wrote {test_report_csv}, {reg['confusion_matrix_csv']}, {reg['misclassified_csv']}")
    return test_df


if __name__ == "__main__":
    evaluate()
