"""Stage 10: single CLI entrypoint for every stage of the pipeline."""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="flower_state_ai", description="Flower open/closed classifier pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("build-labels", help="Stage 1: build trustworthy labels from photo/")

    verify_p = sub.add_parser(
        "verify-labels",
        help="Stage 1.5: Ollama vision re-verification of species 001-015 (incl. review/)",
    )
    verify_p.add_argument("--limit", type=int, default=None)
    verify_p.add_argument("--species", type=str, default=None, help="comma-separated species ids, e.g. '1,2,3'")
    verify_p.add_argument("--model", type=str, default=None)
    verify_p.add_argument("--retry-errors", action="store_true")
    verify_p.add_argument("--merge-only", action="store_true", help="skip Ollama, just (re)merge the existing verification CSV")
    verify_p.add_argument("--dry-run-merge", action="store_true")

    sub.add_parser("preprocess", help="Stage 2: resize+cache images to 800px")
    sub.add_parser("split", help="Stage 3: stratified train/val/test split")
    sub.add_parser("augment", help="Stage 4: rotate-augment the closed class (train split only)")

    train_p = sub.add_parser("train", help="Stage 5: train the classifier")
    train_p.add_argument("--arch", choices=["mobilenet_v2", "tiny_cnn"], default="mobilenet_v2")
    train_p.add_argument("--epochs-head", type=int, default=None)
    train_p.add_argument("--epochs-finetune", type=int, default=None)
    train_p.add_argument("--batch-size", type=int, default=None)

    evaluate_p = sub.add_parser("evaluate", help="Stage 6: evaluate on the held-out test split")
    evaluate_p.add_argument("--arch", choices=["mobilenet_v2", "tiny_cnn"], default="mobilenet_v2")

    export_p = sub.add_parser("export-tiny", help="Stage 7: quantized + ONNX int8 'ultra tiny' export")
    export_p.add_argument("--arch", choices=["mobilenet_v2", "tiny_cnn"], default="mobilenet_v2")

    predict_p = sub.add_parser("predict", help="Predict open/closed for one photo (no LLM)")
    predict_p.add_argument("--photo", required=True)
    predict_p.add_argument("--arch", choices=["mobilenet_v2", "tiny_cnn"], default="mobilenet_v2")
    predict_p.add_argument("--tiny", action="store_true", help="use the exported ONNX int8 tiny model for the chosen --arch")

    ask_p = sub.add_parser("ask", help="End-to-end: ask the LangChain+Ollama agent about a photo")
    ask_p.add_argument("--photo", required=True)
    ask_p.add_argument("--flower", required=True)

    args = parser.parse_args()

    if args.command == "build-labels":
        from flower_state_ai.labeling import build_labels_master
        build_labels_master()
    elif args.command == "verify-labels":
        from flower_state_ai import config
        from flower_state_ai.label_verification import run_verification, merge_verified_labels
        species_ids = [int(s) for s in args.species.split(",")] if args.species else config.VERIFY_SPECIES_IDS
        model = args.model or config.OLLAMA_VISION_MODEL
        if args.merge_only:
            merge_verified_labels(dry_run=args.dry_run_merge)
        else:
            run_verification(model=model, limit=args.limit, species_ids=species_ids, retry_errors=args.retry_errors)
            # Merging is a separate, deliberate step (--merge-only) so results can be
            # spot-checked first - a plain run never writes to labels_master.csv.
            if args.dry_run_merge:
                merge_verified_labels(dry_run=True)
    elif args.command == "preprocess":
        from flower_state_ai.preprocessing import build_cache
        build_cache()
    elif args.command == "split":
        from flower_state_ai.splitting import build_split
        build_split()
    elif args.command == "augment":
        from flower_state_ai.augmentation import build_augmented_train_set
        build_augmented_train_set()
    elif args.command == "train":
        from flower_state_ai.train import main as train_main
        kwargs = {"arch": args.arch}
        if args.epochs_head is not None:
            kwargs["epochs_head"] = args.epochs_head
        if args.epochs_finetune is not None:
            kwargs["epochs_finetune"] = args.epochs_finetune
        if args.batch_size is not None:
            kwargs["batch_size"] = args.batch_size
        train_main(**kwargs)
    elif args.command == "evaluate":
        from flower_state_ai.evaluate import evaluate
        evaluate(arch=args.arch)
    elif args.command == "export-tiny":
        from flower_state_ai.export_tiny import run_export
        run_export(arch=args.arch)
    elif args.command == "predict":
        if args.tiny:
            from flower_state_ai.inference import predict_tiny
            result = predict_tiny(args.photo, arch=args.arch)
        else:
            from flower_state_ai.inference import predict
            result = predict(args.photo, arch=args.arch)
        print(f"{result.label} (confidence={result.confidence:.4f})")
    elif args.command == "ask":
        from flower_state_ai.agent import ask
        print(ask(args.photo, args.flower))


if __name__ == "__main__":
    main()
