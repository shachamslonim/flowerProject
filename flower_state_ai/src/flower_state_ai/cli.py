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
    augment_p = sub.add_parser("augment", help="Stage 4: augment the train split (rotate-balance closed/open, or arch=paligemma boosts low-count species)")
    augment_p.add_argument("--arch", choices=["mobilenet_v2", "paligemma"], default="mobilenet_v2")

    train_p = sub.add_parser("train", help="Stage 5: train the classifier")
    train_p.add_argument("--arch", choices=["mobilenet_v2", "tiny_cnn", "paligemma"], default="mobilenet_v2")
    train_p.add_argument("--epochs-head", type=int, default=None)
    train_p.add_argument("--epochs-finetune", type=int, default=None)
    train_p.add_argument("--batch-size", type=int, default=None)
    train_p.add_argument("--epochs", type=int, default=None, help="paligemma only")
    train_p.add_argument("--lr", type=float, default=None, help="paligemma only")
    train_p.add_argument("--lora-r", type=int, default=None, help="paligemma only")
    train_p.add_argument("--grad-accum-steps", type=int, default=None, help="paligemma only")
    train_p.add_argument("--limit", type=int, default=None, help="paligemma only: cap dataset size for a smoke test")
    train_p.add_argument("--pg-mode", choices=["joint", "state_only"], default="joint",
                          help="paligemma only: 'joint' predicts species+state; 'state_only' is given the species "
                               "name in the prompt (like the classifiers get it) and predicts state alone")

    evaluate_p = sub.add_parser("evaluate", help="Stage 6: evaluate on the held-out test split")
    evaluate_p.add_argument("--arch", choices=["mobilenet_v2", "tiny_cnn", "paligemma"], default="mobilenet_v2")
    evaluate_p.add_argument("--pg-mode", choices=["joint", "state_only"], default="joint", help="paligemma only")

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
            merge_verified_labels(dry_run=args.dry_run_merge, species_ids=[int(s) for s in args.species.split(",")] if args.species else None)
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
        if args.arch == "paligemma":
            from flower_state_ai.paligemma_dataset import build_paligemma_augmented_train, species_accuracy_map
            build_paligemma_augmented_train(accuracy_map=species_accuracy_map())
        else:
            from flower_state_ai.augmentation import build_augmented_train_set
            build_augmented_train_set()
    elif args.command == "train":
        if args.arch == "paligemma":
            from flower_state_ai.paligemma_train import main as paligemma_train_main
            kwargs = {"mode": args.pg_mode}
            if args.epochs is not None:
                kwargs["epochs"] = args.epochs
            if args.lr is not None:
                kwargs["lr"] = args.lr
            if args.lora_r is not None:
                kwargs["lora_r"] = args.lora_r
            if args.grad_accum_steps is not None:
                kwargs["grad_accum_steps"] = args.grad_accum_steps
            if args.limit is not None:
                kwargs["limit"] = args.limit
            paligemma_train_main(**kwargs)
        else:
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
        if args.arch == "paligemma":
            from flower_state_ai.paligemma_evaluate import evaluate as paligemma_evaluate
            paligemma_evaluate(mode=args.pg_mode)
        else:
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
