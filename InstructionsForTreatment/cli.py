"""Command-line interface for generating flower treatment instructions."""

from __future__ import annotations

import argparse
import sys


def parse_bool(value: str) -> bool:
    """Parse a boolean string argument."""
    if value.lower() in ("true", "1", "yes", "y"):
        return True
    if value.lower() in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{value}'")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="InstructionsForTreatment",
        description="Generate shelf life extension instructions for picked flowers.",
    )
    parser.add_argument(
        "--flower",
        required=True,
        help="English name of the flower (e.g., Rose, Anemone, Sunflower)",
    )
    parser.add_argument(
        "--open",
        type=parse_bool,
        default=False,
        help="Whether the flower is open (true/false, default: false)",
    )
    parser.add_argument(
        "--yellow-leaves",
        type=parse_bool,
        default=False,
        help="Whether leaves are yellow (true/false, default: false)",
    )
    parser.add_argument(
        "--leaves-falling",
        type=parse_bool,
        default=False,
        help="Whether leaves are falling (true/false, default: false)",
    )
    parser.add_argument(
        "--language",
        choices=["en", "he"],
        default="en",
        help="Output language: en (English) or he (Hebrew). Default: en",
    )
    parser.add_argument(
        "--mode",
        choices=["template", "llm"],
        default="template",
        help="Generation mode: template (default) or llm (uses Ollama)",
    )
    parser.add_argument(
        "--format",
        choices=["text", "html"],
        default="text",
        help="Output format: text (print to stdout) or html (write .html file). Default: text",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path for html format. Default: <flower>_instructions.html",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    from .main import generate_treatment_instructions

    try:
        if args.format == "html":
            from pathlib import Path

            from .data_loader import load_flower_data
            from .html_renderer import render_instructions_html

            flower_data = load_flower_data(args.flower)
            html_content = render_instructions_html(
                flower_data=flower_data,
                is_open=args.open,
                yellow_leaves=args.yellow_leaves,
                leaves_falling=args.leaves_falling,
                language=args.language,
            )
            if args.output:
                out_path = args.output
            else:
                result_dir = Path(__file__).resolve().parent / "result"
                result_dir.mkdir(exist_ok=True)
                out_path = str(result_dir / f"{args.flower.replace(' ', '_')}_instructions.html")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            print(f"HTML written to: {out_path}")
        else:
            result = generate_treatment_instructions(
                flower_name=args.flower,
                is_open=args.open,
                yellow_leaves=args.yellow_leaves,
                leaves_falling=args.leaves_falling,
                language=args.language,
                mode=args.mode,
            )
            print(result)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except NotImplementedError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
