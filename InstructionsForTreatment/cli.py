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
        help="true renders the CLOSED-flower water-entry handling (controlled "
        "harvest); false renders the OPEN-flower handling (timing critical). "
        "Default: false",
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
        "--audience",
        choices=["farmer", "agronomist", "layperson"],
        default="farmer",
        help="Who the LLM writes for (llm mode only): farmer (default), "
        "agronomist, or layperson",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Ollama model tag for --mode llm (default: OLLAMA_MODEL env var, "
        "else llama3.2:latest)",
    )
    parser.add_argument(
        "--no-translate",
        dest="translate",
        action="store_false",
        help="For --language he --mode llm: trust the model to write Hebrew "
        "directly instead of composing in English and machine-translating "
        "(default: translate, because llama3.2 cannot write Hebrew)",
    )
    parser.add_argument(
        "--read",
        action="store_true",
        help="After printing the instructions, speak them aloud and save an audio "
        "file. Ignored for --format html.",
    )
    parser.add_argument(
        "--speak-engine",
        choices=["pyttsx3", "gtts", "mms"],
        default="pyttsx3",
        help="Speech engine for --read: pyttsx3 (default, offline, WAV), "
        "gtts (online, no API key, MP3, better Hebrew), or "
        "mms (offline neural VITS/MMS-TTS from Hugging Face, WAV)",
    )
    parser.add_argument(
        "--audio-output",
        default=None,
        help="Path for the spoken audio file (default: "
        "result/<flower>_<audience>_instructions.<wav|mp3>)",
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
            if args.read:
                print(
                    "Note: --read is ignored for --format html.", file=sys.stderr
                )
        else:
            result = generate_treatment_instructions(
                flower_name=args.flower,
                is_open=args.open,
                yellow_leaves=args.yellow_leaves,
                leaves_falling=args.leaves_falling,
                language=args.language,
                mode=args.mode,
                audience=args.audience,
                model=args.llm_model,
                translate=args.translate,
            )
            print(result)

            if args.read:
                from pathlib import Path

                from .speak import engine_extension, speak_text

                if args.audio_output:
                    audio_path = args.audio_output
                else:
                    ext = engine_extension(args.speak_engine)
                    result_dir = Path(__file__).resolve().parent / "result"
                    result_dir.mkdir(exist_ok=True)
                    audio_path = str(
                        result_dir
                        / f"{args.flower.replace(' ', '_')}_{args.audience}_instructions{ext}"
                    )
                speak_text(
                    result,
                    out_path=audio_path,
                    language=args.language,
                    engine=args.speak_engine,
                )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except NotImplementedError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
