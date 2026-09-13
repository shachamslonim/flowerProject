"""The traced pipeline orchestrator: photo -> species -> open/closed state -> care
instructions. One @traceable-decorated function per stage plus one wrapping the whole
run, so LangSmith (once monitor.langsmith_setup.init_langsmith() has been called) shows
one parent trace per request with each stage as a nested span - the instructions stage's
langchain_ollama calls (mode="llm") nest automatically inside that span with no extra code,
since LangSmith's tracing context is global/thread-local once LANGCHAIN_TRACING_V2 is set.

If LangSmith isn't configured, @traceable is a no-op wrapper (langsmith's own behavior) -
this module works identically either way.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from langsmith import traceable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))  # flower_vision.py is a loose module at repo root, not a package

MODEL_RECOMMENDATIONS_PATH = REPO_ROOT / "flower_state_ai" / "models" / "model_recommendations.json"


def _default_state_arch(species_name: str) -> str:
    """Looks up the per-species best state-only model from model_recommendations.json
    (built during the PaliGemma experiments - see that file's "known_species_workflow").
    Falls back to mobilenet_v2 (the strongest model with full 85-species coverage) if the
    species isn't in there (i.e. not one of the 24 PaliGemma-covered species) or the file
    is missing."""
    if not MODEL_RECOMMENDATIONS_PATH.exists():
        return "mobilenet_v2"
    with open(MODEL_RECOMMENDATIONS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    for row in data["species"]:
        if row["species_name"].lower() == species_name.lower():
            return row["known_species_workflow"]["recommended_model"] or "mobilenet_v2"
    return "mobilenet_v2"


@traceable(name="identify_species")
def identify_species(image_path: str, vision_backend: str = "google") -> dict:
    """vision_backend="google": flower_vision.py's generic Vision-API label match (broad,
    any flower Vision recognizes, no confidence score, no state).
    vision_backend="paligemma": flower_state_ai's trained joint-mode model (species 1-24
    only, much higher accuracy, also returns a state guess though the pipeline re-derives
    state separately in classify_state() using the identified species)."""
    if vision_backend == "google":
        import flower_vision
        return flower_vision.identify_flower(Path(image_path))
    if vision_backend == "paligemma":
        from flower_state_ai.paligemma_predict import predict_paligemma
        result = predict_paligemma(image_path, mode="joint")
        return {
            "species_name": result["species_name"],
            "confidence": None,
            "all_matches": [result["species_name"]] if result["species_name"] != "UNPARSEABLE" else [],
            "all_labels": [],
            "source": "paligemma_joint",
        }
    raise ValueError(f"unknown vision_backend: {vision_backend!r}")


@traceable(name="classify_state")
def classify_state(image_path: str, species_name: str, arch: str | None = None) -> dict:
    """arch=None auto-picks the best model for this species from model_recommendations.json
    (mobilenet_v2, tiny_cnn, or paligemma_state_only - whichever scored highest in testing).
    Explicit arch overrides that lookup."""
    arch = arch or _default_state_arch(species_name)

    if arch == "paligemma_state_only" or arch == "paligemma":
        from flower_state_ai.paligemma_predict import predict_paligemma
        result = predict_paligemma(image_path, mode="state_only", species_name=species_name)
        return {"state": result["state"], "confidence": None, "arch_used": "paligemma_state_only"}

    from flower_state_ai.inference import predict, predict_tiny
    if arch == "tiny_cnn":
        result = predict_tiny(image_path, arch="tiny_cnn")
    else:
        result = predict(image_path, arch="mobilenet_v2")
    return {"state": result.label, "confidence": result.confidence, "arch_used": arch}


@traceable(name="generate_instructions")
def generate_instructions(
    species_name: str,
    is_open: bool,
    yellow_leaves: bool = False,
    leaves_falling: bool = False,
    language: str = "en",
    mode: str = "template",
    audience: str = "farmer",
    llm_model: str | None = None,
    translate: bool = True,
    read: bool = False,
    speak_engine: str = "pyttsx3",
) -> dict:
    from InstructionsForTreatment import generate_treatment_instructions

    text = generate_treatment_instructions(
        flower_name=species_name,
        is_open=is_open,
        yellow_leaves=yellow_leaves,
        leaves_falling=leaves_falling,
        language=language,
        mode=mode,
        audience=audience,
        model=llm_model,
        translate=translate,
    )

    audio_path = None
    if read:
        from InstructionsForTreatment.speak import engine_extension, speak_text

        result_dir = REPO_ROOT / "InstructionsForTreatment" / "result"
        result_dir.mkdir(exist_ok=True)
        ext = engine_extension(speak_engine)
        audio_path = str(result_dir / f"{species_name.replace(' ', '_')}_{audience}_instructions{ext}")
        # play=False: this runs on the API server, not the caller's machine - playing
        # here would speak out of the server's speakers instead of the client's, and
        # would block the response (so the client sees the result only after playback
        # finishes). The client (GUI) fetches the audio via GET /audio/{filename} and
        # plays it itself, after the text is already shown.
        speak_text(text, out_path=audio_path, language=language, engine=speak_engine, play=False)

    return {"text": text, "audio_path": audio_path}


@traceable(name="flower_care_pipeline")
def run_pipeline(
    image_path: str,
    vision_backend: str = "google",
    state_arch: str | None = None,
    yellow_leaves: bool = False,
    leaves_falling: bool = False,
    language: str = "en",
    mode: str = "template",
    audience: str = "farmer",
    llm_model: str | None = None,
    translate: bool = True,
    read: bool = False,
    speak_engine: str = "pyttsx3",
) -> dict:
    """The end-to-end pipeline the /pipeline endpoint calls - this is the one function
    that shows up as a single traced run in LangSmith with identify/classify/instructions
    as nested child spans."""
    identity = identify_species(image_path, vision_backend=vision_backend)
    species_name = identity["species_name"]
    if not species_name:
        raise ValueError(
            f"Could not identify a flower species (vision_backend={vision_backend!r}). "
            f"Detected labels: {identity.get('all_labels') or identity.get('all_matches')}"
        )

    state_result = classify_state(image_path, species_name, arch=state_arch)
    # InstructionsForTreatment's is_open flag is intentionally inverted relative to the
    # rendered OPEN/CLOSED label (see its README) - True renders CLOSED-flower wording.
    is_open_flag = state_result["state"] == "closed"

    instructions = generate_instructions(
        species_name=species_name,
        is_open=is_open_flag,
        yellow_leaves=yellow_leaves,
        leaves_falling=leaves_falling,
        language=language,
        mode=mode,
        audience=audience,
        llm_model=llm_model,
        translate=translate,
        read=read,
        speak_engine=speak_engine,
    )

    return {
        "species_name": species_name,
        "species_source": identity["source"],
        "state": state_result["state"],
        "state_arch_used": state_result["arch_used"],
        "instructions_text": instructions["text"],
        "audio_path": instructions["audio_path"],
    }
