"""Streamlit GUI for the combined flower-care pipeline. Pure API client - every action
calls the FastAPI endpoints over HTTP (see api/main.py), never the underlying Python
packages directly, so the API is genuinely the single integration point for every agent,
not a parallel path that could drift from what the GUI actually does."""
from __future__ import annotations

import json
import os
from pathlib import Path

import requests
import streamlit as st

# A plain env var, not st.secrets - this is a local dev API base URL, not a secret, and
# st.secrets raises StreamlitSecretNotFoundError on any access at all (not just missing
# keys) when no secrets.toml exists anywhere, which it won't for most local setups.
API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8877")
REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_RECOMMENDATIONS_PATH = REPO_ROOT / "flower_state_ai" / "models" / "model_recommendations.json"

st.set_page_config(page_title="Flower Care Pipeline", page_icon="\U0001F33A", layout="centered")
st.title("\U0001F33A Flower Care Pipeline")
st.caption("Photo → species → open/closed state → care instructions, via the unified API.")


@st.cache_data
def _load_recommendations() -> dict:
    if not MODEL_RECOMMENDATIONS_PATH.exists():
        return {}
    with open(MODEL_RECOMMENDATIONS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return {row["species_name"].lower(): row for row in data["species"]}


def _recommended_state_arch(species_name: str | None) -> str | None:
    if not species_name:
        return None
    row = _load_recommendations().get(species_name.lower())
    return row["known_species_workflow"]["recommended_model"] if row else None


def _api_post(path: str, timeout: float = 120, **kwargs) -> dict:
    """timeout is a client-side safety net (default 2 min) so a genuinely hung request
    fails with a clear error instead of spinning forever - callers on slower stages
    (LLM mode, the full pipeline) pass a longer one. It does not make requests faster."""
    try:
        resp = requests.post(f"{API_BASE}{path}", timeout=timeout, **kwargs)
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"{path} did not respond within {timeout:.0f}s. The server may still be "
            "working (LLM mode and first-time model loading can take a while) - check "
            "api_server.log, or try again with more patience."
        ) from None
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        raise RuntimeError(f"{path} failed ({resp.status_code}): {detail}")
    return resp.json()


for key, default in [("identity", None), ("state", None), ("instructions", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

uploaded = st.file_uploader("Flower photo", type=["jpg", "jpeg", "png", "webp"])

st.subheader("1. Identify species")
vision_backend = st.radio(
    "Vision backend",
    options=["google", "paligemma"],
    format_func=lambda v: "Google Vision (broad, any flower)" if v == "google" else "PaliGemma (species 1-24 only, more accurate)",
    horizontal=True,
)

if uploaded and st.button("Identify"):
    with st.status("Identifying species...", expanded=True) as status:
        st.write(f"Sending your photo to the **{vision_backend}** vision backend.")
        try:
            files = {"image": (uploaded.name, uploaded.getvalue())}
            st.session_state.identity = _api_post(
                "/identify", timeout=60, files=files, data={"vision_backend": vision_backend}
            )
            st.session_state.state = None
            st.session_state.instructions = None
            status.update(label="Species identified", state="complete")
        except RuntimeError as e:
            status.update(label="Identification failed", state="error")
            st.error(str(e))

if st.session_state.identity:
    identity = st.session_state.identity
    st.success(f"Identified: **{identity['species_name'] or 'unknown'}**  (source: {identity['source']})")
    if identity.get("known_species_matches"):
        st.caption(f"flowers.json matches: {', '.join(identity['known_species_matches'])}")
    with st.expander("All detected labels"):
        st.table(identity.get("all_labels", []))

    species_name = st.text_input("Species name (edit if the identification was wrong)", value=identity["species_name"] or "")

    st.subheader("2. Determine open/closed state")
    recommended = _recommended_state_arch(species_name)
    arch_options = ["mobilenet_v2", "tiny_cnn", "paligemma_state_only"]
    default_idx = arch_options.index(recommended) if recommended in arch_options else 0
    state_arch = st.radio(
        "Model" + (f" (recommended for {species_name}: {recommended})" if recommended else ""),
        options=arch_options,
        index=default_idx,
        horizontal=True,
    )

    if uploaded and species_name and st.button("Classify state"):
        with st.status("Classifying open/closed state...", expanded=True) as status:
            st.write(f"Running the **{state_arch}** model on your photo for **{species_name}**.")
            st.write("First call for a given model loads it into memory, so it's slower than later calls.")
            try:
                files = {"image": (uploaded.name, uploaded.getvalue())}
                st.session_state.state = _api_post(
                    "/state", timeout=120, files=files,
                    data={"species_name": species_name, "arch": state_arch},
                )
                st.session_state.instructions = None
                status.update(label="State classified", state="complete")
            except RuntimeError as e:
                status.update(label="Classification failed", state="error")
                st.error(str(e))

if st.session_state.state:
    state = st.session_state.state
    st.success(f"State: **{state['state'].upper()}**  (model: {state['arch_used']})")

    st.subheader("3. Care instructions")
    col1, col2 = st.columns(2)
    with col1:
        language = st.selectbox("Language", ["en", "he"], format_func=lambda v: "English" if v == "en" else "Hebrew")
        mode = st.selectbox("Mode", ["template", "llm"])
        yellow_leaves = st.checkbox("Yellow leaves")
        leaves_falling = st.checkbox("Leaves falling")
    with col2:
        if mode == "llm":
            audience = st.selectbox("Audience", ["farmer", "agronomist", "layperson"])
            llm_model = st.text_input("Ollama model (blank = default)", value="")
        else:
            audience = "farmer"
            llm_model = ""
        translate = st.checkbox("Translate EN->HE for Hebrew LLM output", value=True)
        read = st.checkbox("Read aloud")
        speak_engine = st.selectbox("Speak engine", ["pyttsx3", "gtts", "mms"], disabled=not read)

    if st.button("Generate instructions"):
        with st.status("Generating instructions...", expanded=True) as status:
            st.write("Loading this flower's treatment data (products, rates, environment) from flowers.json.")
            if mode == "llm":
                st.write(
                    f"Asking the local Ollama model (**{llm_model or 'default'}**) to write it up "
                    f"for a **{audience}**, keeping every value exact - up to 2 tries if the first "
                    "reply looks cut off."
                )
                if language == "he":
                    if translate:
                        st.write(
                            "Composing in English first (small local models write broken Hebrew), "
                            "then machine-translating the result to Hebrew."
                        )
                    else:
                        st.write("Asking the model to write directly in Hebrew (--no-translate).")
            else:
                st.write("Filling in the fixed English/Hebrew template - no LLM involved.")
            if read:
                st.write(f"Synthesizing speech from the result with the **{speak_engine}** engine.")
            try:
                # is_open is InstructionsForTreatment's own (intentionally inverted) flag -
                # True renders CLOSED-flower wording, so a "closed" bloom state maps to True.
                is_open_flag = st.session_state.state["state"] == "closed"
                st.session_state.instructions = _api_post(
                    "/instructions",
                    timeout=240 if mode == "llm" else 30,
                    json={
                        "species_name": species_name,
                        "is_open": is_open_flag,
                        "yellow_leaves": yellow_leaves,
                        "leaves_falling": leaves_falling,
                        "language": language,
                        "mode": mode,
                        "audience": audience,
                        "llm_model": llm_model or None,
                        "translate": translate,
                        "read": read,
                        "speak_engine": speak_engine,
                    },
                )
                status.update(label="Instructions ready", state="complete")
            except RuntimeError as e:
                status.update(label="Generation failed", state="error")
                st.error(str(e))

if st.session_state.instructions:
    instructions = st.session_state.instructions
    direction = "rtl" if st.session_state.get("_last_lang") == "he" else "ltr"
    st.text_area("Instructions", value=instructions["text"], height=400)
    if instructions.get("audio_path"):
        filename = Path(instructions["audio_path"]).name
        try:
            audio_bytes = requests.get(f"{API_BASE}/audio/{filename}", timeout=30).content
            # Text is already rendered above; autoplay starts reading it only now.
            st.audio(audio_bytes, autoplay=True)
        except requests.RequestException:
            st.caption(f"Audio saved at: {instructions['audio_path']}")

st.divider()
with st.expander("Run the full pipeline in one click instead"):
    st.caption(
        "Runs identify -> state -> instructions as three chained calls, so each step's "
        "checkmark reflects it actually finishing (see monitor/tracing.py for the "
        "single-request /pipeline endpoint this drives, one call per stage here)."
    )
    pl_language = st.selectbox("Language ", ["en", "he"], key="pl_lang")
    pl_mode = st.selectbox("Mode ", ["template", "llm"], key="pl_mode")
    if pl_mode == "llm":
        pl_audience = st.selectbox(
            "Audience ", ["farmer", "agronomist", "layperson"], key="pl_audience"
        )
    else:
        pl_audience = "farmer"
    pl_read = st.checkbox("Read aloud", key="pl_read")
    pl_speak_engine = st.selectbox(
        "Speak engine ", ["pyttsx3", "gtts", "mms"], key="pl_speak_engine", disabled=not pl_read
    )
    if uploaded and st.button("Run full pipeline"):
        result = None
        with st.status("Running the full pipeline...", expanded=True) as status:
            try:
                st.write(f"**1. Identify** the species with the **{vision_backend}** vision backend...")
                files = {"image": (uploaded.name, uploaded.getvalue())}
                identity = _api_post(
                    "/identify", timeout=60, files=files, data={"vision_backend": vision_backend}
                )
                pl_species = identity["species_name"]
                if not pl_species:
                    raise RuntimeError(
                        "Could not identify a flower species. Detected labels: "
                        f"{identity.get('all_labels') or identity.get('all_matches')}"
                    )
                st.write(f"✅ **1. Identified:** {pl_species}")

                st.write("**2. Classify** open/closed state with the model recommended for that species...")
                files = {"image": (uploaded.name, uploaded.getvalue())}
                pl_state = _api_post(
                    "/state", timeout=120, files=files, data={"species_name": pl_species}
                )
                st.write(f"✅ **2. State:** {pl_state['state'].upper()} (model: {pl_state['arch_used']})")

                if pl_mode == "llm":
                    st.write(
                        f"**3. Generate instructions** with the local Ollama model, written for a "
                        f"**{pl_audience}** - this step is the slow one, typically a minute or so, "
                        "longer on the first call while the model warms up..."
                    )
                else:
                    st.write("**3. Generate instructions** by filling in the fixed template...")
                if pl_read:
                    st.write(f"   (also reading it aloud with the **{pl_speak_engine}** engine)")
                # is_open is InstructionsForTreatment's own (intentionally inverted) flag -
                # True renders CLOSED-flower wording, so a "closed" bloom state maps to True.
                pl_is_open = pl_state["state"] == "closed"
                pl_instructions = _api_post(
                    "/instructions",
                    timeout=240 if pl_mode == "llm" else 30,
                    json={
                        "species_name": pl_species,
                        "is_open": pl_is_open,
                        "language": pl_language,
                        "mode": pl_mode,
                        "audience": pl_audience,
                        "read": pl_read,
                        "speak_engine": pl_speak_engine,
                    },
                )
                st.write("✅ **3. Instructions ready" + (" and audio generated.**" if pl_read else ".**"))

                result = {
                    "species_name": pl_species,
                    "state": pl_state["state"],
                    "instructions_text": pl_instructions["text"],
                    "audio_path": pl_instructions["audio_path"],
                }
                status.update(
                    label=f"Done: {pl_species} - {pl_state['state'].upper()}", state="complete"
                )
            except RuntimeError as e:
                status.update(label="Pipeline failed", state="error")
                st.error(str(e))
        if result:
            st.success(f"{result['species_name']} - {result['state'].upper()}")
            st.text_area("Instructions ", value=result["instructions_text"], height=400)
            if result.get("audio_path"):
                filename = Path(result["audio_path"]).name
                try:
                    audio_bytes = requests.get(f"{API_BASE}/audio/{filename}", timeout=30).content
                    # Text is already rendered above; autoplay starts reading it only now.
                    st.audio(audio_bytes, autoplay=True)
                except requests.RequestException:
                    st.caption(f"Audio saved at: {result['audio_path']}")
