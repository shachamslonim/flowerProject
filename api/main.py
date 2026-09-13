"""Single FastAPI app wrapping the three flower-care components. Endpoints call straight
into monitor.tracing's traced functions - no extra service layer, since those functions
already are the right shape (one per pipeline stage, plus the combined run_pipeline)."""
from __future__ import annotations

import shutil
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from monitor.langsmith_setup import init_langsmith  # noqa: E402
from monitor.tracing import classify_state, generate_instructions, identify_species, run_pipeline  # noqa: E402

from api.schemas import (  # noqa: E402
    Audience,
    IdentifyResponse,
    InstructionsRequest,
    InstructionsResponse,
    Language,
    Mode,
    PipelineResponse,
    SpeakEngine,
    StateArch,
    StateResponse,
    VisionBackend,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_langsmith()
    yield


app = FastAPI(title="Flower Care Pipeline API", lifespan=lifespan)


def _save_upload(image: UploadFile) -> Path:
    suffix = Path(image.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(image.file, tmp)
        return Path(tmp.name)


def _run_traced(fn, *args, **kwargs):
    """ValueError/FileNotFoundError from the underlying components are user errors (bad
    image, unrecognized species, missing file) -> 400; anything else is a real bug -> 500,
    left to FastAPI's default handler so the traceback isn't swallowed."""
    try:
        return fn(*args, **kwargs)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/identify", response_model=IdentifyResponse)
def identify(image: UploadFile = File(...), vision_backend: VisionBackend = Form("google")) -> IdentifyResponse:
    path = _save_upload(image)
    try:
        result = _run_traced(identify_species, str(path), vision_backend=vision_backend)
    finally:
        path.unlink(missing_ok=True)
    return IdentifyResponse(**result)


@app.post("/state", response_model=StateResponse)
def state(
    image: UploadFile = File(...),
    species_name: str = Form(...),
    arch: StateArch | None = Form(None),
) -> StateResponse:
    path = _save_upload(image)
    try:
        result = _run_traced(classify_state, str(path), species_name, arch=arch)
    finally:
        path.unlink(missing_ok=True)
    return StateResponse(**result)


@app.post("/instructions", response_model=InstructionsResponse)
def instructions(req: InstructionsRequest) -> InstructionsResponse:
    result = _run_traced(
        generate_instructions,
        species_name=req.species_name,
        is_open=req.is_open,
        yellow_leaves=req.yellow_leaves,
        leaves_falling=req.leaves_falling,
        language=req.language,
        mode=req.mode,
        audience=req.audience,
        llm_model=req.llm_model,
        translate=req.translate,
        read=req.read,
        speak_engine=req.speak_engine,
    )
    return InstructionsResponse(**result)


@app.post("/pipeline", response_model=PipelineResponse)
def pipeline(
    image: UploadFile = File(...),
    vision_backend: VisionBackend = Form("google"),
    state_arch: StateArch | None = Form(None),
    yellow_leaves: bool = Form(False),
    leaves_falling: bool = Form(False),
    language: Language = Form("en"),
    mode: Mode = Form("template"),
    audience: Audience = Form("farmer"),
    llm_model: str | None = Form(None),
    translate: bool = Form(True),
    read: bool = Form(False),
    speak_engine: SpeakEngine = Form("pyttsx3"),
) -> PipelineResponse:
    path = _save_upload(image)
    try:
        result = _run_traced(
            run_pipeline,
            str(path),
            vision_backend=vision_backend,
            state_arch=state_arch,
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
    finally:
        path.unlink(missing_ok=True)
    return PipelineResponse(**result)


@app.get("/audio/{filename}")
def get_audio(filename: str) -> FileResponse:
    """Serves a generated read-aloud file back to a client (e.g. the Streamlit GUI) that
    only has the path run_pipeline returned, not direct filesystem access."""
    path = REPO_ROOT / "InstructionsForTreatment" / "result" / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(path)
