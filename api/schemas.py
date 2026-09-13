"""Pydantic request/response models. Deliberately thin - these type the exact parameters
InstructionsForTreatment.generate_treatment_instructions() and monitor.tracing's functions
already accept, not a new contract."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

VisionBackend = Literal["google", "paligemma"]
StateArch = Literal["mobilenet_v2", "tiny_cnn", "paligemma_state_only"]
Language = Literal["en", "he"]
Mode = Literal["template", "llm"]
Audience = Literal["farmer", "agronomist", "layperson"]
SpeakEngine = Literal["pyttsx3", "gtts", "mms"]


class LabelScore(BaseModel):
    description: str
    score: float


class IdentifyResponse(BaseModel):
    species_name: str | None
    confidence: float | None
    source: str
    all_matches: list[str] = Field(default_factory=list)
    known_species_matches: list[str] = Field(default_factory=list)
    all_labels: list[LabelScore] = Field(default_factory=list)


class StateResponse(BaseModel):
    state: Literal["open", "closed"]
    confidence: float | None
    arch_used: str


class InstructionsRequest(BaseModel):
    species_name: str
    is_open: bool
    yellow_leaves: bool = False
    leaves_falling: bool = False
    language: Language = "en"
    mode: Mode = "template"
    audience: Audience = "farmer"
    llm_model: str | None = None
    translate: bool = True
    read: bool = False
    speak_engine: SpeakEngine = "pyttsx3"


class InstructionsResponse(BaseModel):
    text: str
    audio_path: str | None


class PipelineResponse(BaseModel):
    species_name: str
    species_source: str
    state: Literal["open", "closed"]
    state_arch_used: str
    instructions_text: str
    audio_path: str | None
