# InstructionsForTreatment

Generate shelf life extension instructions for picked flowers. The module reads flower preservation data from `flowers.json` and produces step-by-step treatment instructions covering sterilization, chemical treatments, environmental conditions, and water management.

Output is available in **English** or **Hebrew**, using either **template-based** or **LLM-based** (Ollama) text generation. In LLM mode the text can be tailored to one of three
readers (**farmer**, **agronomist**, **layperson**), and any result can be **read aloud**
and saved as an audio file via a pluggable text-to-speech engine (`pyttsx3` offline,
`gtts` online, or `mms` offline neural VITS/MMS-TTS).

## Installation

```bash
cd c:\flowerAI\flowerProject
.venv\Scripts\Activate.ps1
pip install -r InstructionsForTreatment\requirements.txt
```

## Usage

### Python API

```python
from InstructionsForTreatment import generate_treatment_instructions

# English template (default)
result = generate_treatment_instructions(
    flower_name="Rose",
    is_open=True,
    yellow_leaves=False,
    leaves_falling=False,
    language="en",
    mode="template"
)
print(result)

# Hebrew template
result = generate_treatment_instructions(
    flower_name="Anemone",
    is_open=False,
    language="he"
)
print(result)

# LLM mode (requires Ollama running with llama3.2), written for a specific reader
result = generate_treatment_instructions(
    flower_name="Rose",
    is_open=True,
    language="en",
    mode="llm",
    audience="agronomist"  # "farmer" (default), "agronomist", or "layperson"
)
print(result)

# Read any result aloud and save an audio file (engine is pluggable)
from InstructionsForTreatment.speak import speak_text
speak_text(result, out_path="Rose_instructions.wav", language="en", engine="pyttsx3")
speak_text(result, out_path="Rose_he.mp3", language="he", engine="gtts")
speak_text(result, out_path="Rose_he_mms.wav", language="he", engine="mms")  # neural, offline
```

### Command Line

```bash
# English, flower is open
python -m InstructionsForTreatment --flower Rose --open true --language en

# Hebrew, flower is closed
python -m InstructionsForTreatment --flower Anemone --open false --language he

# LLM mode
python -m InstructionsForTreatment --flower Rose --open true --language en --mode llm

# LLM mode, tailored to a reader (farmer / agronomist / layperson)
python -m InstructionsForTreatment --flower Rose --open true --mode llm --audience layperson

# LLM mode in Hebrew: composes in English, auto-translates to Hebrew
python -m InstructionsForTreatment --flower Rose --language he --mode llm --audience farmer

# Print the instructions, then speak them aloud and save an audio file
python -m InstructionsForTreatment --flower Rose --open true --language en --read
python -m InstructionsForTreatment --flower Rose --language he --read --speak-engine gtts

# With yellow leaves flag
python -m InstructionsForTreatment --flower Rose --open true --yellow-leaves true --language en

powershell

cd C:\flowerAI\flowerProject
.\.venv\Scripts\python.exe -m InstructionsForTreatment --flower "Ammi visnaga" --open false --language he --format html --output "Ammi_visnaga_he.html

or 

powershell

Set-Location C:\flowerAI\flowerProject; .\.venv\Scripts\python.exe -m InstructionsForTreatment --flower "Ammi visnaga" --open false --language he --format html --output "Ammi_visnaga_he.html"
```

## Function Signature

```python
def generate_treatment_instructions(
    flower_name: str,          # English name of the flower (case-insensitive)
    is_open: bool,             # True -> CLOSED-flower wording; False -> OPEN-flower wording (see note)
    yellow_leaves: bool = False,  # If True, add Gibberellin recommendation
    leaves_falling: bool = False, # Reserved for future use
    language: str = "en",      # "en" (English) or "he" (Hebrew)
    mode: str = "template",    # "template" or "llm"
    audience: str = "farmer",  # "farmer", "agronomist", or "layperson" (llm mode only)
    model: str | None = None   # Ollama model tag (llm mode); None -> OLLAMA_MODEL env / llama3.2
) -> str
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `flower_name` | str | required | English name of the flower (looked up in flowers.json) |
| `is_open` | bool | required | Water-entry wording. `True` → CLOSED-flower handling (controlled harvest); `False` → OPEN-flower handling (timing critical). The 15-minute water-entry time is stated either way. |
| `yellow_leaves` | bool | False | Whether leaves are yellow (triggers Gibberellin) |
| `leaves_falling` | bool | False | Whether leaves are falling (reserved) |
| `language` | str | "en" | Output language: "en" or "he" |
| `mode` | str | "template" | Generation mode: "template" or "llm" |
| `audience` | str | "farmer" | Reader the LLM writes for: "farmer", "agronomist", or "layperson". Only affects `mode="llm"` |
| `model` | str \| None | None | Ollama model tag for `mode="llm"`. `None` → `OLLAMA_MODEL` env var, else `llama3.2:latest` |

## Instruction Output Order

1. **Sterilization** - Clean tools (bins, shears, tables, guillotine) with acetone
2. **Water Entry** - Harvest to water introduction, always 15 minutes.
   `is_open=False` → OPEN flower, timing between harvest and water is critical.
   `is_open=True` → CLOSED flower, controlled harvest.
   (The flag is intentionally inverted relative to the rendered OPEN/CLOSED label.)
3. **Materials and Chemical Treatment** - Biocide options from `farmerTreatment`, ranked
   `Option 1`, `Option 2`, ... (see [Ranking](#ranking-of-materials))
   - **Ethylene blocker** - TOG-L-101 with its DOC rate when listed, otherwise a generic
     STS recommendation when ethylene sensitivity >= 3
   - **Leaf treatment** - TOG-L-103 with its DOC rate when listed
   - **Additives** if specified
4. **Problem-specific treatments** - Gibberellin (`yellow_leaves=True`) and/or 101L
   (`leaves_falling=True`)
5. **Environmental Conditions** - Temperature: 20°C, Moisture: 30%
6. **Water Bucket Height** - Atmospheric pressure in cm (from flowers.json)

### Ranking of materials

Materials are ordered by the `priority` field in `flowers.json`. Everything marked
`recommended` comes first, then `standard`; `PRODUCT_PRIORITY_ORDER` only breaks ties
within a group. A flower can have more than one recommended product.

Labels are positional, and the recommendation is a separate tag:

| Language | Rank label | Recommended tag |
|----------|-----------|-----------------|
| English | `Option 1`, `Option 2`, ... | `, recommended` |
| Hebrew | `אפשרות 1`, `אפשרות 2`, ... | `, מומלץ` |

`Sugar` is an add-on rather than an alternative, so its line is annotated
`[can be added to the option]` / `[ניתן להוסיף לאפשרות]`.

## LLM Mode

LLM mode produces a **mix**: the factual values from `flowers.json` reproduced
**verbatim**, wrapped in **free-worded, plain-language explanation** of what the grower
does with each one and why.

- **Fixed values** (copied exactly, never altered): every `farmerTreatment` product
  name and `concentrationRate`, the `ethyleneBlocker` / `leafTreatment` product and
  rate, `additives`, `ethyleneSensitivity`, `waterBucketHeight`, plus the constants
  `20°C` / `30%` / `15 minutes` and the tool list / `acetone`.
- **Free to reword**: everything else — the sentence around each value, the ordering
  of the explanation, the "why".

The user message is built as an explicit `=== FIXED VALUES ===` block followed by
`=== YOUR JOB ===`; the system prompt tells the model those values are facts it may
only reword around. Requirements:

1. Install Ollama: https://ollama.ai
2. Pull a model: `ollama pull llama3.2`
3. Ensure Ollama is running: `ollama serve`

If Ollama is not available, the module falls back to template mode with a warning.
(Small models still drop or mangle some values despite the contract — see
[prompt tuning](#prompt-tuning-tune_promptpy) and `--compare`.)

### Model (`--llm-model` / `OLLAMA_MODEL`)

The default model is `llama3.2:latest`. Override it per run with `--llm-model <tag>`
or globally with the `OLLAMA_MODEL` environment variable
(precedence: `--llm-model` > `OLLAMA_MODEL` > default). The call sets `repeat_penalty`
and caps `num_predict` to blunt the runaway-repetition failure mode small models fall
into.

### Hebrew LLM output (`--no-translate`)

Small local models (`llama3.2`, `qwen2.5-coder`, …) write accurate **English** but
badly broken **Hebrew** — wrong terms, invented numbers, foreign glyphs. So for
`--language he --mode llm` the module **composes in English and machine-translates the
result to Hebrew** with `deep-translator` (Google Translate — free, no API key). The
facts come from the grounded English; the Hebrew reads fluently and keeps values like
`50 ס"מ`.

```bash
python -m InstructionsForTreatment --flower Rose --language he --mode llm \
  --audience farmer --read --speak-engine mms
```

Pass `--no-translate` (API: `translate=False`) to skip translation and trust the model
to write Hebrew directly — only useful with a genuinely multilingual model:

```bash
ollama pull aya-expanse:8b
python -m InstructionsForTreatment --flower Rose --language he --mode llm \
  --llm-model aya-expanse:8b --no-translate
```

If `deep-translator` is missing or the service fails, the English text is returned
unchanged with a warning. When the LLM itself is unavailable, the deterministic
Hebrew **template** is used (always accurate) regardless of `--no-translate`.

### Prompt tuning (`tune_prompt.py`)

The English LLM system prompt can be optimized automatically against the template
(treated as ground truth):

```bash
python -m InstructionsForTreatment.tune_prompt --iterations 5
```

Each iteration, for a set of eval flowers (`EVAL_CASES`):

1. the **template** facts are the truth (derived straight from `flowers.json`),
2. generate the LLM version with the same parameters and the current prompt,
3. score it deterministically — count facts that are **missing** (product names;
   exact rates, each of which must sit *next to its own product*; `acetone`, `20°C`,
   `30%`, water-bucket height, `15 minutes`, STS…) and values it **contradicts**,
4. the judge model (`qwen2.5-coder:7b-instruct` by default) proposes **general rule
   lines** to add — it may not emit product names, numbers, or example text (those
   are filtered out), so it can only tighten the rules, never leak answers.

The base prompt (`_DEFAULT_SYSTEM_BASE_EN`) is fixed; accepted rule lines accumulate
in its `Rules:` block. After N iterations the best rule set is **validated on
held-out flowers** (`VALIDATION_CASES`) and written to `prompts/system_en.txt` **only
if it beats the default there**. `llm_renderer` loads that file automatically in place
of the default; delete it to revert. Full log: `prompts/tuning_log.md`.

| Flag | Default | |
|---|---|---|
| `--iterations` | 5 | compare/rewrite rounds |
| `--gen-model` | `llama3.2:latest` | model whose prompt is tuned |
| `--judge-model` | `qwen2.5-coder:7b-instruct` | model that rewrites the prompt |
| `--dry-run` | off | write `prompts/system_en.candidate.txt`, don't apply |
| `--compare` | off | no tuning — dump **template vs LLM text** for every case to `prompts/comparison.md` so you can read the actual output |

```bash
# just look at what the LLM produces vs the template, with the current prompt
python -m InstructionsForTreatment.tune_prompt --compare
```

### Audience (`--audience`)

In LLM mode the instructions are written for **one reader**, chosen with `--audience`
(API: `audience=`). The data used is identical; only the wording, level of detail and
tone change.

| Audience | Written for | Style |
|----------|-------------|-------|
| `farmer` (default) | A working grower in the field | Short imperative steps, plain field language, keeps product names/rates, skips the chemistry theory |
| `agronomist` | A crop specialist | Precise terminology, every rate and method, brief mode-of-action rationale |
| `layperson` | Someone with no agricultural background | Everyday words, jargon and product codes explained on first use, friendly tone |

`--audience` has no effect in `template` mode (and none when LLM mode falls back to the
template).

## Read Aloud (`--read`)

`--read` prints the instructions as usual, then synthesizes them to speech, saves an
audio file, and plays it. The speech engine is **pluggable** — pick one with
`--speak-engine` (API: `speak_text(..., engine=...)` from `InstructionsForTreatment.speak`):

| `--speak-engine` | Kind | Output | Notes |
|---|---|---|---|
| `pyttsx3` (default) | Offline, OS speech engine (SAPI5 on Windows) | WAV | Needs a matching OS voice for the language |
| `gtts` | Online (Google Translate TTS), no API key | MP3 | Handles Hebrew without any OS voice; needs internet |
| `mms` | Offline neural VITS ([Meta MMS-TTS](https://huggingface.co/facebook/mms-tts)) via `transformers` | WAV | `facebook/mms-tts-eng` / `-heb`; downloads ~145 MB per language on first use, then offline. Needs `transformers` + `torch` + `scipy` |

Before synthesizing, `speak_text` prints the exact text it is about to read (to stderr),
framed by `--- Reading aloud (<engine>, <lang>) ---` … `--- end (<n> chars) ---`.

```bash
# Offline (English)
python -m InstructionsForTreatment --flower Rose --open true --read
# -> result/Rose_farmer_instructions.wav  (then auto-plays)

# Hebrew — gtts (online) or mms (offline neural), no Hebrew OS voice needed
python -m InstructionsForTreatment --flower Rose --language he --read --speak-engine gtts
python -m InstructionsForTreatment --flower Anemone --language he --read --speak-engine mms

python -m InstructionsForTreatment --flower Rose --read --audio-output C:\tmp\rose.wav
```

Notes:
- Default location: `InstructionsForTreatment/result/<flower>_<audience>_instructions.<wav|mp3>`
  (git-ignored). The extension follows the engine.
- `pyttsx3` + Hebrew needs a Hebrew voice installed in the OS; without one the default
  voice reads the Hebrew text poorly and a warning suggests another engine.
- `mms` reads digits and Latin product codes (`TOG-30`, `0.5`) less cleanly than words.
  Its first run needs internet to fetch the model; a uroman-only MMS language is
  detected and refused with a warning.
- A missing engine package, no internet (gtts / first-run mms), or no audio device
  prints a warning — the instructions still print and the command exits 0.
- Very long text (> `MAX_SPEAK_CHARS`, 8000) is truncated with a warning before synthesis.
- `--read` is ignored with `--format html`.

### `speak` package

```
InstructionsForTreatment/speak/
├── __init__.py          # exports speak_text, SPEAK_ENGINES, engine_extension
├── core.py              # engine-agnostic speak_text(): dispatch, print text, path, play
├── pyttsx3_engine.py    # offline WAV engine (synthesize + Hebrew voice pick)
├── gtts_engine.py       # online MP3 engine (synthesize + he→iw lang map)
├── mms_engine.py        # offline neural VITS/MMS-TTS engine (transformers + torch)
└── player.py            # play_file(): winsound for WAV, default app otherwise
```

To add another engine, create `speak/<name>_engine.py` exposing `EXT` and
`synthesize(text, out_path, language, rate) -> Path | None`, then register it in
`speak/core.py`'s `_ENGINES` dict.

## Example Output (English)

```
=== Treatment Instructions for Ageratum ===
1. STERILIZATION
   Sterilize the following tools with acetone: bins, shears, tables, guillotine
2. WATER ENTRY (HARVEST TO WATER INTRODUCTION)
   Flower is CLOSED: Controlled harvest.
   Harvest when flowers are closed. Time between harvest and
   water introduction (sink) is not critical for closed flowers.
3. MATERIALS AND CHEMICAL TREATMENT
   Purpose: Biocides - fungicides and bactericides for killing fungi
   and bacteria in the water solution.
   - TOG-3 (Option 1, recommended): concentration 0.15, method: mixing
   - TOG-6 (Option 2): concentration 0.0089, method: mixing
   - TOG-30 (Option 3): concentration 0.035, method: mixing
   - TOG-75 (Option 4): concentration 0.1, method: mixing
   ETHYLENE BLOCKER:
   - STS (TOG-L-101, recommended): concentration 0.25, method: mixing
     Purpose: Prevents petal drop caused by ethylene sensitivity.
4. ENVIRONMENTAL CONDITIONS
   Temperature: 20°C
   Moisture: 30%
5. WATER BUCKET HEIGHT (ATMOSPHERIC PRESSURE)
   Water bucket height: not specified
```

## Example Output (Hebrew)

```
=== הוראות טיפול עבור אגרטום ===
1. חיטוי
   יש לחטא את הכלים הבאים באצטון: מיכלים, מזמרות, שולחנות, גיליוטינה
2. כניסת מים (זמן בין קטיף להכנסה למים)
   הפרח סגור: קטיף מבוקר.
   יש לקטוף כשהפרחים סגורים.
   זמן בין קטיף להכנסה למים (sink) לא קריטי לפרחים סגורים.
3. חומרים וטיפול כימי
   מטרה: ביוצידים – חומרי קטילת פטריות וחיידקים במי ההשקיה.
   - TOG-3 (אפשרות 1, מומלץ): ריכוז 0.15, שיטת יישום: mixing
   - TOG-6 (אפשרות 2): ריכוז 0.0089, שיטת יישום: mixing
   - TOG-30 (אפשרות 3): ריכוז 0.035, שיטת יישום: mixing
   - TOG-75 (אפשרות 4): ריכוז 0.1, שיטת יישום: mixing
   חוסם אתילן:
   - STS (TOG-L-101, מומלץ): ריכוז 0.25, שיטת יישום: mixing
     מטרה: מניעת נשירה הנגרמת מרגישות לאתילן.
4. תנאי סביבה
   טמפרטורה: 20°C
   לחות: 30%
5. גובה מים בדלי (לחץ אטמוספרי)
   גובה מים בדלי: לא צוין
```

> Hebrew output needs a UTF-8 console. In PowerShell:
> `$env:PYTHONIOENCODING='utf-8'`

## Running Tests

```bash
cd c:\flowerAI\flowerProject
.venv\Scripts\Activate.ps1
python -m pytest InstructionsForTreatment\tests\ -v
```

## Project Structure

```
InstructionsForTreatment/
├── __init__.py           # Package entry, exports generate_treatment_instructions
├── __main__.py           # Enables: python -m InstructionsForTreatment
├── main.py               # Main function: generate_treatment_instructions()
├── data_loader.py        # Loads flower data from flowers.json
├── template_renderer.py  # Template-based EN/HE instruction generators
├── llm_renderer.py       # LLM-based instruction generator (Ollama), audience-aware
├── tune_prompt.py        # Optimize the English system prompt vs. the template (5-round loop)
├── prompts/              # tune_prompt output: system_en.txt (applied), tuning_log.md
├── translate.py          # EN→HE machine translation (deep-translator) for Hebrew LLM output
├── speak/                # Pluggable read-aloud package for --read (pyttsx3 / gtts / mms)
├── cli.py                # Command-line interface (argparse)
├── add_water_height.py   # Migration script for waterBucketHeight field
├── extract_doc_table.py  # Reads the TOG table (incl. red marking) out of the DOC
├── sync_from_doc.py      # Rewrites flowers.json preservation data from doc_table.json
├── doc_table.json        # Extracted DOC table snapshot (source of truth for the sync)
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── tests/
    ├── __init__.py
    ├── test_data_loader.py
    ├── test_template_en.py
    ├── test_template_he.py
    ├── test_main.py
    ├── test_llm_renderer.py
    ├── test_translate.py
    ├── test_tune_prompt.py
    └── test_speak.py
```

## Data Source

Flower data is read from `c:\flowerAI\flowerProject\flowers.json`. Key fields used:

- `englishName` / `hebrewName` - Flower identification
- `preservation.farmerTreatment` - Ranked biocide options (product, rate, method, `priority`)
- `preservation.ethyleneBlocker` - TOG-L-101 (STS) with its rate, or `null`
- `preservation.leafTreatment` - TOG-L-103 with its rate, or `null`
- `preservation.ethyleneSensitivity` - Scale 1-4; >=3 triggers a generic STS recommendation
  when no explicit `ethyleneBlocker` rate exists
- `preservation.additives` - Additional chemicals if specified
- `preservation.waterBucketHeight` - Water height in cm (atmospheric pressure)

### Syncing with the TOG recommended-use DOC

`flowers.json` is derived from the recommended-use table in
`A_monthly_updated_match_schedule.doc`. Two conventions in that table drive the data:

- *"Products & rates marked in red are highly adaptable"* -> `priority: "recommended"`
- *"\* = Dipping"* -> `applicationMethod: "dipping"` (otherwise `"mixing"`)

Because the recommendation is encoded as **font colour**, plain text extraction cannot
recover it. `extract_doc_table.py` therefore drives Microsoft Word over COM
(needs `pywin32` and Word installed) and records a `red` flag per cell:

```bash
# Only needed when the DOC itself changes
python -m InstructionsForTreatment.extract_doc_table   # -> doc_table.json

# Apply the extracted table to flowers.json (no Word required)
python -m InstructionsForTreatment.sync_from_doc
```

The sync rebuilds `farmerTreatment` from the *Pre-treatment grower* columns
(TOG-3, TOG-6, TOG-10, TOG-30, TOG-75, TOG-Galileo) plus `Sugar`, so products the table
does not list for a flower are removed rather than left behind. It also refreshes
`ethyleneSensitivity`, `ethyleneBlocker`, `leafTreatment` and `additives`.

Ten flowers in `flowers.json` have no row in the DOC table and are left untouched:
Tulip, Daffodil, Sage, Geranium, Petunia, Poppy, Dahlia, Zinnia, Cosmos, Clarkia.
