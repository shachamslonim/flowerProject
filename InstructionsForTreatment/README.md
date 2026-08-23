# InstructionsForTreatment

Generate shelf life extension instructions for picked flowers. The module reads flower preservation data from `flowers.json` and produces step-by-step treatment instructions covering sterilization, chemical treatments, environmental conditions, and water management.

Output is available in **English** or **Hebrew**, using either **template-based** or **LLM-based** (Ollama) text generation.

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

# LLM mode (requires Ollama running with llama3.2)
result = generate_treatment_instructions(
    flower_name="Rose",
    is_open=True,
    language="en",
    mode="llm"
)
print(result)
```

### Command Line

```bash
# English, flower is open
python -m InstructionsForTreatment --flower Rose --open true --language en

# Hebrew, flower is closed
python -m InstructionsForTreatment --flower Anemone --open false --language he

# LLM mode
python -m InstructionsForTreatment --flower Rose --open true --language en --mode llm

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
    is_open: bool,             # If True, water entry time = 15 minutes
    yellow_leaves: bool = False,  # If True, add Gibberellin recommendation
    leaves_falling: bool = False, # Reserved for future use
    language: str = "en",      # "en" (English) or "he" (Hebrew)
    mode: str = "template"     # "template" or "llm"
) -> str
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `flower_name` | str | required | English name of the flower (looked up in flowers.json) |
| `is_open` | bool | required | Whether the flower is open (triggers 15-min water entry) |
| `yellow_leaves` | bool | False | Whether leaves are yellow (triggers Gibberellin) |
| `leaves_falling` | bool | False | Whether leaves are falling (reserved) |
| `language` | str | "en" | Output language: "en" or "he" |
| `mode` | str | "template" | Generation mode: "template" or "llm" |

## Instruction Output Order

1. **Sterilization** - Clean tools (bins, shears, tables, guillotine) with acetone
2. **Water Entry** - Harvest to water introduction. 15 minutes and timing-critical when
   `is_open=True`; controlled harvest and non-critical timing when closed
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

LLM mode uses Ollama with the `llama3.2:latest` model to rephrase the same data into natural language instructions. Requirements:

1. Install Ollama: https://ollama.ai
2. Pull the model: `ollama pull llama3.2`
3. Ensure Ollama is running: `ollama serve`

If Ollama is not available, the module falls back to template mode with a warning.

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
├── llm_renderer.py       # LLM-based instruction generator (Ollama)
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
    └── test_llm_renderer.py
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
