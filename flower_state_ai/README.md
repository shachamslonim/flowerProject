# flower_state_ai

Local pipeline that trains a MobileNetV2 classifier to tell whether a flower photo shows an
**open** or **closed** flower, across all 85 species under `C:\flowerProject\photo\`, plus a
LangChain + Ollama agent that answers questions like "is this Rose open or closed?" using the
trained model as a tool.

## Setup

From `C:\flowerProject`, using the existing project venv:

```powershell
.\.venv\Scripts\Activate.ps1

# PyTorch — pick ONE of these first, then install the rest:
# CPU-only machine:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# NVIDIA/CUDA machine (match the CUDA version to your driver, e.g. cu121):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r flower_state_ai\requirements.txt
pip install -e flower_state_ai
```

Make sure Ollama is running and the agent's model is pulled:

```powershell
ollama serve
ollama pull llama3.2:latest
```

## Running the pipeline

```powershell
python -m flower_state_ai.cli build-labels
python -m flower_state_ai.cli preprocess
python -m flower_state_ai.cli split
python -m flower_state_ai.cli augment
python -m flower_state_ai.cli train
python -m flower_state_ai.cli evaluate
python -m flower_state_ai.cli export-tiny
```

Each stage reads/writes CSV reports under `data/` and `models/logs/` (gitignored) so you can
inspect intermediate results — see the plan's verification checklist for what to check at each
step (e.g. `data/labels/label_conflicts.csv`, `models/logs/test_report.csv`).

## Using it

```powershell
# Direct model prediction, no LLM involved
python -m flower_state_ai.cli predict --photo path\to\photo.jpg
python -m flower_state_ai.cli predict --photo path\to\photo.jpg --tiny

# Full agent: natural-language answer via the local Ollama LLM
python -m flower_state_ai.cli ask --photo path\to\photo.jpg --flower "Rose"
```

## Notes

- Training auto-selects CUDA if available, else CPU (`flower_state_ai.train.get_device`). The
  same code can run on this machine or be copied to another machine with an NVIDIA GPU — only
  the PyTorch install command differs (see Setup above).
- The classifier is one global open/closed model trained across all species; the flower name is
  only used by the agent to phrase its answer, never fed into the model itself.
