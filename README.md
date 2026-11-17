# TextPA-Qwen Reproduction

ModeA/ModeB 连续确定性评分、ModeC parser 改造和完全不使用 Qwen 的 PCC
重算方法见 [docs/DETERMINISTIC_SCORING.md](docs/DETERMINISTIC_SCORING.md)。

This repository reproduces the core pipeline of **TextPA: Read to Hear — A Zero-Shot Pronunciation Assessment Using Textual Descriptions and LLMs**, replacing GPT/Gemini with Qwen/通义千问.

The implementation keeps the paper's main idea:

```text
speech audio
  -> ASR transcript
  -> recognized CMU phone sequence with pause markers
  -> recognized IPA phone sequence
  -> canonical IPA from the ASR transcript
  -> Qwen scores accuracy and fluency from textual cues
  -> Smith-Waterman IPA match score refines accuracy
  -> Pearson correlation with human scores
```

## 1. Module layout

The project is intentionally compressed into 7 code modules:

| Module | File | Responsibility |
|---|---|---|
| 1 | `data.py` | Prepare Speechocean762, build test manifest, load custom manifests. |
| 2 | `speech_cues.py` | Extract TextPA textual acoustic cues: Whisper transcript, Charsiu CMU+pauses, Wav2Vec2 IPA, phonemizer canonical IPA. |
| 3 | `prompts.py` | Figure-5-style TextPA prompt and ablation prompts. |
| 4 | `qwen_client.py` | Qwen API/local inference, JSON retry and parsing. |
| 5 | `scoring.py` | Smith-Waterman IPA/CMU match scoring, fixed-scale normalization, TextPA final score, Pearson correlation. |
| 6 | `pipeline.py` | End-to-end orchestration, cue cache, LLM cache, CSV/JSON output. |
| 7 | `cli.py` | Command-line entry points for preparing data, extracting cues, running Qwen, and evaluation. |

`utils.py` contains shared helpers and does not count as a research module.

## 2. Install

```bash
conda create -n textpa python=3.10 -y
conda activate textpa

sudo apt-get update
sudo apt-get install -y ffmpeg espeak-ng

bash scripts/setup_env.sh
```

### Charsiu setup

Charsiu is required for a paper-faithful CMU phone sequence with pause information:

```bash
mkdir -p external
git clone https://github.com/lingjzhu/charsiu external/charsiu
export CHARSIU_PATH=$PWD/external/charsiu
```

If Charsiu cannot be installed, set `allow_cmu_fallback: true` in the config for a smoke test only. Do **not** use the fallback for reported reproduction results.

## 3. Qwen setup

### Option A: DashScope / Alibaba Cloud Model Studio

```bash
export QWEN_API_KEY=your_api_key
export QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export QWEN_MODEL=qwen-plus
export QWEN_BACKEND=openai_compatible
```

`qwen-plus` is a good cost/performance starting point. Use `qwen-turbo` or `qwen-flash` for cheaper runs; use `qwen-max` for stronger but more expensive runs.

### Option B: Local vLLM

```bash
pip install vllm
vllm serve Qwen/Qwen2.5-7B-Instruct

export QWEN_BACKEND=openai_compatible
export QWEN_BASE_URL=http://localhost:8000/v1
export QWEN_MODEL=Qwen/Qwen2.5-7B-Instruct
export QWEN_API_KEY=EMPTY
```

### Option C: Local Transformers

In `configs/default.yaml`:

```yaml
models:
  qwen_backend: local_transformers
  qwen_model: Qwen/Qwen2.5-7B-Instruct
```

## 4. Smoke test

```bash
bash scripts/run_smoke_test.sh
```

This uses mock LLM and CMU fallback. It verifies code flow only.

## 5. Speechocean reproduction

If you already have the corpus in this repository as `speechocean762/`, prepare the
test manifest without downloading from Hugging Face:

```bash
textpa-prepare-speechocean --local-root speechocean762 --output-dir data/speechocean762 --split test
```

On Windows, the code automatically looks for eSpeak NG at
`C:\Program Files\eSpeak NG\libespeak-ng.dll`. If your install is elsewhere, set:

```bash
set PHONEMIZER_ESPEAK_LIBRARY=C:\path\to\libespeak-ng.dll
```

PowerShell equivalent:

```powershell
$env:PHONEMIZER_ESPEAK_LIBRARY = "C:\path\to\libespeak-ng.dll"
```

```bash
textpa-prepare-speechocean --output-dir data/speechocean762 --split test
textpa-run-experiment --config configs/default.yaml --ablations all
```

Outputs are written to:

```text
runs/textpa_qwen_speechocean/
  cues.jsonl
  llm_all.jsonl
  results_all.csv
  results_all_scored.csv
  results_all_metrics.json
  summary.json
```

Run ablations:

```bash
bash scripts/run_ablation_qwen.sh
```

## 5.1 MultiPA reproduction

If the MultiPA dataset is available as `MultiPA/`, prepare its 50-clip manifest:

```powershell
.venv\Scripts\textpa-prepare-multipa.exe --local-root MultiPA --output-dir data\multipa --split test
```

Run the full TextPA-Qwen pipeline with Qwen3.6-Plus:

```powershell
$env:CHARSIU_PATH = "D:\PythonProgram\textpa_qwen_reproduction\external\charsiu"
$env:QWEN_API_KEY = "your_api_key"
$env:QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:QWEN_MODEL = "qwen3.6-plus"
$env:QWEN_BACKEND = "openai_compatible"
$env:TEXTPA_OUTPUT_DIR = "runs\textpa_qwen_multipa_qwen36plus"
.venv\Scripts\textpa-run-experiment.exe --config configs\multipa.yaml --ablations all
```

Ablations:

```text
transcript       = LLM sees only ASR transcript
transcript_ipa   = transcript + recognized IPA
transcript_cmu   = transcript + recognized CMU phones with pauses
all              = transcript + CMU + IPA
TextPA           = LLM: all + normalized IPA match scoring for accuracy
```

## 6. Notes on exact reproducibility

The paper uses GPT/Gemini backends and reports single-run results. This reproduction deliberately replaces that LLM with Qwen, so the final correlations will not be numerically identical. The acoustic cue extraction and scoring protocol are kept as close as possible to the paper. The paper does not publish all Smith-Waterman scoring weights; this repo exposes them in `configs/default.yaml`.

## 7. Where to add your later innovations

Your next project step can be built on this code:

- Add pitch/intonation cues in `speech_cues.py`.
- Add linking/liaison rules as textual cues in `speech_cues.py`.
- Add new prompt variants in `prompts.py`.
- Add new scoring metrics and ablations in `scoring.py` and `pipeline.py`.
