# Codex Guide

## Goal

Maintain a faithful, easy-to-modify reproduction of TextPA with Qwen as the LLM backend.

## Important invariants

1. Do not pass ground-truth/reference text into the LLM for the TextPA reproduction. The paper evaluates Speechocean but TextPA itself operates without using the ground-truth transcript.
2. The LLM input should be human-readable textual cues only: ASR transcript, recognized CMU phones with pause markers, recognized IPA phones.
3. Canonical IPA from the transcript is used only for IPA match scoring, not for the LLM prompt.
4. IPA match scoring refines only accuracy; fluency remains LLM-based.
5. Charsiu fallback is debugging-only and must not be used in paper-style results.

## File map

- `src/textpa_qwen/data.py`: dataset preparation and manifest loading.
- `src/textpa_qwen/speech_cues.py`: all audio-to-textual-cue extraction.
- `src/textpa_qwen/prompts.py`: prompt templates and ablation field selection.
- `src/textpa_qwen/qwen_client.py`: LLM client abstraction.
- `src/textpa_qwen/scoring.py`: Smith-Waterman and evaluation.
- `src/textpa_qwen/pipeline.py`: orchestration and caching.
- `src/textpa_qwen/cli.py`: command-line entry points.

## Extension hooks for the user's later project

### Add intonation/pitch textual cue

Add a function in `speech_cues.py`:

```python
def pitch_summary(audio_path) -> str:
    ...
```

Then add a new field to `SpeechCues`, update `build_input_object()` in `prompts.py`, and add a new ablation value.

### Add linking/liaison cue

For scripted reading mode, add a reference-text-based module. Keep it out of the pure TextPA reproduction path because TextPA does not use ground-truth transcript for scoring.

### Add Qwen model variants

Change only `configs/default.yaml` or environment variables:

```bash
export QWEN_MODEL=qwen-turbo
export QWEN_MODEL=qwen-plus
export QWEN_MODEL=qwen-max
```

For local models:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct
export QWEN_BASE_URL=http://localhost:8000/v1
export QWEN_MODEL=Qwen/Qwen2.5-7B-Instruct
```

## Common failure modes

### Charsiu import failed

Clone Charsiu and export `CHARSIU_PATH`:

```bash
git clone https://github.com/lingjzhu/charsiu external/charsiu
export CHARSIU_PATH=$PWD/external/charsiu
```

### phonemizer cannot find espeak

Install system package:

```bash
sudo apt-get install -y espeak-ng
```

### Qwen returns non-JSON

Increase retries in config. The parser already strips code fences and extracts the first JSON object.

### Correlations differ from paper

Expected if using Qwen instead of GPT/Gemini. Also check whether the CMU fallback was accidentally enabled.
