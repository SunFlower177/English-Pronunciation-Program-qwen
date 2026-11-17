# Reproduction Plan

## Paper-faithful core

The paper's TextPA method extracts text-based acoustic descriptors from audio and sends them to an LLM:

1. ASR transcript: Whisper large-v3-en in the paper. This repo uses `openai-whisper` `large-v3` with `language=en`, or Hugging Face `openai/whisper-large-v3`.
2. Recognized CMU sequence with pauses: Charsiu predictive aligner.
3. Recognized IPA sequence: `facebook/wav2vec2-lv-60-espeak-cv-ft`.
4. Canonical IPA from ASR transcript: `phonemizer` with `EspeakBackend("en-us")`.
5. LLM assessment: replaced by Qwen.
6. IPA match scoring: Smith-Waterman similarity between recognized IPA and canonical IPA.
7. Final accuracy: mean of min-max-normalized LLM accuracy and min-max-normalized IPA match score.
8. Final fluency: min-max-normalized LLM fluency.

## Datasets

### Public dataset: Speechocean762

Use the Hugging Face dataset:

```python
from datasets import load_dataset
testset = load_dataset("mispeech/speechocean762", split="test")
```

The paper uses only the test split because TextPA is zero-shot and does not require training data.

### MultiPA

The paper also evaluates on MultiPA, a small open-response dataset with 50 clips. If the dataset is not available to you, use Speechocean for exact public reproduction and optionally collect a small open-response set for extension experiments.

## Experiments to run

### Experiment 1: Main TextPA-Qwen on Speechocean

```bash
textpa-run-experiment --config configs/default.yaml --ablations all
```

Report:

- Pearson correlation of `textpa_accuracy` with `human_accuracy`.
- Pearson correlation of `textpa_fluency` with `human_fluency`.
- Pearson correlation of `ipa_match` with `human_accuracy`.
- Pearson correlation of `cmu_match` with `human_accuracy`.

### Experiment 2: Textual cue ablation

```bash
textpa-run-experiment --config configs/default.yaml --ablations transcript transcript_ipa transcript_cmu all
```

Expected trend from the paper:

- Transcript alone is weakest.
- IPA improves accuracy.
- CMU with pauses helps fluency.
- All cues plus IPA match scoring is strongest for accuracy.

### Experiment 3: ASR quality ablation

Edit config:

```yaml
models:
  whisper_model: tiny.en
  whisper_hf_model: openai/whisper-tiny.en
```

Run the same ablations and compare with `large-v3`.

## Output interpretation

`results_all_scored.csv` contains:

- `llm_accuracy`, `llm_fluency`: raw Qwen 1-5 scores.
- `ipa_match`: Smith-Waterman similarity before test-set min-max normalization.
- `cmu_match`: optional CMU similarity ablation.
- `llm_accuracy_norm`: min-max-normalized Qwen accuracy.
- `ipa_match_norm`: min-max-normalized IPA match.
- `textpa_accuracy`: final TextPA accuracy prediction.
- `textpa_fluency`: final TextPA fluency prediction.

`results_all_metrics.json` contains Pearson correlations.
