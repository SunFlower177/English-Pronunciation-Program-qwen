# Architecture: ProsodyLink-TextPA Mode A

## Goal

Mode A assumes the learner reads a known reference text. The system can therefore compare the learner audio against canonical words, phones, stress, and expected phrase boundaries. The LLM is not the primary judge; it is only a feedback verbalizer.

## Pipeline

```text
reference text + audio
  -> audio_io.prepare_audio
  -> transcription.WhisperXTranscriber
  -> reference_alignment.AlignmentEngine
       - tokenize reference
       - CMUdict/G2P -> ARPAbet/IPA/stress
       - DP word alignment with ASR
       - optional MFA phone boundaries
       - fallback estimated phone boundaries
  -> diagnostics.EvidenceAnalyzer
       - TextPA cues and IPA match score
       - word/phoneme evidence
       - pause boundary evidence
       - linking opportunity evidence
       - lexical stress evidence
       - sentence stress evidence
       - intonation evidence
       - rhythm evidence
  -> feedback.EvidenceGroundedFeedbackGenerator
       - template fallback or Qwen text/Qwen2-Audio
       - unknown issue_id outputs discarded
  -> report.json + annotated HTML
```

## Evidence design

Every diagnosis is stored as `EvidenceIssue`:

```json
{
  "issue_id": "I0007",
  "type": "linking_missing",
  "severity": "low",
  "word": "pick it",
  "observed": "gap=0.31s",
  "expected": "consonant-vowel linking k+ɪ",
  "start": 1.42,
  "end": 1.73,
  "confidence": 0.81,
  "evidence": {"gap_s": 0.31, "type": "c_v_linking"},
  "suggestion_hint": "..."
}
```

The Qwen prompt includes only this JSON and explicitly forbids inventing new errors. `_merge_llm_with_template()` discards any generated feedback item whose `issue_id` does not exist in the evidence.

## Innovation modules

### Pause

Pause is computed from gaps between aligned words. Each boundary is classified as `none`, `phrase`, `clause`, or `sentence` using punctuation and lightweight lexical rules. The system flags:

- missing pause after phrase/clause/sentence boundary
- unexpected pause inside phrase
- overly long pause at a legal boundary

### Linking

The system first detects linking opportunities, then uses the acoustic gap to judge whether the boundary sounds overly separated. Supported types:

- C-V linking: `pick it`, `read a`
- V-V glide: `see it`, `go out`
- r-linking: `far away`
- same consonant smoothing: `big game`
- weak-form chunks: `want to`, `going to`, `kind of`

Linking findings should be treated as naturalness suggestions.

### Lexical stress

Canonical stress comes from CMUdict digits. Observed prominence is computed for syllable windows using:

- duration z-score
- intensity z-score
- F0 z-score

The system flags words where the expected primary syllable is not the most prominent or the contrast is weak.

### Sentence stress

The system compares mean intensity of content words and function words. If content words are not more prominent, it outputs a sentence-stress naturalness issue.

### Intonation

Pitch is extracted by Parselmouth when available, otherwise librosa.yin. It is normalized around speaker median F0 into semitones. The system checks:

- final slope for statements and questions
- sentence-level pitch range for monotone risk

### Rhythm

The system computes:

- speech rate
- articulation rate
- pause ratio
- mean length of run in words
- word-duration nPVI
- word-duration coefficient of variation

Rhythm issues are global, so they may not have `ref_index`.

## Heavy dependencies

- WhisperX/faster-whisper are needed for real ASR.
- MFA is optional but recommended for phone boundaries.
- Wav2Vec2 phoneme CTC is optional but recommended for phone-level diagnosis.
- Qwen text API is optional; template feedback is deterministic and useful for tests.

## Reproducibility notes

Set these for deterministic development:

```bash
CAPT_ASR_BACKEND=transformers_whisper
CAPT_TRANSFORMERS_WHISPER_MODEL=openai/whisper-tiny.en
CAPT_USE_MFA=0
CAPT_USE_PHONEME_MODEL=0
CAPT_USE_QWEN_TEXT=0
CAPT_WHISPERX_MODEL=tiny.en
```

Use this for paper experiments:

```bash
CAPT_DEVICE=cuda
CAPT_COMPUTE_TYPE=float16
CAPT_WHISPERX_MODEL=large-v3
CAPT_USE_MFA=1
CAPT_USE_PHONEME_MODEL=1
```
