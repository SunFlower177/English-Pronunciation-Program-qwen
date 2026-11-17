# Codex Guide

## Task boundary

This repository implements only **Mode A: read-aloud with reference text**. Do not add free-speaking assumptions to this pipeline. Mode B should be implemented as a separate analyzer because word/phoneme error diagnosis is not reliable without a reference.

## Where to modify

- Add new acoustic features in `capt_pronunciation/audio_io.py`.
- Add new reference text or G2P logic in `capt_pronunciation/reference_alignment.py`.
- Add new evidence detectors in `capt_pronunciation/diagnostics.py`.
- Add new feedback wording or LLM prompt logic in `capt_pronunciation/feedback.py`.
- Add API endpoints or CLI arguments in `capt_pronunciation/runtime.py`.
- Add metrics in `capt_pronunciation/evaluation.py`.

## Contract rules

1. Any diagnostic output must be an `EvidenceIssue`.
2. Every issue should include `observed`, `expected`, `confidence`, and a structured `evidence` dictionary.
3. Do not let Qwen invent issues. The feedback generator must discard unknown `issue_id`s.
4. Global rhythm/intonation issues may have `ref_index=None`; word-level issues should provide `ref_index`.
5. Linking and sentence-stress issues are naturalness suggestions, not absolute errors.

## Adding a new detector

1. Add an `IssueType` in `contracts.py`.
2. Implement `detect_xxx_issues()` in `diagnostics.py`.
3. Call it from `EvidenceAnalyzer.analyze()`.
4. Add score penalties in `feedback.compute_scores()`.
5. Add tests with synthetic `AlignmentResult` and tracks.

## Testing

```bash
pytest -q
```

Core tests do not require WhisperX, MFA, Wav2Vec2, or Qwen. Do not make unit tests download models.

## API

The Web app calls:

```http
POST /api/analyze-reading
FormData: reference_text, audio
```

Response is a complete `AnalysisReport` JSON.
