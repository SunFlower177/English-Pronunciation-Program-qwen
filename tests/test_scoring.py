import numpy as np
import pandas as pd

from textpa_qwen.qwen_client import run_llm_with_json_retries
from textpa_qwen.data import prepare_local_speechocean762, prepare_multipa
from textpa_qwen.prompts import build_textpa_prompt
from textpa_qwen.scoring import (
    add_textpa_scores,
    fixed_scale_score_norm,
    normalized_sw_similarity,
    phone_tokens,
    smith_waterman_score,
)
from textpa_qwen.speech_cues import PhoneSegment, format_cmu_with_pauses
from textpa_qwen.utils import extract_first_json_object


def test_phone_tokens_remove_pauses():
    seq = "M EH (0.12s pause) B IY"
    assert phone_tokens(seq) == ["M", "EH", "B", "IY"]


def test_smith_waterman_perfect_match():
    a = "m eh b iy".split()
    assert smith_waterman_score(a, a) == 2 * len(a)
    assert normalized_sw_similarity("m eh b iy", "m eh b iy") == 1.0


def test_smith_waterman_partial_match():
    sim = normalized_sw_similarity("m ah m b iy", "m eh b iy")
    assert 0.0 < sim < 1.0


def test_format_cmu_with_pause_gap():
    segs = [PhoneSegment("M", 0.0, 0.1), PhoneSegment("EH", 0.2, 0.3)]
    out = format_cmu_with_pauses(segs, min_pause_s=0.03)
    assert out == "M (0.10s pause) EH"


def test_format_cmu_drops_bracketed_silence():
    segs = [PhoneSegment("[SIL]", 0.0, 0.2), PhoneSegment("M", 0.2, 0.3), PhoneSegment("[SIL]", 0.3, 0.4)]
    out = format_cmu_with_pauses(segs, min_pause_s=0.03)
    assert out == "M"


def test_prompt_all_fields():
    cues = {"transcript": "maybe we should", "phonemes_cmu": "M EY B IY", "phonemes_ipa": "m ey b iy"}
    prompt = build_textpa_prompt(cues, ablation="all")
    assert "Phonemes_CMU" in prompt.user
    assert "Phonemes_IPA" in prompt.user
    assert "Accuracy" in prompt.user


def test_json_extraction_from_markdown():
    obj = extract_first_json_object('```json\n{"Accuracy": 4, "Fluency": 3, "Reasoning": "ok"}\n```')
    assert obj["Accuracy"] == 4


def test_llm_retries_client_exception_then_json():
    class FlakyClient:
        def __init__(self):
            self.calls = 0

        def complete(self, prompt):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary connection issue")
            return '{"Accuracy": 4, "Fluency": 5, "Reasoning": "ok"}'

    prompt = build_textpa_prompt({"transcript": "hello"}, ablation="transcript")
    result = run_llm_with_json_retries(FlakyClient(), prompt, max_retries=2)
    assert result.valid_json
    assert result.attempts == 2
    assert result.accuracy == 4


def test_add_textpa_scores():
    df = pd.DataFrame(
        {
            "phonemes_ipa": ["m ey b iy", "k ae t"],
            "canonical_ipa_from_transcript": ["m ey b iy", "k aa t"],
            "phonemes_cmu": ["M EY B IY", "K AE T"],
            "canonical_cmu_from_transcript": ["M EY B IY", "K AA T"],
            "llm_accuracy": [5, 2],
            "llm_fluency": [5, 3],
            "human_accuracy": [10, 5],
            "human_fluency": [10, 7],
        }
    )
    out = add_textpa_scores(df)
    assert "textpa_accuracy" in out.columns
    assert np.isfinite(out["textpa_accuracy"]).all()
    assert out.loc[0, "llm_accuracy_norm"] == 1.0
    assert out.loc[1, "llm_accuracy_norm"] == 0.25
    assert out.loc[0, "textpa_accuracy"] == 1.0


def test_fixed_scale_norm_does_not_use_batch_max():
    norm = fixed_scale_score_norm(pd.Series([1, 3]))
    assert norm.tolist() == [0.0, 0.5]


def test_prepare_local_speechocean_manifest(tmp_path):
    root = tmp_path / "speechocean762"
    split = root / "test"
    wave_dir = root / "WAVE" / "SPEAKER0003"
    resource = root / "resource"
    split.mkdir(parents=True)
    wave_dir.mkdir(parents=True)
    resource.mkdir(parents=True)
    (wave_dir / "000030012.WAV").write_bytes(b"RIFF")
    (split / "text").write_text("000030012\tMARK IS GOING TO SEE ELEPHANT\n", encoding="utf-8")
    (split / "wav.scp").write_text("000030012\tWAVE/SPEAKER0003/000030012.WAV\n", encoding="utf-8")
    (split / "utt2spk").write_text("000030012 0003\n", encoding="utf-8")
    (split / "spk2gender").write_text("0003 F\n", encoding="utf-8")
    (split / "spk2age").write_text("0003 10\n", encoding="utf-8")
    (resource / "scores.json").write_text(
        '{"000030012": {"text": "MARK IS GOING TO SEE ELEPHANT", "accuracy": 7, "fluency": 8, "prosodic": 6, "total": 7.2}}',
        encoding="utf-8",
    )

    manifest = prepare_local_speechocean762(root, tmp_path / "prepared", split="test")
    df = pd.read_csv(manifest, dtype={"utt_id": str, "speaker": str})

    assert df.loc[0, "utt_id"] == "000030012"
    assert df.loc[0, "human_accuracy"] == 7
    assert df.loc[0, "speaker"] == "0003"


def test_prepare_multipa_manifest_aggregates_annotators(tmp_path):
    root = tmp_path / "MultiPA"
    wav_dir = root / "wav"
    wav_dir.mkdir(parents=True)
    audio = "speaker-a---utt-b.wav"
    (wav_dir / audio).write_bytes(b"RIFF")
    (root / "annotation.csv").write_text(
        "accuracy,annotator,audio,fluency,word-label,prosody\n"
        "\"3 (Fair)\",1,speaker-a---utt-b.wav,\"4 (Mostly Fluent)\",[],\"2 (Limited Prosody)\"\n"
        "\"5 (Excellent)\",2,speaker-a---utt-b.wav,\"2 (Not Very Fluent)\",[],\"4 (Good Prosody)\"\n",
        encoding="utf-8",
    )

    manifest = prepare_multipa(root, tmp_path / "prepared", split="test")
    df = pd.read_csv(manifest, dtype={"utt_id": str, "speaker": str})

    assert df.loc[0, "utt_id"] == "speaker-a---utt-b"
    assert df.loc[0, "human_accuracy"] == 4.0
    assert df.loc[0, "human_fluency"] == 3.0
    assert df.loc[0, "human_prosodic"] == 3.0
    assert df.loc[0, "annotator_count"] == 2
