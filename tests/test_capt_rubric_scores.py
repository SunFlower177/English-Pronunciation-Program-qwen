import copy

from capt_pronunciation.contracts import (
    AlignmentResult,
    EvidenceBundle,
    PhonemeHypothesis,
    ProsodyFeatures,
    ReferenceToken,
    WordAlignment,
)
from capt_pronunciation.deterministic_scoring import (
    DEFAULT_SCORE_CONFIG,
    score_evidence,
)
from capt_pronunciation.feedback import compute_scores


def _alignment(words, operations=None):
    operations = operations or ["equal"] * len(words)
    return AlignmentResult(
        reference_tokens=[
            ReferenceToken(index=index, raw=word, normalized=word.lower())
            for index, word in enumerate(words)
        ],
        word_alignments=[
            WordAlignment(
                ref_index=index,
                ref_word=word,
                hyp_word=word if operation == "equal" else ("word" if operation == "substitute" else None),
                start=index * 0.4,
                end=index * 0.4 + 0.3,
                operation=operation,
                confidence=0.9,
            )
            for index, (word, operation) in enumerate(zip(words, operations))
        ],
    )


def _natural_prosody():
    return ProsodyFeatures(
        speech_rate_wpm=145.0,
        articulation_rate_wpm=175.0,
        pause_ratio=0.10,
        mean_length_run_words=9.0,
        npvi_word_duration=48.0,
        word_duration_cv=0.38,
        pitch_range_st=7.0,
    )


def test_missing_evidence_is_neutral_not_perfect():
    evidence = EvidenceBundle()
    scores = compute_scores(evidence, alignment=AlignmentResult(reference_tokens=[], word_alignments=[]))

    assert scores == {
        "accuracy": 50.0,
        "phoneme": 50.0,
        "fluency": 50.0,
        "prosody": 50.0,
        "overall": 50.0,
    }


def test_continuous_word_alignment_preserves_substitution_similarity():
    clean = EvidenceBundle(prosody=_natural_prosody())
    degraded = EvidenceBundle(prosody=_natural_prosody())
    clean_score = compute_scores(clean, alignment=_alignment(["hello", "world"]))
    degraded_score = compute_scores(
        degraded,
        alignment=_alignment(["hello", "world"], ["equal", "substitute"]),
    )

    assert clean_score["accuracy"] > degraded_score["accuracy"]
    assert 0.0 < degraded_score["accuracy"] < 100.0


def test_phone_coverage_blends_with_word_evidence_instead_of_assuming_missing_is_correct():
    words = ["good", "morning", "to", "everyone"]
    alignment = _alignment(words, ["equal", "equal", "substitute", "delete"])
    no_phone = EvidenceBundle(prosody=_natural_prosody())
    partial_phone = EvidenceBundle(
        prosody=_natural_prosody(),
        phoneme_hypotheses=[
            PhonemeHypothesis(
                ref_index=0,
                word="good",
                canonical_phones=["g", "u", "d"],
                predicted_phones=["g", "u", "d"],
                confidence=0.9,
                ctc_gop=0.8,
            )
        ],
    )

    baseline = compute_scores(no_phone, alignment=alignment)
    partial = compute_scores(partial_phone, alignment=alignment)

    assert partial["phoneme"] > baseline["phoneme"]
    assert partial["phoneme"] < 100.0


def test_global_phone_edit_distance_penalizes_bad_tails():
    alignment = _alignment(["example"])
    good = EvidenceBundle(
        phoneme_hypotheses=[
            PhonemeHypothesis(
                ref_index=0,
                word="example",
                canonical_phones=["e", "g", "z", "a", "m", "p", "l"],
                predicted_phones=["e", "g", "z", "a", "m", "p", "l"],
                confidence=0.9,
            )
        ],
        prosody=_natural_prosody(),
    )
    bad_tail = EvidenceBundle(
        phoneme_hypotheses=[
            PhonemeHypothesis(
                ref_index=0,
                word="example",
                canonical_phones=["e", "g", "z", "a", "m", "p", "l"],
                predicted_phones=["e", "g", "z"],
                confidence=0.9,
            )
        ],
        prosody=_natural_prosody(),
    )

    assert compute_scores(good, alignment=alignment)["phoneme"] > compute_scores(
        bad_tail,
        alignment=alignment,
    )["phoneme"]


def test_continuous_fluency_and_prosody_features_order_examples():
    alignment = _alignment([f"w{index}" for index in range(12)])
    natural = EvidenceBundle(prosody=_natural_prosody())
    hesitant = EvidenceBundle(
        prosody=ProsodyFeatures(
            speech_rate_wpm=62.0,
            articulation_rate_wpm=82.0,
            pause_ratio=0.42,
            mean_length_run_words=1.4,
            npvi_word_duration=12.0,
            word_duration_cv=1.05,
            pitch_range_st=1.7,
        )
    )

    natural_score = compute_scores(natural, alignment=alignment)
    hesitant_score = compute_scores(hesitant, alignment=alignment)
    assert natural_score["fluency"] > hesitant_score["fluency"]
    assert natural_score["prosody"] > hesitant_score["prosody"]


def test_mode_b_uses_continuous_reference_distances():
    alignment = _alignment(["pick", "it", "up"])
    close = EvidenceBundle(
        continuous_features={
            "mode_b": {
                "phone_reference_similarity": 0.92,
                "phone_canonical_similarity": 0.88,
                "tempo_log_ratio": 0.03,
                "pause_mae_s": 0.02,
                "duration_mae": 0.08,
                "f0_dtw": 0.20,
                "f0_correlation": 0.86,
                "sentence_stress_overlap": 0.80,
                "word_stress_similarity": 0.90,
                "ssl_soft_dtw": 0.12,
            }
        }
    )
    distant = EvidenceBundle(
        continuous_features={
            "mode_b": {
                "phone_reference_similarity": 0.35,
                "phone_canonical_similarity": 0.42,
                "tempo_log_ratio": 0.70,
                "pause_mae_s": 0.45,
                "duration_mae": 0.90,
                "f0_dtw": 2.30,
                "f0_correlation": -0.20,
                "sentence_stress_overlap": 0.15,
                "word_stress_similarity": 0.25,
                "ssl_soft_dtw": 0.90,
            }
        }
    )

    close_score = score_evidence(close, alignment, mode="modeB")
    distant_score = score_evidence(distant, alignment, mode="modeB")
    for metric in ("accuracy", "phoneme", "fluency", "prosody", "overall"):
        assert close_score.scores[metric] > distant_score.scores[metric]


def test_frozen_yaml_weights_can_change_feature_fusion_without_code_changes():
    alignment = _alignment(["hello", "world"])
    evidence = EvidenceBundle(prosody=_natural_prosody())
    default = score_evidence(evidence, alignment)
    changed_config = copy.deepcopy(DEFAULT_SCORE_CONFIG)
    changed_config["mode_a"]["accuracy_weights"] = {"word_alignment": 1.0}
    changed = score_evidence(evidence, alignment, config=changed_config)

    assert changed.scores["accuracy"] == 100.0
    assert default.config_version == "continuous_v3_alignment_gop"


def test_fluency_transform_does_not_plateau_or_penalize_fast_clear_speech():
    alignment = _alignment([f"w{index}" for index in range(20)])
    moderate = EvidenceBundle(
        prosody=ProsodyFeatures(
            speech_rate_wpm=120.0,
            articulation_rate_wpm=170.0,
            pause_ratio=0.08,
            mean_length_run_words=8.0,
            raw={"long_pause_ratio": 0.03, "filled_pause_rate_per_100_words": 0.0},
        )
    )
    fast_clear = EvidenceBundle(
        prosody=ProsodyFeatures(
            speech_rate_wpm=175.0,
            articulation_rate_wpm=240.0,
            pause_ratio=0.04,
            mean_length_run_words=14.0,
            raw={"long_pause_ratio": 0.01, "filled_pause_rate_per_100_words": 0.0},
        )
    )

    assert compute_scores(fast_clear, alignment=alignment)["fluency"] > compute_scores(
        moderate, alignment=alignment
    )["fluency"]
