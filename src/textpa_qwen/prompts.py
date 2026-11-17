from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

Ablation = Literal["transcript", "transcript_ipa", "transcript_cmu", "all"]

BASIC_SYSTEM_PROMPT = """You are an expert evaluator of English pronunciation.
Assess the accuracy and fluency of the given text input on a scale of 1 to 5, with higher scores indicating better performance. A score of 5 represents native-speaker-level proficiency.
""".strip()

# Optional detailed guideline. The TextPA paper reports that detailed guidelines are dataset-dependent;
# by default the reproduction uses the basic prompt from Figure 5.
DETAILED_GUIDELINE = """
Scoring guideline:
Accuracy:
5 = Excellent: nearly perfect articulation of all sounds.
4 = Good: minor pronunciation errors, easily understandable.
3 = Fair: noticeable errors but the sentence remains understandable.
2 = Poor: many errors that make parts hard to understand.
1 = Very poor: pronunciation is mostly unintelligible.
Fluency:
5 = Smooth, natural pace with appropriate pauses.
4 = Mostly smooth with small hesitations.
3 = Uneven rhythm or several hesitations, but understandable.
2 = Frequent pauses/repetitions that disrupt communication.
1 = Very halting or fragmented speech.
""".strip()

INPUT_FORMATS: dict[Ablation, str] = {
    "transcript": '{"Transcript": "<Recognized ASR sentence>"}',
    "transcript_ipa": '{"Transcript": "<Recognized ASR sentence>", "Phonemes_IPA": "<Recognized IPA pronouncing phoneme sequence.>"}',
    "transcript_cmu": '{"Transcript": "<Recognized ASR sentence>", "Phonemes_CMU": "<Recognized CMU pronouncing phoneme sequence, with (time.s pause) indicating pauses in speech.>"}',
    "all": '{"Transcript": "<Recognized ASR sentence>", "Phonemes_CMU": "<Recognized CMU pronouncing phoneme sequence, with (time.s pause) indicating pauses in speech.>", "Phonemes_IPA": "<Recognized IPA pronouncing phoneme sequence.>"}',
}

TASK_FORMAT = """Task: Return a dictionary with the following format:
{"Accuracy": <the assessment accuracy score>, "Fluency": <the assessment fluency score>, "Reasoning": <detailed reasoning for the assigned score>}
Note: Do not include any other text other than the json object.
""".strip()

STRICT_JSON_SUFFIX = """
Important for this reproduction: output must be valid JSON. Do not use Markdown. Do not add keys other than Accuracy, Fluency, and Reasoning.
""".strip()


@dataclass
class TextPAPrompt:
    system: str
    user: str


def build_input_object(cues: dict, ablation: Ablation = "all") -> dict[str, str]:
    obj: dict[str, str] = {"Transcript": cues.get("transcript", "")}
    if ablation in ("transcript_ipa", "all"):
        obj["Phonemes_IPA"] = cues.get("phonemes_ipa", "")
    if ablation in ("transcript_cmu", "all"):
        obj["Phonemes_CMU"] = cues.get("phonemes_cmu", "")
    # Keep the key ordering close to the paper's prompt for all features.
    if ablation == "all":
        obj = {
            "Transcript": cues.get("transcript", ""),
            "Phonemes_CMU": cues.get("phonemes_cmu", ""),
            "Phonemes_IPA": cues.get("phonemes_ipa", ""),
        }
    return obj


def build_textpa_prompt(
    cues: dict,
    ablation: Ablation = "all",
    basic: bool = True,
    qwen_strict_json: bool = True,
) -> TextPAPrompt:
    """Construct the TextPA prompt.

    The Figure 5 prompt uses Transcript, Phonemes_CMU, and Phonemes_IPA. Ablation variants
    remove fields but keep the same scoring task.
    """
    system = BASIC_SYSTEM_PROMPT
    if not basic:
        system += "\n" + DETAILED_GUIDELINE
    user_parts = [
        "Input format:",
        INPUT_FORMATS[ablation],
    ]
    user_parts.append(TASK_FORMAT)
    if qwen_strict_json:
        user_parts.append(STRICT_JSON_SUFFIX)
    user_parts.append("Input:")
    user_parts.append(json.dumps(build_input_object(cues, ablation), ensure_ascii=False))
    return TextPAPrompt(system=system, user="\n".join(user_parts))
