"""Reference text processing, CMUdict/G2P phonemization, and alignment.

Functional module 3/7.

This module joins three jobs that are tightly coupled in a reading-aloud CAPT
system:
1. Normalize the reference text while preserving punctuation cues.
2. Convert each reference word to canonical phonemes using CMUdict/G2P.
3. Align reference words to ASR words and optionally enrich boundaries with MFA.
"""
from __future__ import annotations

import difflib
import json
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .contracts import (
    AlignmentResult,
    AsrResult,
    AsrWord,
    CaptSettings,
    PhoneInterval,
    ReferenceToken,
    WordAlignment,
)

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)?|[0-9]+|[^\w\s]", re.UNICODE)
_VOWEL_ARPA = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
_VOWEL_IPA = {"ɑ", "æ", "ʌ", "ɔ", "aʊ", "aɪ", "ɛ", "ɝ", "eɪ", "ɪ", "i", "oʊ", "ɔɪ", "ʊ", "u", "ə"}

ARPABET_TO_IPA = {
    "AA": "ɑ",
    "AE": "æ",
    "AH": "ʌ",
    "AO": "ɔ",
    "AW": "aʊ",
    "AY": "aɪ",
    "B": "b",
    "CH": "tʃ",
    "D": "d",
    "DH": "ð",
    "EH": "ɛ",
    "ER": "ɝ",
    "EY": "eɪ",
    "F": "f",
    "G": "ɡ",
    "HH": "h",
    "IH": "ɪ",
    "IY": "i",
    "JH": "dʒ",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "ŋ",
    "OW": "oʊ",
    "OY": "ɔɪ",
    "P": "p",
    "R": "r",
    "S": "s",
    "SH": "ʃ",
    "T": "t",
    "TH": "θ",
    "UH": "ʊ",
    "UW": "u",
    "V": "v",
    "W": "w",
    "Y": "j",
    "Z": "z",
    "ZH": "ʒ",
}

# Small fallback lexicon keeps examples/tests useful without installing cmudict.
MINI_LEXICON = {
    "a": ["AH0"],
    "an": ["AE1", "N"],
    "and": ["AH0", "N", "D"],
    "apple": ["AE1", "P", "AH0", "L"],
    "english": ["IH1", "NG", "G", "L", "IH0", "SH"],
    "hello": ["HH", "AH0", "L", "OW1"],
    "i": ["AY1"],
    "improve": ["IH2", "M", "P", "R", "UW1", "V"],
    "is": ["IH1", "Z"],
    "it": ["IH1", "T"],
    "my": ["M", "AY1"],
    "pick": ["P", "IH1", "K"],
    "pronunciation": ["P", "R", "AH0", "N", "AH2", "N", "S", "IY0", "EY1", "SH", "AH0", "N"],
    "quickly": ["K", "W", "IH1", "K", "L", "IY0"],
    "read": ["R", "IY1", "D"],
    "the": ["DH", "AH0"],
    "think": ["TH", "IH1", "NG", "K"],
    "to": ["T", "UW1"],
    "up": ["AH1", "P"],
    "want": ["W", "AA1", "N", "T"],
    "world": ["W", "ER1", "L", "D"],
}


class ReferenceTextProcessor:
    def __init__(self, target_accent: str = "american"):
        self.target_accent = target_accent
        self.phonemizer = CmuPhonemizer()

    def tokenize(self, text: str) -> List[ReferenceToken]:
        pieces = _WORD_RE.findall(text)
        tokens: List[ReferenceToken] = []
        pending_before = ""
        sentence_index = 0
        for piece in pieces:
            if re.match(r"[A-Za-z0-9]", piece):
                norm = normalize_word(piece)
                if not norm:
                    continue
                phones = self.phonemizer.phones_for_word(norm)
                canonical_ipa = arpabet_to_ipa(phones)
                stress = stress_pattern_from_arpabet(phones)
                pos_hint = guess_pos_hint(norm)
                tokens.append(
                    ReferenceToken(
                        index=len(tokens),
                        raw=piece,
                        normalized=norm,
                        before_punct=pending_before,
                        sentence_index=sentence_index,
                        canonical_arpabet=phones,
                        canonical_ipa=canonical_ipa,
                        stress_pattern=stress,
                        syllable_count=count_syllables_from_arpabet(phones),
                        pos_hint=pos_hint,
                        is_content_word=is_content_word(norm, pos_hint),
                    )
                )
                pending_before = ""
            else:
                if tokens:
                    tokens[-1].after_punct += piece
                    if piece in ".!?":
                        sentence_index += 1
                else:
                    pending_before += piece
        self._assign_boundary_types(tokens)
        return tokens

    def _assign_boundary_types(self, tokens: List[ReferenceToken]) -> None:
        for i, tok in enumerate(tokens):
            punct = tok.after_punct or ""
            if any(ch in punct for ch in ".!?"):
                tok.boundary_after = "sentence"
            elif any(ch in punct for ch in ";:"):
                tok.boundary_after = "clause"
            elif "," in punct:
                tok.boundary_after = "phrase"
            elif i + 1 < len(tokens):
                nxt = tokens[i + 1]
                # Heuristic phrase break: content word followed by a new clause marker/conjunction/preposition.
                if tok.is_content_word and nxt.normalized in {"and", "but", "or", "because", "when", "while", "if", "although"}:
                    tok.boundary_after = "clause"
                elif tok.is_content_word and nxt.normalized in {"in", "on", "at", "for", "with", "from", "to", "by", "about", "after", "before"}:
                    tok.boundary_after = "phrase"
                else:
                    tok.boundary_after = "none"
            else:
                tok.boundary_after = "sentence"

FUNCTION_WORDS = {
    "a", "an", "the", "to", "of", "in", "on", "at", "for", "with", "from", "by", "about", "as", "and",
    "or", "but", "if", "when", "while", "because", "that", "this", "these", "those", "my", "your", "his", "her",
    "our", "their", "i", "you", "he", "she", "it", "we", "they", "me", "him", "them", "is", "am", "are",
    "was", "were", "be", "been", "being", "do", "does", "did", "have", "has", "had", "will", "would",
    "can", "could", "should", "may", "might", "must", "not", "no", "yes"
}

PREPOSITIONS = {"in", "on", "at", "for", "with", "from", "by", "about", "as", "into", "onto", "over", "under", "after", "before", "between", "through"}
CONJUNCTIONS = {"and", "or", "but", "because", "although", "when", "while", "if", "so"}
PRONOUNS = {"i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "our", "their"}
AUXILIARIES = {"is", "am", "are", "was", "were", "be", "been", "being", "do", "does", "did", "have", "has", "had", "will", "would", "can", "could", "should", "may", "might", "must"}


def guess_pos_hint(word: str) -> str:
    w = normalize_word(word)
    if w in PREPOSITIONS:
        return "prep"
    if w in CONJUNCTIONS:
        return "conj"
    if w in PRONOUNS:
        return "pron"
    if w in AUXILIARIES:
        return "aux"
    if w in {"a", "an", "the"}:
        return "det"
    if w.endswith("ly"):
        return "adv"
    if w.endswith(("ing", "ed", "ize", "ise")):
        return "verb"
    if w.endswith(("tion", "sion", "ment", "ness", "ity", "ship")):
        return "noun"
    if w.endswith(("ous", "ful", "able", "ible", "ive", "al", "ic")):
        return "adj"
    return "content"


def is_content_word(word: str, pos_hint: str = "") -> bool:
    w = normalize_word(word)
    return bool(w and w not in FUNCTION_WORDS and pos_hint not in {"det", "prep", "conj", "pron", "aux"})


def count_syllables_from_arpabet(phones: Sequence[str]) -> int:
    count = 0
    for ph in phones:
        base = re.sub(r"[0-2]", "", ph.upper())
        if base in _VOWEL_ARPA:
            count += 1
    return count


class CmuPhonemizer:
    """CMUdict-first phonemizer with g2p fallback."""

    def __init__(self):
        self._dict: Dict[str, List[str]] = {}
        self._g2p = None
        self._load_cmudict()

    def _load_cmudict(self) -> None:
        try:
            import cmudict  # type: ignore

            raw = cmudict.dict()
            self._dict = {k.lower(): list(v[0]) for k, v in raw.items() if v}
            logger.info("Loaded cmudict package with %d entries", len(self._dict))
            return
        except Exception:
            pass
        try:
            import pronouncing  # type: ignore

            # Lazy dictionary using pronouncing.phones_for_word inside phones_for_word.
            self._dict = {}
            self._pronouncing = pronouncing  # type: ignore[attr-defined]
            logger.info("Using pronouncing package as CMUdict provider")
            return
        except Exception:
            self._pronouncing = None  # type: ignore[attr-defined]
        self._dict = dict(MINI_LEXICON)
        logger.warning("Using MINI_LEXICON fallback. Install cmudict for real experiments.")

    def phones_for_word(self, word: str) -> List[str]:
        word = normalize_word(word)
        if not word:
            return []
        if word in self._dict:
            return list(self._dict[word])
        pronouncing = getattr(self, "_pronouncing", None)
        if pronouncing is not None:
            phones = pronouncing.phones_for_word(word)
            if phones:
                return phones[0].split()
        # g2p_en fallback often outputs ARPAbet plus spaces.
        try:
            if self._g2p is None:
                from g2p_en import G2p  # type: ignore

                self._g2p = G2p()
            out = [x for x in self._g2p(word) if x.strip() and re.search(r"[A-Z]", x)]
            if out:
                return out
        except Exception:
            pass
        # Final rough fallback: letters as pseudo-phones, marked as low quality by diagnostics.
        return [ch.upper() for ch in word if ch.isalpha()]


class AlignmentEngine:
    def __init__(self, settings: CaptSettings):
        self.settings = settings
        self.reference_processor = ReferenceTextProcessor(settings.target_accent)
        self.mfa = MfaAligner(settings)
        self.ctc = CtcForcedAligner(settings)

    def align(self, reference_text: str, asr: AsrResult, audio_path: str | Path, analysis_id: str) -> AlignmentResult:
        tokens = self.reference_processor.tokenize(reference_text)
        word_alignments = self.align_words(tokens, asr.words)
        source = "asr_dp"
        raw: Dict[str, Any] = {"asr_word_count": len(asr.words), "reference_word_count": len(tokens)}

        if self.settings.use_mfa:
            try:
                mfa_result = self.mfa.align(reference_text, audio_path, analysis_id)
                if mfa_result.get("words") or mfa_result.get("phones"):
                    self._merge_mfa_intervals(tokens, word_alignments, mfa_result)
                    source = "asr_dp+mfa"
                    raw["mfa"] = {"word_count": len(mfa_result.get("words", [])), "phone_count": len(mfa_result.get("phones", []))}
            except Exception as exc:
                logger.warning("MFA alignment failed, falling back to estimated phones: %s", exc)
                raw["mfa_error"] = str(exc)

        if self.settings.use_ctc_alignment and "mfa" not in source:
            try:
                ctc_result = self.ctc.align(tokens, audio_path)
                if ctc_result.get("words"):
                    self._merge_forced_word_intervals(word_alignments, ctc_result)
                    source = "asr_dp+ctc_fa"
                    raw["ctc_fa"] = {"word_count": len(ctc_result.get("words", []))}
            except Exception as exc:
                logger.warning("CTC forced alignment failed, falling back to ASR timestamps: %s", exc)
                raw["ctc_fa_error"] = str(exc)

        self._estimate_missing_phone_intervals(tokens, word_alignments)
        return AlignmentResult(reference_tokens=tokens, word_alignments=word_alignments, source=source, raw=raw)

    def align_words(self, reference_tokens: List[ReferenceToken], asr_words: List[AsrWord]) -> List[WordAlignment]:
        ref = [t.normalized for t in reference_tokens]
        hyp = [normalize_word(w.word) for w in asr_words]
        # Dynamic programming alignment.
        n, m = len(ref), len(hyp)
        dp = [[0.0] * (m + 1) for _ in range(n + 1)]
        back: List[List[Tuple[int, int, str]]] = [[(0, 0, "")] * (m + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            dp[i][0] = i
            back[i][0] = (i - 1, 0, "delete")
        for j in range(1, m + 1):
            dp[0][j] = j
            back[0][j] = (0, j - 1, "insert")
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if ref[i - 1] == hyp[j - 1]:
                    sub_cost = 0.0
                    op = "equal"
                else:
                    ratio = difflib.SequenceMatcher(None, ref[i - 1], hyp[j - 1]).ratio()
                    sub_cost = 0.65 if ratio > 0.78 else 1.0
                    op = "substitute"
                candidates = [
                    (dp[i - 1][j - 1] + sub_cost, i - 1, j - 1, op),
                    (dp[i - 1][j] + 1.0, i - 1, j, "delete"),
                    (dp[i][j - 1] + 1.0, i, j - 1, "insert"),
                ]
                best = min(candidates, key=lambda x: x[0])
                dp[i][j] = best[0]
                back[i][j] = (best[1], best[2], best[3])
        pairs: List[Tuple[Optional[int], Optional[int], str]] = []
        i, j = n, m
        while i > 0 or j > 0:
            pi, pj, op = back[i][j]
            if op in {"equal", "substitute"}:
                pairs.append((i - 1, j - 1, op))
            elif op == "delete":
                pairs.append((i - 1, None, op))
            elif op == "insert":
                pairs.append((None, j - 1, op))
            i, j = pi, pj
        pairs.reverse()

        alignments: List[WordAlignment] = []
        for ref_i, hyp_j, op in pairs:
            ref_word = reference_tokens[ref_i].normalized if ref_i is not None else None
            hyp_word = asr_words[hyp_j].word.strip() if hyp_j is not None else None
            start = asr_words[hyp_j].start if hyp_j is not None else None
            end = asr_words[hyp_j].end if hyp_j is not None else None
            conf = asr_words[hyp_j].score if hyp_j is not None else None
            alignments.append(
                WordAlignment(
                    ref_index=ref_i,
                    ref_word=ref_word,
                    hyp_word=hyp_word,
                    start=start,
                    end=end,
                    operation=op,  # type: ignore[arg-type]
                    confidence=conf,
                    phones=[],
                )
            )
        return alignments

    def _merge_mfa_intervals(self, tokens: List[ReferenceToken], alignments: List[WordAlignment], mfa_result: Dict[str, List[Dict[str, Any]]]) -> None:
        mfa_words = [x for x in mfa_result.get("words", []) if normalize_word(str(x.get("label", "")))]
        mfa_phones = [x for x in mfa_result.get("phones", []) if str(x.get("label", "")).strip()]
        # Map MFA word intervals to reference indices by normalized sequence order.
        ref_i = 0
        mfa_map: Dict[int, Dict[str, Any]] = {}
        for w in mfa_words:
            label = normalize_word(str(w.get("label", "")))
            while ref_i < len(tokens) and tokens[ref_i].normalized != label:
                ref_i += 1
            if ref_i < len(tokens):
                mfa_map[ref_i] = w
                ref_i += 1
        for ali in alignments:
            if ali.ref_index is None or ali.ref_index not in mfa_map:
                continue
            w = mfa_map[ali.ref_index]
            # Do not assign forced timings to words that ASR believes were deleted.
            if ali.operation != "delete":
                ali.start = _safe_float(w.get("start"))
                ali.end = _safe_float(w.get("end"))
            word_start = _safe_float(w.get("start"))
            word_end = _safe_float(w.get("end"))
            if word_start is None or word_end is None:
                continue
            for ph in mfa_phones:
                ps = _safe_float(ph.get("start"))
                pe = _safe_float(ph.get("end"))
                label = str(ph.get("label", "")).strip()
                if not label or ps is None or pe is None:
                    continue
                if ps >= word_start - 0.02 and pe <= word_end + 0.02:
                    ali.phones.append(PhoneInterval(phone=label, start=ps, end=pe, source="mfa"))

    def _merge_forced_word_intervals(self, alignments: List[WordAlignment], forced_result: Dict[str, List[Dict[str, Any]]]) -> None:
        forced_words = {int(x["ref_index"]): x for x in forced_result.get("words", []) if x.get("ref_index") is not None}
        for ali in alignments:
            if ali.ref_index is None or ali.ref_index not in forced_words or ali.operation == "delete":
                continue
            w = forced_words[ali.ref_index]
            ali.start = _safe_float(w.get("start"))
            ali.end = _safe_float(w.get("end"))
            # Keep ASR lexical confidence.  The MMS alignment score is a
            # frame/token posterior on a different scale and must not be used
            # as a word-recognition confidence gate for phoneme diagnostics.
            ali.phones = []

    def _estimate_missing_phone_intervals(self, tokens: List[ReferenceToken], alignments: List[WordAlignment]) -> None:
        for ali in alignments:
            if ali.ref_index is None or ali.phones or ali.start is None or ali.end is None or ali.operation == "delete":
                continue
            phones = tokens[ali.ref_index].canonical_ipa or arpabet_to_ipa(tokens[ali.ref_index].canonical_arpabet)
            if not phones:
                continue
            dur = max(0.001, ali.end - ali.start)
            step = dur / len(phones)
            ali.phones = [
                PhoneInterval(phone=p, start=ali.start + i * step, end=ali.start + (i + 1) * step, source="estimated")
                for i, p in enumerate(phones)
            ]


class CtcForcedAligner:
    """Reference-guided CTC word alignment using torchaudio MMS_FA.

    This is a practical fallback when MFA is unavailable. It yields reliable
    reference word timestamps, then phone intervals are still estimated within
    each forced word span.
    """

    def __init__(self, settings: CaptSettings):
        self.settings = settings
        self._bundle = None
        self._model = None
        self._tokenizer = None
        self._aligner = None

    def align(self, tokens: List[ReferenceToken], audio_path: str | Path) -> Dict[str, List[Dict[str, Any]]]:
        import torch  # type: ignore
        import torchaudio  # type: ignore

        self._ensure_loaded()
        assert self._bundle is not None and self._model is not None and self._tokenizer is not None and self._aligner is not None
        words: List[str] = []
        ref_indices: List[int] = []
        label_dict = self._bundle.get_dict()
        allowed = set(label_dict.keys())
        for token in tokens:
            word = token.normalized
            if not word or any(ch not in allowed for ch in word):
                continue
            words.append(word)
            ref_indices.append(token.index)
        if not words:
            return {"words": [], "phones": []}

        waveform, sample_rate = torchaudio.load(str(audio_path))
        if waveform.ndim > 1 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if int(sample_rate) != int(self._bundle.sample_rate):
            waveform = torchaudio.functional.resample(waveform, int(sample_rate), int(self._bundle.sample_rate))
        device = self.settings.device if self.settings.device == "cuda" and torch.cuda.is_available() else "cpu"
        waveform = waveform.to(device)
        with torch.inference_mode():
            emission, _ = self._model(waveform)
        tokenized = self._tokenizer(words)
        spans = self._aligner(emission[0].cpu(), tokenized)
        frame_to_s = waveform.shape[1] / max(1, emission.shape[1]) / float(self._bundle.sample_rate)
        out_words: List[Dict[str, Any]] = []
        for ref_index, word, group in zip(ref_indices, words, spans):
            if not group:
                continue
            start = float(group[0].start * frame_to_s)
            end = float(group[-1].end * frame_to_s)
            if end <= start:
                continue
            score = float(sum(s.score for s in group) / max(1, len(group)))
            out_words.append({"ref_index": ref_index, "label": word, "start": start, "end": end, "score": score})
        return {"words": out_words, "phones": []}

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch  # type: ignore
        import torchaudio  # type: ignore

        self._bundle = torchaudio.pipelines.MMS_FA
        self._tokenizer = self._bundle.get_tokenizer()
        self._aligner = self._bundle.get_aligner()
        self._model = self._bundle.get_model()
        device = self.settings.device if self.settings.device == "cuda" and torch.cuda.is_available() else "cpu"
        self._model.to(device)
        self._model.eval()


class MfaAligner:
    """Thin wrapper around the Montreal Forced Aligner command-line tool."""

    def __init__(self, settings: CaptSettings):
        self.settings = settings

    def align(self, reference_text: str, audio_path: str | Path, analysis_id: str) -> Dict[str, List[Dict[str, Any]]]:
        mfa_command = resolve_mfa_command(self.settings.mfa_command)
        if mfa_command is None:
            raise RuntimeError("MFA command not found. Install montreal-forced-aligner and download English models.")
        run_dir = self.settings.work_dir / analysis_id / "mfa"
        corpus_dir = run_dir / "corpus"
        output_dir = run_dir / "aligned"
        corpus_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        # MFA requires paired audio/transcript with matching basename.
        wav_path = corpus_dir / "utt.wav"
        txt_path = corpus_dir / "utt.txt"
        shutil.copy2(audio_path, wav_path)
        txt_path.write_text(reference_text, encoding="utf-8")

        cmd = [
            mfa_command,
            "align",
            str(corpus_dir),
            self.settings.mfa_dictionary,
            self.settings.mfa_acoustic_model,
            str(output_dir),
            "--clean",
            "--overwrite",
            "--output_format",
            "json",
            "--single_speaker",
            "-j",
            str(self.settings.mfa_num_jobs),
        ]
        if self.settings.mfa_g2p_model:
            cmd.extend(["--g2p_model_path", self.settings.mfa_g2p_model])
        logger.info("Running MFA: %s", " ".join(cmd))
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"MFA failed with code {proc.returncode}\nSTDOUT:\n{proc.stdout[-2000:]}\nSTDERR:\n{proc.stderr[-2000:]}")
        parsed = self._parse_mfa_output(output_dir)
        if not parsed.get("words") and not parsed.get("phones"):
            raise RuntimeError(f"MFA completed but no word/phone intervals were parsed from {output_dir}")
        return parsed

    def _parse_mfa_output(self, output_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
        json_files = list(output_dir.rglob("*.json"))
        for path in json_files:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                parsed = parse_mfa_json(data)
                if parsed["words"] or parsed["phones"]:
                    return parsed
            except Exception as exc:
                logger.warning("Failed to parse MFA json %s: %s", path, exc)
        # Fallback to TextGrid if json output changed or MFA ignored output_format.
        tg_files = list(output_dir.rglob("*.TextGrid"))
        for path in tg_files:
            try:
                parsed = parse_textgrid(path)
                if parsed["words"] or parsed["phones"]:
                    return parsed
            except Exception as exc:
                logger.warning("Failed to parse TextGrid %s: %s", path, exc)
        return {"words": [], "phones": []}


def parse_mfa_json(data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    words: List[Dict[str, Any]] = []
    phones: List[Dict[str, Any]] = []

    def add_interval(container: List[Dict[str, Any]], label: Any, start: Any, end: Any) -> None:
        label = str(label or "").strip()
        if not label or label.lower() in {"sp", "sil", "<eps>"}:
            return
        s = _safe_float(start)
        e = _safe_float(end)
        if s is None or e is None or e <= s:
            return
        container.append({"label": label, "start": s, "end": e})

    # Known-ish MFA JSON variants.
    if "tiers" in data:
        tiers = data.get("tiers") or []
        if isinstance(tiers, dict):
            tiers = [{"name": k, **(v if isinstance(v, dict) else {"entries": v})} for k, v in tiers.items()]
        for tier in tiers:
            name = str(tier.get("name") or tier.get("tier_name") or "").lower()
            entries = tier.get("entries") or tier.get("intervals") or []
            for it in entries:
                label = it.get("label", it.get("text", it.get("mark", ""))) if isinstance(it, dict) else ""
                start = it.get("start", it.get("xmin", it.get("begin"))) if isinstance(it, dict) else None
                end = it.get("end", it.get("xmax", it.get("finish"))) if isinstance(it, dict) else None
                if "phone" in name:
                    add_interval(phones, label, start, end)
                elif "word" in name or "transcript" in name:
                    add_interval(words, label, start, end)
    # Recursive search for dicts with label/start/end and tier hints.
    def walk(obj: Any, hint: str = "") -> None:
        if isinstance(obj, dict):
            keys = {str(k).lower() for k in obj.keys()}
            local_hint = hint
            if any(k in keys for k in ["phones", "phone_intervals"]):
                local_hint = "phone"
            if any(k in keys for k in ["words", "word_intervals"]):
                local_hint = "word"
            label = obj.get("label", obj.get("text", obj.get("mark")))
            start = obj.get("start", obj.get("xmin", obj.get("begin")))
            end = obj.get("end", obj.get("xmax", obj.get("finish")))
            if label is not None and start is not None and end is not None:
                if "phone" in local_hint:
                    add_interval(phones, label, start, end)
                elif "word" in local_hint:
                    add_interval(words, label, start, end)
            for k, v in obj.items():
                name = str(k).lower()
                child_hint = local_hint
                if "phone" in name:
                    child_hint = "phone"
                elif "word" in name or "utterance" in name:
                    child_hint = "word"
                walk(v, child_hint)
        elif isinstance(obj, list):
            for item in obj:
                walk(item, hint)

    walk(data)
    words = _dedupe_intervals(words)
    phones = _dedupe_intervals(phones)
    return {"words": words, "phones": phones}


def parse_textgrid(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    try:
        from praatio import textgrid  # type: ignore

        tg = textgrid.openTextgrid(str(path), includeEmptyIntervals=False)
        words: List[Dict[str, Any]] = []
        phones: List[Dict[str, Any]] = []
        for tier_name in tg.tierNames:
            tier = tg.getTier(tier_name)
            target = phones if "phone" in tier_name.lower() else words if "word" in tier_name.lower() else None
            if target is None:
                continue
            for start, end, label in tier.entries:
                label = str(label).strip()
                if label and label.lower() not in {"sp", "sil"}:
                    target.append({"label": label, "start": float(start), "end": float(end)})
        return {"words": _dedupe_intervals(words), "phones": _dedupe_intervals(phones)}
    except Exception:
        # Very small regex parser for MFA TextGrid text files.
        text = path.read_text(encoding="utf-8", errors="ignore")
        tiers = re.split(r"item \[\d+\]:", text)
        out = {"words": [], "phones": []}
        for tier in tiers:
            name_match = re.search(r'name = "([^"]+)"', tier)
            if not name_match:
                continue
            name = name_match.group(1).lower()
            target = out["phones"] if "phone" in name else out["words"] if "word" in name else None
            if target is None:
                continue
            for xmin, xmax, label in re.findall(r'xmin = ([0-9.]+).*?xmax = ([0-9.]+).*?text = "([^"]*)"', tier, flags=re.S):
                label = label.strip()
                if label and label.lower() not in {"sp", "sil"}:
                    target.append({"label": label, "start": float(xmin), "end": float(xmax)})
        return {"words": _dedupe_intervals(out["words"]), "phones": _dedupe_intervals(out["phones"])}


def resolve_mfa_command(command: str) -> Optional[str]:
    found = shutil.which(command)
    if found:
        return found
    candidate = Path(command)
    if candidate.exists():
        return str(candidate)
    scripts_dir = Path(sys.prefix) / ("Scripts" if sys.platform.startswith("win") else "bin")
    for name in ("mfa.exe", "mfa"):
        candidate = scripts_dir / name
        if candidate.exists():
            return str(candidate)
    return None


def normalize_word(word: str) -> str:
    word = word.strip().lower().replace("’", "'").replace("-", "")
    # Expand a few contractions common in reading passages.
    contractions = {"i'm": "im", "you're": "youre", "can't": "cant", "won't": "wont", "don't": "dont", "it's": "its"}
    word = contractions.get(word, word)
    word = re.sub(r"[^a-z0-9']", "", word)
    word = word.replace("'", "")
    return word


def arpabet_to_ipa(phones: Sequence[str]) -> List[str]:
    ipa: List[str] = []
    for p in phones:
        base = re.sub(r"[0-2]", "", p.upper())
        ipa.append(ARPABET_TO_IPA.get(base, base.lower()))
    return ipa


def stress_pattern_from_arpabet(phones: Sequence[str]) -> List[int]:
    pattern: List[int] = []
    for p in phones:
        base = re.sub(r"[0-2]", "", p.upper())
        if base in _VOWEL_ARPA:
            m = re.search(r"([0-2])", p)
            pattern.append(int(m.group(1)) if m else 0)
    return pattern


def is_vowel_phone(phone: str) -> bool:
    base = re.sub(r"[0-2]", "", phone.upper())
    if base in _VOWEL_ARPA:
        return True
    p = phone.lower()
    return p in _VOWEL_IPA or any(v in p for v in ["ɑ", "æ", "ʌ", "ɔ", "ɛ", "ɝ", "ɪ", "i", "u", "ʊ", "ə"])


def is_consonant_phone(phone: str) -> bool:
    return bool(phone) and not is_vowel_phone(phone)


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _dedupe_intervals(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for x in sorted(items, key=lambda z: (float(z.get("start", 0)), float(z.get("end", 0)), str(z.get("label", "")))):
        key = (str(x.get("label")), round(float(x.get("start", 0)), 3), round(float(x.get("end", 0)), 3))
        if key in seen:
            continue
        seen.add(key)
        out.append(x)
    return out
