"""Optional NLP parser wrapper for Mode C.

The analyzer prefers spaCy when an English model is installed, can collect a
Stanza constituency tree when available, and otherwise falls back to a compact
rule-based tokenizer. This keeps the web demo usable while preserving the
recommended architecture for a full parser-backed setup.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from ..contracts import CaptSettings
from .schemas import SyntaxToken
from .text_input import split_sentence_spans, word_spans

logger = logging.getLogger(__name__)


class SyntaxParser:
    def __init__(self, settings: CaptSettings):
        self.settings = settings
        self._spacy_nlp: Any = None
        self._spacy_failed = False
        self._stanza_pipeline: Any = None
        self._stanza_failed = False

    def parse(self, sentence: str) -> Dict[str, Any]:
        spacy_doc = self._parse_spacy(sentence)
        tokens: List[SyntaxToken]
        noun_chunks: List[Dict[str, Any]] = []
        parse_trees: List[str] = []
        constituency_spans: List[Dict[str, Any]] = []
        backend = "heuristic"
        warnings: List[str] = []

        if spacy_doc is not None:
            backend = "spacy"
            tokens = [
                SyntaxToken(
                    index=i,
                    text=t.text,
                    lemma=t.lemma_,
                    pos=t.pos_,
                    tag=t.tag_,
                    dep=t.dep_,
                    head=t.head.i if t.head is not None else None,
                    start=t.idx,
                    end=t.idx + len(t.text),
                    morph=t.morph.to_dict(),
                )
                for i, t in enumerate(spacy_doc)
                if not t.is_space
            ]
            try:
                noun_chunks = [
                    {
                        "text": chunk.text,
                        "start": chunk.start_char,
                        "end": chunk.end_char,
                        "root": chunk.root.text,
                        "root_index": chunk.root.i,
                    }
                    for chunk in spacy_doc.noun_chunks
                ]
            except Exception:
                noun_chunks = []
        stanza_result = self._parse_stanza(sentence)
        if spacy_doc is None and stanza_result is not None:
            tokens = stanza_result["tokens"]
            parse_trees = stanza_result["parse_trees"]
            constituency_spans = stanza_result["constituency_spans"]
            backend = "stanza_dependency+constituency"
        elif spacy_doc is None:
            tokens = self._heuristic_tokens(sentence)
            warnings.append(
                "No trained dependency parser is available; heuristic output is low confidence."
            )
        elif stanza_result is not None:
            parse_trees = stanza_result["parse_trees"]
            constituency_spans = stanza_result["constituency_spans"]
            backend = f"{backend}+stanza_constituency"

        if stanza_result is None and self._stanza_failed:
            warnings.append("Stanza parser unavailable; constituency agreement cannot be checked.")

        return {
            "backend": backend,
            "tokens": tokens,
            "noun_chunks": noun_chunks,
            "sentence_spans": split_sentence_spans(sentence),
            "parse_trees": parse_trees,
            "constituency_spans": constituency_spans,
            "warnings": warnings,
        }

    def _parse_spacy(self, sentence: str) -> Optional[Any]:
        if self._spacy_failed:
            return None
        if self._spacy_nlp is None:
            try:
                import spacy  # type: ignore

                model_names = [
                    getattr(self.settings, "mode_c_spacy_model", ""),
                    "en_core_web_trf",
                    "en_core_web_sm",
                ]
                last_exc: Optional[Exception] = None
                for name in [x for x in model_names if x]:
                    try:
                        self._spacy_nlp = spacy.load(name)
                        break
                    except Exception as exc:
                        last_exc = exc
                if self._spacy_nlp is None:
                    raise last_exc or RuntimeError("no spaCy model candidates")
            except Exception as exc:
                logger.info("Mode C spaCy parser unavailable: %s", exc)
                self._spacy_failed = True
                return None
        return self._spacy_nlp(sentence)

    def _parse_stanza(self, sentence: str) -> Optional[Dict[str, Any]]:
        if self._stanza_failed:
            return None
        if self._stanza_pipeline is None:
            try:
                import stanza  # type: ignore

                self._stanza_pipeline = stanza.Pipeline(
                    lang="en",
                    processors="tokenize,pos,lemma,depparse,constituency",
                    tokenize_no_ssplit=True,
                    verbose=False,
                )
            except Exception as exc:
                logger.info("Mode C Stanza parser unavailable: %s", exc)
                self._stanza_failed = True
                return None
        try:
            doc = self._stanza_pipeline(sentence)
            tokens: List[SyntaxToken] = []
            trees: List[str] = []
            spans: List[Dict[str, Any]] = []
            cursor = 0
            for parsed_sentence in doc.sentences:
                sentence_tokens: List[SyntaxToken] = []
                base = len(tokens)
                for local_index, word in enumerate(parsed_sentence.words):
                    start = getattr(word, "start_char", None)
                    end = getattr(word, "end_char", None)
                    if start is None or end is None:
                        start = sentence.find(str(word.text), cursor)
                        start = cursor if start < 0 else start
                        end = start + len(str(word.text))
                    cursor = int(end)
                    head_value = int(getattr(word, "head", 0) or 0)
                    global_index = base + local_index
                    head = global_index if head_value == 0 else base + head_value - 1
                    token = SyntaxToken(
                        index=global_index,
                        text=str(word.text),
                        lemma=str(getattr(word, "lemma", "") or ""),
                        pos=str(getattr(word, "upos", "") or ""),
                        tag=str(getattr(word, "xpos", "") or ""),
                        dep=normalize_stanza_dep(str(getattr(word, "deprel", "") or "")),
                        head=head,
                        start=int(start),
                        end=int(end),
                        morph=parse_stanza_morph(str(getattr(word, "feats", "") or "")),
                    )
                    sentence_tokens.append(token)
                    tokens.append(token)
                tree = getattr(parsed_sentence, "constituency", None)
                if tree is not None:
                    trees.append(str(tree))
                    spans.extend(constituency_spans(tree, sentence_tokens))
            if not tokens:
                return None
            return {
                "tokens": tokens,
                "parse_trees": trees,
                "constituency_spans": spans,
            }
        except Exception as exc:
            logger.info("Mode C Stanza parse failed: %s", exc)
            self._stanza_failed = True
            return None

    def _heuristic_tokens(self, sentence: str) -> List[SyntaxToken]:
        aux = {"am", "is", "are", "was", "were", "be", "been", "being", "has", "have", "had", "do", "does", "did", "will", "would", "can", "could", "may", "might", "must", "should"}
        preps = {"in", "on", "at", "for", "with", "by", "from", "to", "of", "about", "through", "without", "within", "over", "under", "between", "among", "during", "before", "after", "because", "although", "if", "while", "since", "unless"}
        spans = word_spans(sentence)
        tokens: List[SyntaxToken] = []
        root_index = self._guess_root_index([s.text for s in spans])
        for i, span in enumerate(spans):
            lower = span.text.lower()
            pos = "AUX" if lower in aux else "ADP" if lower in preps else "VERB" if i == root_index else "NOUN"
            dep = "ROOT" if i == root_index else "aux" if lower in aux else "mark" if lower in preps else ""
            tokens.append(
                SyntaxToken(
                    index=i,
                    text=span.text,
                    lemma=lower,
                    pos=pos,
                    tag="",
                    dep=dep,
                    head=root_index if i != root_index else i,
                    start=span.start,
                    end=span.end,
                )
            )
        return tokens

    def _guess_root_index(self, words: List[str]) -> int:
        verb_like = re.compile(r"(ed|ing|s)$", re.I)
        for i, word in enumerate(words):
            lower = word.lower()
            if lower in {
                "is", "are", "was", "were", "be", "been", "became", "become",
                "continue", "continued", "rely", "explained", "challenged",
                "suggests", "shows", "indicates", "said", "thought", "found",
                "told", "knew", "believed", "reported", "claimed",
            }:
                return i
            if i > 0 and verb_like.search(word):
                return i
        return min(1, max(0, len(words) - 1))


def normalize_stanza_dep(dep: str) -> str:
    mapping = {
        "root": "ROOT",
        "nsubj:pass": "nsubjpass",
        "csubj:pass": "csubj",
        "aux:pass": "auxpass",
        "acl:relcl": "relcl",
    }
    return mapping.get(dep, dep)


def parse_stanza_morph(feats: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for item in str(feats or "").split("|"):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        result[key] = value
    return result


def constituency_spans(tree: Any, tokens: List[SyntaxToken]) -> List[Dict[str, Any]]:
    """Convert a Stanza tree into character spans that downstream code can use."""
    cursor = [0]
    spans: List[Dict[str, Any]] = []
    accepted = {"S", "SBAR", "SBARQ", "SINV", "SQ", "NP", "VP"}

    def visit(node: Any, parent_label: str = "") -> Optional[tuple[int, int, int, int]]:
        children = list(getattr(node, "children", None) or [])
        label = str(getattr(node, "label", "") or "")
        if not children:
            if cursor[0] >= len(tokens):
                return None
            token_index = cursor[0]
            token = tokens[token_index]
            cursor[0] += 1
            return token.start, token.end, token_index, token_index + 1

        child_spans = [visit(child, label) for child in children]
        child_spans = [span for span in child_spans if span is not None]
        if not child_spans:
            return None
        start = min(span[0] for span in child_spans)
        end = max(span[1] for span in child_spans)
        token_start = min(span[2] for span in child_spans)
        token_end = max(span[3] for span in child_spans)
        if label in accepted:
            spans.append(
                {
                    "label": label,
                    "parent_label": parent_label,
                    "start": start,
                    "end": end,
                    "token_start": token_start,
                    "token_end": token_end,
                }
            )
        return start, end, token_start, token_end

    visit(tree)
    deduped: Dict[tuple[str, int, int], Dict[str, Any]] = {}
    for span in spans:
        deduped[(span["label"], span["start"], span["end"])] = span
    return sorted(deduped.values(), key=lambda item: (item["start"], item["end"], item["label"]))
