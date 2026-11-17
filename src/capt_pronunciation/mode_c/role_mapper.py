"""Dependency-grounded argument mapping for Mode C.

This is deliberately described as an argument map rather than full PropBank
SRL.  Every ARG0/ARG1 span is tied to dependency edges from a trained parser;
the heuristic backend emits only the conservative main-clause fallback.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from .schemas import ClauseInfo, MainClause, SemanticRole, SyntaxToken


class RoleMapper:
    def map_roles(
        self,
        sentence: str,
        main_clause: MainClause,
        clauses: List[ClauseInfo],
        tokens: Optional[List[SyntaxToken]] = None,
    ) -> List[SemanticRole]:
        tokens = tokens or []
        predicates = [
            token
            for token in tokens
            if token.pos in {"VERB", "AUX"}
            and token.dep in {"ROOT", "root", "advcl", "relcl", "acl", "ccomp", "xcomp", "csubj"}
        ]
        if not predicates:
            return self._fallback(main_clause)

        roles: List[SemanticRole] = []
        for predicate in predicates:
            subjects = children_with_deps(
                tokens,
                predicate.index,
                {"nsubj", "nsubjpass", "csubj", "expl"},
            )
            objects = children_with_deps(
                tokens,
                predicate.index,
                {"obj", "dobj", "iobj", "attr", "acomp", "oprd"},
            )
            argm: Dict[str, str] = {}
            for clause in clauses:
                if not (clause.start <= predicate.start < clause.end):
                    continue
                if "reason" in clause.type:
                    argm["AM-CAU"] = clause.span
                elif "condition" in clause.type:
                    argm["AM-CND"] = clause.span
                elif "time" in clause.type:
                    argm["AM-TMP"] = clause.span
                elif "concession" in clause.type:
                    argm["AM-ADV"] = clause.span
                elif "relative" in clause.type:
                    argm["MOD"] = clause.span
            roles.append(
                SemanticRole(
                    predicate=predicate.text,
                    arg0=span_text(sentence, subtree_union(tokens, subjects)),
                    arg1=span_text(sentence, subtree_union(tokens, objects)),
                    argm=argm,
                    explanation_zh="基于依存关系提取的谓词—论元结构；它不是生成式推测。",
                    confidence=0.82 if subjects else 0.64,
                )
            )
        return roles

    def _fallback(self, main_clause: MainClause) -> List[SemanticRole]:
        if not main_clause.predicate:
            return []
        return [
            SemanticRole(
                predicate=main_clause.predicate,
                arg0=main_clause.subject,
                arg1=main_clause.object_or_complement,
                explanation_zh="未获得训练解析器结果，仅保留低置信度主干论元映射。",
                confidence=min(0.45, main_clause.confidence),
            )
        ]


def children_with_deps(
    tokens: List[SyntaxToken],
    head: int,
    dependencies: set[str],
) -> List[SyntaxToken]:
    return [token for token in tokens if token.head == head and token.dep in dependencies]


def subtree_union(tokens: List[SyntaxToken], roots: Iterable[SyntaxToken]) -> List[SyntaxToken]:
    indices = {root.index for root in roots}
    pending = list(indices)
    while pending:
        head = pending.pop()
        for token in tokens:
            if token.index not in indices and token.head == head:
                indices.add(token.index)
                pending.append(token.index)
    return sorted((token for token in tokens if token.index in indices), key=lambda token: token.start)


def span_text(sentence: str, tokens: List[SyntaxToken]) -> str:
    if not tokens:
        return ""
    return sentence[min(token.start for token in tokens) : max(token.end for token in tokens)].strip(" ,;")
