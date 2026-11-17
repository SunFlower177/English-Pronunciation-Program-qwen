"""Pipeline orchestration, CLI, and FastAPI interface.

Functional module 6/7.

This is the only module that wires dependencies together. Individual research
components should remain in audio_io, transcription, reference_alignment,
diagnostics, feedback, or evaluation.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Optional

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
except Exception:  # pragma: no cover
    FastAPI = File = Form = HTTPException = UploadFile = None  # type: ignore[assignment]
    HTMLResponse = JSONResponse = StaticFiles = None  # type: ignore[assignment]

from .audio_io import AudioProcessor
from .contracts import AnalysisReport, AsrResult, AsrSegment, AsrWord, CaptSettings, model_to_dict
from .diagnostics import EvidenceAnalyzer
from .feedback import EvidenceGroundedFeedbackGenerator, compute_scores
from .imitation_analysis import ImitationAnalyzer
from .learning import LearningIssue, LearningOrchestrator
from .mode_d import ListeningAnalyzer
from .mode_e import WritingAnalyzer
from .mode_c import LongSentenceAnalyzer
from .reference_alignment import AlignmentEngine
from .reference_profile import ReferenceProfileBuilder
from .transcription import WhisperXTranscriber

logger = logging.getLogger(__name__)


class CaptPipeline:
    """End-to-end pipeline for Mode A: scripted reading."""

    def __init__(self, settings: Optional[CaptSettings] = None):
        self.settings = settings or CaptSettings.from_env()
        self.audio_processor = AudioProcessor(self.settings)
        self.transcriber = WhisperXTranscriber(self.settings)
        self.aligner = AlignmentEngine(self.settings)
        self.analyzer = EvidenceAnalyzer(self.settings)
        self.reference_profiler = ReferenceProfileBuilder(self.settings)
        self.imitation_analyzer = ImitationAnalyzer(self.settings)
        self.feedback = EvidenceGroundedFeedbackGenerator(self.settings)
        self.long_sentence_analyzer = LongSentenceAnalyzer(self.settings)
        self.listening_analyzer = ListeningAnalyzer(self.settings)
        self.writing_analyzer = WritingAnalyzer(self.settings)
        self.learning_orchestrator = LearningOrchestrator(self.settings)

    def analyze_reading(self, reference_text: str, audio_path: str | Path, analysis_id: Optional[str] = None) -> AnalysisReport:
        reference_text = (reference_text or "").strip()
        if not reference_text:
            raise ValueError("reference_text is required for Mode A scripted reading")
        analysis_id = analysis_id or uuid.uuid4().hex[:12]
        run_dir = self.settings.work_dir / analysis_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "reference.txt").write_text(reference_text, encoding="utf-8")

        logger.info("[%s] Preparing audio", analysis_id)
        audio = self.audio_processor.prepare_audio(audio_path, analysis_id)
        logger.info("[%s] ASR transcription", analysis_id)
        asr = self.transcriber.transcribe(audio.path)
        _write_json(run_dir / "asr.json", asr)

        logger.info("[%s] Reference/ASR/MFA alignment", analysis_id)
        alignment = self.aligner.align(reference_text, asr, audio.path, analysis_id)
        _write_json(run_dir / "alignment.json", alignment)

        logger.info("[%s] Evidence diagnostics", analysis_id)
        evidence = self.analyzer.analyze(audio, alignment)
        _write_json(run_dir / "evidence.json", evidence)

        logger.info("[%s] Feedback generation", analysis_id)
        feedback = self.feedback.generate(reference_text, audio.path, asr, alignment, evidence)
        report = AnalysisReport(
            analysis_id=analysis_id,
            mode="reading",
            reference_text=reference_text,
            audio=audio,
            asr=asr,
            alignment=alignment,
            evidence=evidence,
            feedback=feedback,
        )
        _write_json(run_dir / "report.json", report)
        return report

    def analyze_imitation(
        self,
        reference_audio_path: str | Path,
        learner_audio_path: str | Path,
        reference_text: Optional[str] = None,
        analysis_id: Optional[str] = None,
    ) -> AnalysisReport:
        """Mode A2: compare learner reading against a reference recording.

        The optional transcript remains the text bridge. If it is omitted, the
        reference recording is transcribed first and that ASR transcript becomes
        the shared reference text for both recordings.
        """
        analysis_id = analysis_id or uuid.uuid4().hex[:12]
        run_dir = self.settings.work_dir / analysis_id
        run_dir.mkdir(parents=True, exist_ok=True)

        logger.info("[%s] Preparing reference audio", analysis_id)
        reference_audio = self.audio_processor.prepare_audio(reference_audio_path, f"{analysis_id}/reference")
        logger.info("[%s] Preparing learner audio", analysis_id)
        learner_audio = self.audio_processor.prepare_audio(learner_audio_path, f"{analysis_id}/learner")

        logger.info("[%s] Reference ASR transcription", analysis_id)
        reference_asr = self.transcriber.transcribe(reference_audio.path)
        _write_json(run_dir / "reference_asr.json", reference_asr)

        reference_text = (reference_text or "").strip()
        reference_text_source = "user" if reference_text else "reference_asr"
        if not reference_text:
            reference_text = reference_asr.text.strip()
        if not reference_text:
            raise ValueError("reference_text is required or must be recoverable from reference_audio")
        (run_dir / "reference.txt").write_text(reference_text, encoding="utf-8")

        logger.info("[%s] Learner ASR transcription", analysis_id)
        learner_asr = self.transcriber.transcribe(learner_audio.path)
        _write_json(run_dir / "asr.json", learner_asr)

        # Use a synthetic equal-word ASR for the reference recording so forced
        # alignment can target every transcript word even if reference ASR has
        # minor recognition differences.
        reference_bridge_asr = self._asr_from_reference_text(reference_text)

        logger.info("[%s] Reference forced alignment", analysis_id)
        reference_alignment = self.aligner.align(reference_text, reference_bridge_asr, reference_audio.path, f"{analysis_id}/reference")
        _write_json(run_dir / "reference_alignment.json", reference_alignment)

        logger.info("[%s] Learner reference/ASR alignment", analysis_id)
        learner_alignment = self.aligner.align(reference_text, learner_asr, learner_audio.path, f"{analysis_id}/learner")
        _write_json(run_dir / "alignment.json", learner_alignment)

        logger.info("[%s] Reference profile extraction", analysis_id)
        reference_profile = self.reference_profiler.build(
            reference_audio,
            reference_alignment,
            reference_text,
            reference_text_source=reference_text_source,
            asr_text=reference_asr.text,
        )
        _write_json(run_dir / "reference_profile.json", reference_profile)

        logger.info("[%s] Imitation evidence diagnostics", analysis_id)
        evidence = self.imitation_analyzer.analyze(
            reference_profile=reference_profile,
            learner_audio=learner_audio,
            learner_asr=learner_asr,
            learner_alignment=learner_alignment,
            reference_text=reference_text,
        )
        _write_json(run_dir / "evidence.json", evidence)

        logger.info("[%s] Feedback generation", analysis_id)
        feedback = self.feedback.generate(reference_text, learner_audio.path, learner_asr, learner_alignment, evidence)
        feedback.overall_scores = compute_scores(evidence, self.settings, learner_alignment)
        report = AnalysisReport(
            analysis_id=analysis_id,
            mode="imitation",
            reference_text=reference_text,
            reference_text_source=reference_text_source,  # type: ignore[arg-type]
            reference_audio=reference_audio,
            reference_profile=reference_profile,
            audio=learner_audio,
            asr=learner_asr,
            alignment=learner_alignment,
            evidence=evidence,
            feedback=feedback,
        )
        _write_json(run_dir / "report.json", report)
        return report

    def _asr_from_reference_text(self, reference_text: str) -> AsrResult:
        tokens = self.aligner.reference_processor.tokenize(reference_text)
        words = [AsrWord(word=t.raw, start=None, end=None, score=1.0, source="reference_text") for t in tokens]
        return AsrResult(
            text=reference_text,
            language=self.settings.language,
            segments=[AsrSegment(text=reference_text, words=words)],
            words=words,
            raw={"backend": "reference_text_bridge"},
        )

    def analyze_free_speaking(self, audio_path: str | Path, analysis_id: Optional[str] = None) -> AnalysisReport:
        """Mode B placeholder with honest task boundary.

        Free speaking has no reference text, so word/phoneme error diagnosis is
        intentionally disabled. This method is included to make the future
        extension point explicit without mixing Mode B assumptions into Mode A.
        """
        raise NotImplementedError(
            "Mode B is intentionally not enabled yet. Add a separate free-speaking analyzer that reports ASR, fluency, "
            "prosody, grammar/expression feedback, and only high-confidence pronunciation risks."
        )

    def analyze_long_sentence(
        self,
        sentence: str,
        level: str = "TOEFL",
        detail: str = "full",
        analysis_id: Optional[str] = None,
    ):
        """Mode C: long-sentence reading-comprehension analysis."""

        analysis_id = analysis_id or uuid.uuid4().hex[:12]
        run_dir = self.settings.work_dir / analysis_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "sentence.txt").write_text(sentence, encoding="utf-8")
        report = self.long_sentence_analyzer.analyze(sentence, level=level, detail=detail, analysis_id=analysis_id)
        _write_json(run_dir / "report.json", report)
        return report

    def analyze_listening(
        self,
        audio_path: str | Path | None = None,
        reference_transcript: str = "",
        student_transcript: str = "",
        student_notes: str = "",
        user_id: str = "local_user",
        analysis_id: Optional[str] = None,
    ):
        """Mode D: listening dictation, notes, and drills."""

        analysis_id = analysis_id or uuid.uuid4().hex[:12]
        run_dir = self.settings.work_dir / analysis_id
        run_dir.mkdir(parents=True, exist_ok=True)
        audio = None
        asr = None
        alignment = None
        if audio_path is not None:
            logger.info("[%s] Preparing listening audio", analysis_id)
            audio = self.audio_processor.prepare_audio(audio_path, analysis_id)
            logger.info("[%s] Listening ASR transcription", analysis_id)
            asr = self.transcriber.transcribe(audio.path)
            _write_json(run_dir / "asr.json", asr)
        reference_transcript = (reference_transcript or "").strip()
        if not reference_transcript and asr is not None:
            reference_transcript = asr.text.strip()
        if audio is not None and asr is not None and reference_transcript:
            logger.info("[%s] Listening transcript/audio alignment", analysis_id)
            alignment = self.aligner.align(reference_transcript, asr, audio.path, f"{analysis_id}/listening")
            _write_json(run_dir / "alignment.json", alignment)
        report = self.listening_analyzer.analyze(
            reference_transcript=reference_transcript,
            student_transcript=student_transcript,
            student_notes=student_notes,
            audio=audio,
            asr=asr,
            alignment=alignment,
            analysis_id=analysis_id,
        )
        report.learning = self.learning_orchestrator.process(user_id, "D", "listening", report.learning_issues)
        _write_json(run_dir / "report.json", report)
        return report

    def analyze_writing(
        self,
        prompt: str,
        essay: str,
        exam_type: str = "IELTS_Task_2",
        previous_essay: str = "",
        user_id: str = "local_user",
        analysis_id: Optional[str] = None,
    ):
        """Mode E: rubric-aligned writing diagnosis and revision coaching."""

        analysis_id = analysis_id or uuid.uuid4().hex[:12]
        run_dir = self.settings.work_dir / analysis_id
        run_dir.mkdir(parents=True, exist_ok=True)
        report = self.writing_analyzer.analyze(
            prompt=prompt,
            essay=essay,
            exam_type=exam_type,
            previous_essay=previous_essay,
            analysis_id=analysis_id,
        )
        report.learning = self.learning_orchestrator.process(user_id, "E", "writing", report.learning_issues)
        _write_json(run_dir / "report.json", report)
        return report


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(_json_safe(model_to_dict(obj)), ensure_ascii=False, indent=2), encoding="utf-8")


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def learning_issues_from_pronunciation(report: AnalysisReport, user_id: str = "local_user") -> list[LearningIssue]:
    out: list[LearningIssue] = []
    source_mode = "A" if report.mode == "reading" else "B"
    source_task = "reading" if report.mode == "reading" else "imitation"
    for issue in report.evidence.all_issues:
        issue_type = issue.type.value
        tags = pronunciation_tags(issue_type)
        out.append(
            LearningIssue(
                issue_id=f"L{source_mode}_{issue.issue_id}",
                user_id=user_id,
                source_mode=source_mode,  # type: ignore[arg-type]
                source_task=source_task,
                issue_type=issue_type,
                skill_tags=tags,
                span_text=issue.word or "",
                reference_text=issue.expected or report.reference_text or "",
                user_output=issue.observed or "",
                audio_start=issue.start,
                audio_end=issue.end,
                severity=issue.severity.value,
                confidence=issue.confidence,
                evidence=model_to_dict(issue),
            )
        )
    return out[:80]


def pronunciation_tags(issue_type: str) -> list[str]:
    if issue_type.startswith("phoneme"):
        return ["pronunciation.phoneme", "listening.minimal_pair"]
    if "linking" in issue_type:
        return ["speaking.linking", "listening.linking"]
    if "pause" in issue_type or issue_type == "rhythm":
        return ["speaking.fluency", "listening.chunking"]
    if "stress" in issue_type:
        return ["speaking.stress", "listening.prominence"]
    if "intonation" in issue_type:
        return ["speaking.intonation", "listening.prosody"]
    if issue_type.startswith("word"):
        return ["speaking.word_accuracy", "listening.dictation"]
    return ["speaking.pronunciation"]


def learning_issues_from_long_sentence(report: Any, user_id: str = "local_user") -> list[LearningIssue]:
    evidence = report.evidence
    out: list[LearningIssue] = []
    for idx, clause in enumerate(evidence.clauses, start=1):
        tags = ["reading.clause_chunking"]
        if "定语" in clause.function_zh or clause.marker in {"which", "that", "who", "whom", "whose"}:
            tags.append("reading.relative_clause")
            tags.append("grammar.relative_clause")
        elif "原因" in clause.function_zh or clause.marker == "because":
            tags.append("reading.discourse_relation")
        out.append(
            LearningIssue(
                issue_id=f"LC_CLAUSE_{idx:03d}",
                user_id=user_id,
                source_mode="C",
                source_task="long_sentence",
                issue_type=f"clause_{clause.type}",
                skill_tags=sorted(set(tags)),
                span_text=clause.span,
                reference_text=clause.modifies,
                user_output="",
                severity="low",
                confidence=clause.confidence or 0.6,
                evidence=model_to_dict(clause),
            )
        )
    for idx, point in enumerate(evidence.grammar_points, start=1):
        out.append(
            LearningIssue(
                issue_id=f"LC_GRAMMAR_{idx:03d}",
                user_id=user_id,
                source_mode="C",
                source_task="long_sentence",
                issue_type=point.type,
                skill_tags=["reading.grammar", f"grammar.{point.type}"],
                span_text=point.span,
                reference_text=point.explanation_zh,
                user_output="",
                severity="low",
                confidence=point.confidence or 0.6,
                evidence=model_to_dict(point),
            )
        )
    return out[:60]


# ---------- CLI ----------


def cli_main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evidence-grounded L2 English pronunciation diagnosis")
    parser.add_argument("--audio", required=True, help="Path to user/learner reading audio")
    parser.add_argument("--reference", default=None, help="Reference text or path to a .txt file")
    parser.add_argument("--reference-audio", default=None, help="Optional reference recording for imitation mode")
    parser.add_argument("--analysis-id", default=None, help="Optional deterministic run id")
    parser.add_argument("--work-dir", default=None, help="Override CAPT_WORK_DIR")
    parser.add_argument(
        "--score-config",
        default=None,
        help="YAML parameters for deterministic continuous scoring",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s:%(name)s:%(message)s")

    settings = CaptSettings.from_env()
    if args.work_dir:
        settings.work_dir = Path(args.work_dir).resolve()
        settings.work_dir.mkdir(parents=True, exist_ok=True)
    if args.score_config:
        settings.deterministic_score_config = str(Path(args.score_config).resolve())
    reference_text = None
    if args.reference:
        ref_arg = Path(args.reference)
        reference_text = ref_arg.read_text(encoding="utf-8") if ref_arg.exists() else args.reference
    pipeline = CaptPipeline(settings)
    if args.reference_audio:
        report = pipeline.analyze_imitation(args.reference_audio, args.audio, reference_text, args.analysis_id)
    else:
        if not reference_text:
            parser.error("--reference is required unless --reference-audio is provided")
        report = pipeline.analyze_reading(reference_text, args.audio, args.analysis_id)
    print(json.dumps(model_to_dict(report), ensure_ascii=False, indent=2))
    return 0


# ---------- FastAPI ----------


def create_app():
    if FastAPI is None or File is None or Form is None or UploadFile is None:  # pragma: no cover
        raise RuntimeError("FastAPI is not installed. Install with `pip install fastapi uvicorn python-multipart`.")

    settings = CaptSettings.from_env()
    pipeline = CaptPipeline(settings)
    app = FastAPI(title="Evidence-Grounded CAPT", version="0.1.0")
    web_candidates = [
        Path(__file__).resolve().parents[2] / "web",
        Path(__file__).resolve().parents[1] / "web",
    ]
    web_dir = next((p for p in web_candidates if p.exists()), web_candidates[0])
    if web_dir.exists():
        app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index():
        index_path = web_dir / "index.html"
        if index_path.exists():
            return index_path.read_text(encoding="utf-8")
        return "<h1>Evidence-Grounded CAPT</h1><p>POST /api/analyze-reading or /api/analyze-long-sentence</p>"

    @app.post("/api/analyze-reading")
    async def analyze_reading(
        reference_text: str = Form(...),
        audio: UploadFile = File(...),
        user_id: str = Form("local_user"),
    ):
        analysis_id = uuid.uuid4().hex[:12]
        suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
        upload_path = settings.upload_dir / f"{analysis_id}{suffix}"
        try:
            with upload_path.open("wb") as f:
                shutil.copyfileobj(audio.file, f)
            report = pipeline.analyze_reading(reference_text, upload_path, analysis_id=analysis_id)
            data = _json_safe(model_to_dict(report))
            data["learning"] = _json_safe(model_to_dict(pipeline.learning_orchestrator.process(
                user_id, "A", "reading", learning_issues_from_pronunciation(report, user_id)
            )))
            _write_json(settings.work_dir / analysis_id / "learning.json", data["learning"])
            return JSONResponse(data)
        except Exception as exc:
            logger.exception("Analysis failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/analyze-imitation")
    async def analyze_imitation(
        reference_audio: UploadFile = File(...),
        learner_audio: UploadFile = File(...),
        reference_text: str = Form(""),
        user_id: str = Form("local_user"),
    ):
        analysis_id = uuid.uuid4().hex[:12]
        ref_suffix = Path(reference_audio.filename or "reference.wav").suffix or ".wav"
        learner_suffix = Path(learner_audio.filename or "learner.wav").suffix or ".wav"
        reference_upload_path = settings.upload_dir / f"{analysis_id}.reference{ref_suffix}"
        learner_upload_path = settings.upload_dir / f"{analysis_id}.learner{learner_suffix}"
        try:
            with reference_upload_path.open("wb") as f:
                shutil.copyfileobj(reference_audio.file, f)
            with learner_upload_path.open("wb") as f:
                shutil.copyfileobj(learner_audio.file, f)
            report = pipeline.analyze_imitation(
                reference_upload_path,
                learner_upload_path,
                reference_text=reference_text,
                analysis_id=analysis_id,
            )
            data = _json_safe(model_to_dict(report))
            data["learning"] = _json_safe(model_to_dict(pipeline.learning_orchestrator.process(
                user_id, "B", "imitation", learning_issues_from_pronunciation(report, user_id)
            )))
            _write_json(settings.work_dir / analysis_id / "learning.json", data["learning"])
            return JSONResponse(data)
        except Exception as exc:
            logger.exception("Imitation analysis failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/analyze-long-sentence")
    async def analyze_long_sentence(
        sentence: str = Form(...),
        level: str = Form("TOEFL"),
        detail: str = Form("full"),
        user_id: str = Form("local_user"),
    ):
        analysis_id = uuid.uuid4().hex[:12]
        try:
            report = pipeline.analyze_long_sentence(sentence, level=level, detail=detail, analysis_id=analysis_id)
            data = _json_safe(model_to_dict(report))
            data["learning"] = _json_safe(model_to_dict(pipeline.learning_orchestrator.process(
                user_id, "C", "long_sentence", learning_issues_from_long_sentence(report, user_id)
            )))
            _write_json(settings.work_dir / analysis_id / "learning.json", data["learning"])
            return JSONResponse(data)
        except Exception as exc:
            logger.exception("Long-sentence analysis failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/analyze-listening")
    async def analyze_listening(
        audio: Optional[UploadFile] = File(None),
        reference_transcript: str = Form(""),
        student_transcript: str = Form(""),
        student_notes: str = Form(""),
        user_id: str = Form("local_user"),
    ):
        analysis_id = uuid.uuid4().hex[:12]
        upload_path: Optional[Path] = None
        try:
            if audio is not None:
                suffix = Path(audio.filename or "listening.wav").suffix or ".wav"
                upload_path = settings.upload_dir / f"{analysis_id}.listening{suffix}"
                with upload_path.open("wb") as f:
                    shutil.copyfileobj(audio.file, f)
            report = pipeline.analyze_listening(
                audio_path=upload_path,
                reference_transcript=reference_transcript,
                student_transcript=student_transcript,
                student_notes=student_notes,
                user_id=user_id,
                analysis_id=analysis_id,
            )
            return JSONResponse(_json_safe(model_to_dict(report)))
        except Exception as exc:
            logger.exception("Listening analysis failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/analyze-writing")
    async def analyze_writing(
        prompt: str = Form(...),
        essay: str = Form(...),
        exam_type: str = Form("IELTS_Task_2"),
        previous_essay: str = Form(""),
        user_id: str = Form("local_user"),
    ):
        analysis_id = uuid.uuid4().hex[:12]
        try:
            report = pipeline.analyze_writing(
                prompt=prompt,
                essay=essay,
                exam_type=exam_type,
                previous_essay=previous_essay,
                user_id=user_id,
                analysis_id=analysis_id,
            )
            return JSONResponse(_json_safe(model_to_dict(report)))
        except Exception as exc:
            logger.exception("Writing analysis failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/api/reports/{analysis_id}")
    def get_report(analysis_id: str):
        path = settings.work_dir / analysis_id / "report.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail="report not found")
        return JSONResponse(_json_safe(json.loads(path.read_text(encoding="utf-8"))))

    @app.get("/api/health")
    def health():
        return {"status": "ok", "work_dir": str(settings.work_dir), "device": settings.device}

    return app


try:  # Allows `uvicorn capt_pronunciation.runtime:app`.
    app = create_app()
except Exception:  # pragma: no cover
    app = None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
