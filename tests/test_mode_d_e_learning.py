from capt_pronunciation.contracts import CaptSettings
from capt_pronunciation.learning import LearningOrchestrator
from capt_pronunciation.mode_d import ListeningAnalyzer
from capt_pronunciation.mode_e import WritingAnalyzer


def test_mode_d_dictation_notes_and_practice(tmp_path):
    settings = CaptSettings(work_dir=tmp_path, upload_dir=tmp_path, use_qwen_text=False)
    analyzer = ListeningAnalyzer(settings)
    report = analyzer.analyze(
        reference_transcript="The theory had been widely accepted for decades, but new evidence challenged it.",
        student_transcript="The theory widely accepted for decades but new evident challenged it.",
        student_notes="theory accepted for a long time; new evidence challenged",
        analysis_id="listen-unit",
    )

    assert report.mode == "listening"
    assert report.evidence.scores.dictation_accuracy < 100
    assert any(i.issue_type in {"function_word_omission", "content_word_misheard", "spelling_or_word_form"} for i in report.learning_issues)
    assert report.evidence.note_coverage
    assert report.practice_items


def test_mode_e_writing_rubric_and_revision_tasks(tmp_path):
    settings = CaptSettings(work_dir=tmp_path, upload_dir=tmp_path, use_qwen_text=False)
    analyzer = WritingAnalyzer(settings)
    report = analyzer.analyze(
        prompt="Some people believe online learning is better than traditional classroom learning. To what extent do you agree or disagree?",
        essay="Many student believe online learning are convenient. I think online learning is very good because people can save time.",
        exam_type="IELTS_Task_2",
        analysis_id="write-unit",
    )

    assert report.mode == "writing"
    assert report.evidence.rubric_scores.overall > 0
    assert any(e.type == "plural_error" for e in report.evidence.local_errors)
    assert any(e.type == "subject_verb_agreement" for e in report.evidence.local_errors)
    assert report.evidence.revision_tasks
    assert any("writing.grammar" in i.skill_tags for i in report.learning_issues)


def test_mode_e_avoids_many_adjective_false_positive_and_caps_scores(tmp_path):
    settings = CaptSettings(work_dir=tmp_path, upload_dir=tmp_path, use_qwen_text=False)
    analyzer = WritingAnalyzer(settings)
    essay = (
        "Some people believe that online learning is better than traditional classroom learning. "
        "I am partly agree with this opinion, but I also think classroom learning still have many important advantages. "
        "First of all, online learning is more freedom and convenient. I watched many videos on internet and understood some knowledges better. "
        "If a student lives far away from campus, online learning can make his life more easier. "
        "However, students may feel boring online and did not learn nothing from that course. "
        "This shows that online learning need strong self-control. "
        "University is a place to get informations, but also a place to develop social skills."
    )
    report = analyzer.analyze(
        prompt="Some people believe online learning is better than traditional classroom learning. To what extent do you agree or disagree?",
        essay=essay,
        exam_type="IELTS_Task_2",
        analysis_id="write-caps",
    )

    spans = {e.span for e in report.evidence.local_errors}
    types = {e.type for e in report.evidence.local_errors}
    assert "many important" not in spans
    assert "classroom learning still have" in spans
    assert "some knowledges" in spans
    assert "more easier" in spans
    assert "did not learn nothing" in spans
    assert {"verb_form_error", "subject_verb_agreement", "uncountable_noun_error", "comparative_error", "double_negative"} <= types
    assert report.evidence.rubric_scores.overall <= 6.5


def test_learning_orchestrator_recommends_cross_mode(tmp_path):
    settings = CaptSettings(work_dir=tmp_path, upload_dir=tmp_path, use_qwen_text=False)
    report = ListeningAnalyzer(settings).analyze(
        reference_transcript="I want to find out more about it.",
        student_transcript="I want find out more about it.",
        analysis_id="orch-source",
    )
    result = LearningOrchestrator(settings).process("u1", "D", "listening", report.learning_issues)

    assert result.issues
    assert result.profile_updates
    assert result.recommendations
    assert result.practice_items
    assert any(item.target_mode in {"B", "C", "D"} for item in result.practice_items)
