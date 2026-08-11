from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests
from reportlab.pdfgen import canvas
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).parents[1] / "frontend" / "app.py"
PROFILE_VALUES = {
    "current_background": "Backend developer",
    "target_role": "AI Engineer",
    "job_description": "Build production LLM services using Python and APIs.",
    "skills_text": "Python, APIs",
    "project_experience": "Built a production API.",
}
PROFILE_PAYLOAD = {
    "current_background": PROFILE_VALUES["current_background"],
    "target_role": PROFILE_VALUES["target_role"],
    "job_description": PROFILE_VALUES["job_description"],
    "skills": ["Python", "APIs"],
    "project_experience": PROFILE_VALUES["project_experience"],
}
ANALYSIS = {
    "match_score": 82,
    "score_breakdown": {
        "technical_skills": 90,
        "domain_experience": 70,
        "seniority": 60,
        "role_specific_requirements": 80,
        "education": 100,
        "communication_leadership": 80,
    },
    "strengths": ["Python"],
    "skill_gaps": ["LLM evaluation"],
    "next_steps": ["Build an LLM project"],
}
RESUME_TEXT = "Python developer with production API experience"
RESUME_ANALYSIS = {**ANALYSIS, "resume_text": RESUME_TEXT}
TAILORING = {
    "top_changes": ["Lead with production Python work"],
    "skills_to_emphasize": ["Python"],
    "experiences_to_emphasize": ["Production API work"],
    "missing_keywords": ["LLM evaluation"],
    "bullet_rewrite_suggestions": ["Built a production API using Python."],
    "overall_advice": ["Lead with relevant production experience."],
}
CAREER_AGENT_RESULT = {
    "final_answer": "Your Python API experience is relevant to this role.",
    "tools_used": ["analyze_candidate_fit", "tailor_resume"],
    "tool_call_count": 2,
}
JD_EXTRACTION = {
    "job_title": "AI Engineer",
    "seniority_level": "Senior",
    "responsibilities": ["Build production LLM services"],
    "required_skills": ["Python", "APIs"],
    "preferred_skills": ["Kubernetes"],
    "required_experience": ["5+ years of software engineering experience"],
    "preferred_experience": [],
    "education_requirements": ["Bachelor's degree or equivalent experience"],
    "tools_and_technologies": ["Python", "Kubernetes"],
    "domain_knowledge": ["Generative AI"],
    "soft_skills": ["Cross-functional collaboration"],
}


def _pdf_bytes(text: str = RESUME_TEXT) -> bytes:
    pdf_buffer = BytesIO()
    pdf = canvas.Canvas(pdf_buffer)
    pdf.drawString(72, 720, text)
    pdf.save()
    return pdf_buffer.getvalue()


def _response(data: dict) -> Mock:
    response = Mock()
    response.json.return_value = data
    return response


def _widget_by_label(widgets, label: str):
    return next(widget for widget in widgets if widget.label == label)


def _button(app: AppTest, label: str):
    return _widget_by_label(app.button, label)


def _base_app(input_method: str = "Upload resume PDF") -> AppTest:
    app = AppTest.from_file(str(APP_PATH)).run()
    if app.radio[0].value != input_method:
        app.radio[0].set_value(input_method).run()
    _widget_by_label(app.text_input, "Target role").set_value(
        PROFILE_VALUES["target_role"]
    )
    _widget_by_label(app.text_area, "Job description").set_value(
        PROFILE_VALUES["job_description"]
    )
    return app


def _manual_app() -> AppTest:
    app = _base_app("Enter profile manually")
    _widget_by_label(app.text_area, "Current background").set_value(
        PROFILE_VALUES["current_background"]
    )
    _widget_by_label(app.text_input, "Skills (comma-separated)").set_value(
        PROFILE_VALUES["skills_text"]
    )
    _widget_by_label(app.text_area, "Project experience").set_value(
        PROFILE_VALUES["project_experience"]
    )
    return app


def _upload_resume(app: AppTest) -> bytes:
    pdf_bytes = _pdf_bytes()
    _widget_by_label(app.file_uploader, "Upload your resume").set_value(
        ("resume.pdf", pdf_bytes, "application/pdf")
    )
    return pdf_bytes


def test_pdf_input_shows_upload_and_hides_manual_fields() -> None:
    app = _base_app()

    assert app.radio[0].value == "Upload resume PDF"
    assert [uploader.label for uploader in app.file_uploader] == ["Upload your resume"]
    assert "Current background" not in [area.label for area in app.text_area]
    assert "Skills (comma-separated)" not in [item.label for item in app.text_input]
    assert "Project experience" not in [area.label for area in app.text_area]
    assert "Rule-based Analysis" not in [button.label for button in app.button]
    assert any("Rule-based analysis is available" in caption.value for caption in app.caption)


def test_manual_input_shows_profile_fields_and_hides_upload() -> None:
    app = _manual_app()

    assert app.radio[0].value == "Enter profile manually"
    assert not app.file_uploader
    assert "Current background" in [area.label for area in app.text_area]
    assert "Skills (comma-separated)" in [item.label for item in app.text_input]
    assert "Project experience" in [area.label for area in app.text_area]
    assert "Rule-based Analysis" in [button.label for button in app.button]


@pytest.mark.parametrize("input_method", ["Upload resume PDF", "Enter profile manually"])
def test_target_role_and_job_description_are_shared(input_method: str) -> None:
    app = _base_app(input_method)

    assert len([item for item in app.text_input if item.label == "Target role"]) == 1
    assert len([area for area in app.text_area if area.label == "Job description"]) == 1


@patch("requests.post")
def test_pdf_ai_analysis_uses_resume_endpoint(mock_post: Mock) -> None:
    mock_post.return_value.json.return_value = RESUME_ANALYSIS
    app = _base_app()
    pdf_bytes = _upload_resume(app)

    _button(app, "AI Analysis").click().run()

    assert not app.exception
    mock_post.assert_called_once_with(
        "http://localhost:8000/analyze-resume/llm",
        files={"file": ("resume.pdf", pdf_bytes, "application/pdf")},
        data={"job_description": PROFILE_VALUES["job_description"]},
        timeout=60,
    )
    assert any(metric.label == "Match score" and metric.value == "82%" for metric in app.metric)


@patch("requests.post")
def test_manual_ai_analysis_uses_profile_endpoint(mock_post: Mock) -> None:
    mock_post.return_value.json.return_value = ANALYSIS
    app = _manual_app()

    _button(app, "AI Analysis").click().run()

    mock_post.assert_called_once_with(
        "http://localhost:8000/analyze-profile/llm",
        json=PROFILE_PAYLOAD,
        timeout=60,
    )


@patch("requests.post")
def test_pdf_career_agent_parses_resume_and_uses_extracted_text(mock_post: Mock) -> None:
    mock_post.side_effect = [
        _response({"filename": "resume.pdf", "text": RESUME_TEXT}),
        _response(CAREER_AGENT_RESULT),
    ]
    app = _base_app()
    pdf_bytes = _upload_resume(app)
    goal = "Assess my fit and improve my resume."
    _widget_by_label(
        app.text_area, "What would you like the Career Agent to do?"
    ).set_value(goal)

    _button(app, "Run Career Agent").click().run()

    assert mock_post.call_args_list[0].args == ("http://localhost:8000/parse-resume",)
    assert mock_post.call_args_list[0].kwargs == {
        "files": {"file": ("resume.pdf", pdf_bytes, "application/pdf")},
        "timeout": 60,
    }
    assert mock_post.call_args_list[1].args == ("http://localhost:8000/career-agent",)
    assert mock_post.call_args_list[1].kwargs == {
        "json": {
            "user_request": goal,
            "current_background": RESUME_TEXT,
            "target_role": PROFILE_VALUES["target_role"],
            "job_description": PROFILE_VALUES["job_description"],
            "skills": [],
            "project_experience": "",
        },
        "timeout": 60,
    }
    assert any(
        markdown.value == "- Analyze Candidate Fit" for markdown in app.markdown
    )
    assert any(metric.label == "Tool calls" and metric.value == "2" for metric in app.metric)


@patch("requests.post")
def test_manual_career_agent_uses_manual_profile_fields(mock_post: Mock) -> None:
    mock_post.return_value.json.return_value = CAREER_AGENT_RESULT
    app = _manual_app()
    goal = "Assess my fit."
    _widget_by_label(
        app.text_area, "What would you like the Career Agent to do?"
    ).set_value(goal)

    _button(app, "Run Career Agent").click().run()

    mock_post.assert_called_once_with(
        "http://localhost:8000/career-agent",
        json={"user_request": goal, **PROFILE_PAYLOAD},
        timeout=60,
    )
    assert any(
        CAREER_AGENT_RESULT["final_answer"] in markdown.value
        for markdown in app.markdown
    )


@patch("requests.post")
def test_pdf_tailoring_uses_upload_endpoint(mock_post: Mock) -> None:
    mock_post.return_value.json.return_value = {**TAILORING, "resume_text": RESUME_TEXT}
    app = _base_app()
    pdf_bytes = _upload_resume(app)

    _button(app, "Tailor Resume").click().run()

    mock_post.assert_called_once_with(
        "http://localhost:8000/tailor-resume/upload",
        files={"file": ("resume.pdf", pdf_bytes, "application/pdf")},
        data={"job_description": PROFILE_VALUES["job_description"]},
        timeout=60,
    )
    assert any(
        heading.value == "Resume Tailoring Recommendations" for heading in app.subheader
    )


@patch("requests.post")
def test_manual_tailoring_uses_profile_endpoint(mock_post: Mock) -> None:
    mock_post.return_value.json.return_value = TAILORING
    app = _manual_app()

    _button(app, "Tailor Resume").click().run()

    mock_post.assert_called_once_with(
        "http://localhost:8000/tailor-resume",
        json=PROFILE_PAYLOAD,
        timeout=60,
    )


@pytest.mark.parametrize("input_method", ["Upload resume PDF", "Enter profile manually"])
@patch("requests.post")
def test_jd_extraction_is_available_for_both_input_methods(
    mock_post: Mock,
    input_method: str,
) -> None:
    mock_post.return_value.json.return_value = JD_EXTRACTION
    app = _base_app(input_method)

    _button(app, "Extract JD Requirements").click().run()

    mock_post.assert_called_once_with(
        "http://localhost:8000/extract-jd",
        json={"job_description": PROFILE_VALUES["job_description"]},
        timeout=60,
    )
    assert any(
        heading.value == "Structured JD Requirements" for heading in app.subheader
    )


@patch("requests.post")
def test_manual_rule_based_analysis_uses_existing_endpoint(mock_post: Mock) -> None:
    mock_post.return_value.json.return_value = {**ANALYSIS, "score_breakdown": None}
    app = _manual_app()

    _button(app, "Rule-based Analysis").click().run()

    mock_post.assert_called_once_with(
        "http://localhost:8000/analyze-profile",
        json=PROFILE_PAYLOAD,
        timeout=60,
    )


@patch("requests.post")
def test_rate_limit_error_preserves_existing_analysis_result(mock_post: Mock) -> None:
    success = _response(ANALYSIS)
    rate_limit_response = Mock(status_code=429)
    mock_post.side_effect = [
        success,
        requests.HTTPError("internal detail", response=rate_limit_response),
    ]
    app = _manual_app()

    _button(app, "AI Analysis").click().run()
    _widget_by_label(
        app.text_area, "What would you like the Career Agent to do?"
    ).set_value("Assess my fit.")
    _button(app, "Run Career Agent").click().run()

    assert not app.exception
    assert app.error[0].value == (
        "Too many AI requests. Please wait a little and try again."
    )
    assert any(metric.label == "Match score" and metric.value == "82%" for metric in app.metric)
    assert "internal detail" not in app.error[0].value


@patch("requests.post")
def test_empty_job_description_does_not_call_backend(mock_post: Mock) -> None:
    app = _manual_app()
    _widget_by_label(app.text_area, "Job description").set_value("   ")

    _button(app, "AI Analysis").click().run()

    mock_post.assert_not_called()
    assert app.error[0].value == "Please enter a job description before continuing."
