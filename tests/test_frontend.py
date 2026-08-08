from pathlib import Path
from io import BytesIO
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
RESUME_ANALYSIS = {**ANALYSIS, "resume_text": "Python developer"}
TAILORING = {
    "top_changes": ["Lead with production Python work"],
    "skills_to_emphasize": ["Python"],
    "experiences_to_emphasize": ["Production API work"],
    "missing_keywords": ["LLM evaluation"],
    "bullet_rewrite_suggestions": ["Built a production API using Python."],
    "overall_advice": ["Lead with relevant production experience."],
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


def _pdf_bytes(text: str = "Python developer") -> bytes:
    pdf_buffer = BytesIO()
    pdf = canvas.Canvas(pdf_buffer)
    pdf.drawString(72, 720, text)
    pdf.save()
    return pdf_buffer.getvalue()


def _filled_app(mode: str | None = None) -> AppTest:
    app = AppTest.from_file(str(APP_PATH)).run()
    if mode is not None:
        app.radio[0].set_value(mode)
    app.text_area[1].set_value(PROFILE_VALUES["current_background"])
    app.text_input[0].set_value(PROFILE_VALUES["target_role"])
    app.text_area[2].set_value(PROFILE_VALUES["job_description"])
    app.text_input[1].set_value(PROFILE_VALUES["skills_text"])
    app.text_area[3].set_value(PROFILE_VALUES["project_experience"])
    return app


@pytest.mark.parametrize(
    ("mode", "expected_path", "expected_label"),
    [
        (None, "/analyze-profile/llm", "Gemini AI analysis"),
        ("Rule-based analysis", "/analyze-profile", "Rule-based analysis"),
    ],
)
@patch("requests.post")
def test_frontend_submits_selected_analysis_mode(
    mock_post: Mock,
    mode: str | None,
    expected_path: str,
    expected_label: str,
) -> None:
    mock_post.return_value.json.return_value = ANALYSIS

    app = _filled_app(mode)
    app.button[3].click().run()

    assert not app.exception
    assert app.radio[0].value == (mode or "AI analysis")
    mock_post.assert_called_once_with(
        f"http://localhost:8000{expected_path}",
        json={
            "current_background": PROFILE_VALUES["current_background"],
            "target_role": PROFILE_VALUES["target_role"],
            "job_description": PROFILE_VALUES["job_description"],
            "skills": ["Python", "APIs"],
            "project_experience": PROFILE_VALUES["project_experience"],
        },
        timeout=60,
    )
    assert app.info[0].value == f"Result source: {expected_label}"
    assert app.metric[0].value == "82%"
    assert [(metric.label, metric.value) for metric in app.metric[1:7]] == [
        ("Technical skills", "90%"),
        ("Domain experience", "70%"),
        ("Seniority", "60%"),
        ("Role-specific requirements", "80%"),
        ("Education", "100%"),
        ("Communication / leadership", "80%"),
    ]


@patch("requests.post")
def test_frontend_shows_safe_error_and_preserves_form_data(mock_post: Mock) -> None:
    mock_post.side_effect = requests.HTTPError("secret backend detail")

    app = _filled_app()
    app.button[3].click().run()

    assert not app.exception
    assert app.error[0].value == (
        "The profile analysis could not be completed. Please try again."
    )
    assert "secret backend detail" not in app.error[0].value
    assert app.text_area[1].value == PROFILE_VALUES["current_background"]
    assert app.text_input[0].value == PROFILE_VALUES["target_role"]
    assert app.text_area[2].value == PROFILE_VALUES["job_description"]
    assert app.text_input[1].value == PROFILE_VALUES["skills_text"]
    assert app.text_area[3].value == PROFILE_VALUES["project_experience"]


@patch("requests.post")
def test_frontend_shows_timeout_specific_error(mock_post: Mock) -> None:
    mock_post.side_effect = requests.ReadTimeout("internal timeout detail")

    app = _filled_app()
    app.button[3].click().run()

    assert not app.exception
    assert app.error[0].value == "The AI analysis took too long. Please try again."
    assert "internal timeout detail" not in app.error[0].value


@pytest.mark.parametrize("mode", ["AI analysis", "Rule-based analysis"])
@patch("requests.post")
def test_frontend_does_not_submit_empty_job_description(
    mock_post: Mock, mode: str
) -> None:
    app = _filled_app(mode)
    app.text_area[2].set_value("   ")

    app.button[3].click().run()

    assert not app.exception
    mock_post.assert_not_called()
    assert app.error[0].value == (
        "Please enter a job description before analyzing your profile."
    )
    assert app.text_area[2].value == "   "


@patch("requests.post")
def test_frontend_submits_resume_analysis(mock_post: Mock) -> None:
    mock_post.return_value.json.return_value = RESUME_ANALYSIS
    pdf_bytes = _pdf_bytes()
    app = AppTest.from_file(str(APP_PATH)).run()
    app.file_uploader[0].set_value(("resume.pdf", pdf_bytes, "application/pdf"))
    app.text_area[0].set_value(PROFILE_VALUES["job_description"])

    app.button[0].click().run()

    assert not app.exception
    assert app.file_uploader[0].label == "Upload your resume"
    assert app.text_area[0].label == "Target job description"
    assert app.button[0].label == "Analyze Resume"
    assert app.expander[0].label == "No resume? Enter profile manually"
    mock_post.assert_called_once_with(
        "http://localhost:8000/analyze-resume/llm",
        files={"file": ("resume.pdf", pdf_bytes, "application/pdf")},
        data={"job_description": PROFILE_VALUES["job_description"]},
        timeout=60,
    )
    assert app.metric[0].value == "82%"
    assert [(metric.label, metric.value) for metric in app.metric[1:7]] == [
        ("Technical skills", "90%"),
        ("Domain experience", "70%"),
        ("Seniority", "60%"),
        ("Role-specific requirements", "80%"),
        ("Education", "100%"),
        ("Communication / leadership", "80%"),
    ]
    assert [heading.value for heading in app.subheader[-3:]] == [
        "Strengths",
        "Skill gaps",
        "Next steps",
    ]


@patch("requests.post")
def test_frontend_can_tailor_first_then_analyze_with_both_results(
    mock_post: Mock,
) -> None:
    tailoring_response = Mock()
    tailoring_response.json.return_value = {
        **TAILORING,
        "resume_text": RESUME_ANALYSIS["resume_text"],
    }
    analysis_response = Mock()
    analysis_response.json.return_value = ANALYSIS
    mock_post.side_effect = [tailoring_response, analysis_response]
    pdf_bytes = _pdf_bytes()
    app = AppTest.from_file(str(APP_PATH)).run()
    app.file_uploader[0].set_value(("resume.pdf", pdf_bytes, "application/pdf"))
    app.text_area[0].set_value(PROFILE_VALUES["job_description"])

    assert [button.label for button in app.button[:2]] == [
        "Analyze Resume",
        "Tailor Resume",
    ]
    app.button[1].click().run()

    assert mock_post.call_args_list[0].args == (
        "http://localhost:8000/tailor-resume/upload",
    )
    assert app.code[0].value == TAILORING["bullet_rewrite_suggestions"][0]

    app.button[0].click().run()

    assert not app.exception
    assert mock_post.call_args_list[1].kwargs == {
        "json": {
            "current_background": RESUME_ANALYSIS["resume_text"],
            "target_role": "",
            "job_description": PROFILE_VALUES["job_description"],
            "skills": [],
            "project_experience": "",
        },
        "timeout": 60,
    }
    assert mock_post.call_args_list[1].args == ("http://localhost:8000/analyze-profile/llm",)
    assert app.metric[0].value == "82%"
    assert any(
        heading.value == "Resume Tailoring Recommendations"
        for heading in app.subheader
    )
    assert app.code[0].value == TAILORING["bullet_rewrite_suggestions"][0]


@patch("requests.post")
def test_frontend_analysis_persists_when_tailoring_runs(mock_post: Mock) -> None:
    analysis_response = Mock()
    analysis_response.json.return_value = RESUME_ANALYSIS
    tailoring_response = Mock()
    tailoring_response.json.return_value = TAILORING
    mock_post.side_effect = [analysis_response, tailoring_response]
    app = AppTest.from_file(str(APP_PATH)).run()
    app.file_uploader[0].set_value(
        ("resume.pdf", _pdf_bytes(), "application/pdf")
    )
    app.text_area[0].set_value(PROFILE_VALUES["job_description"])

    app.button[0].click().run()
    app.button[1].click().run()

    assert not app.exception
    assert mock_post.call_args_list[1].args == ("http://localhost:8000/tailor-resume",)
    assert app.metric[0].value == "82%"
    assert any(
        heading.value == "Resume Tailoring Recommendations"
        for heading in app.subheader
    )


@patch("requests.post")
def test_frontend_extracts_jd_without_resume(mock_post: Mock) -> None:
    mock_post.return_value.json.return_value = JD_EXTRACTION
    app = AppTest.from_file(str(APP_PATH)).run()
    app.text_area[0].set_value(PROFILE_VALUES["job_description"])

    assert app.button[2].label == "Extract JD Requirements"
    app.button[2].click().run()

    assert not app.exception
    mock_post.assert_called_once_with(
        "http://localhost:8000/extract-jd",
        json={"job_description": PROFILE_VALUES["job_description"]},
        timeout=60,
    )
    assert any(
        heading.value == "Structured JD Requirements" for heading in app.subheader
    )
    assert any("Python" in markdown.value for markdown in app.markdown)
    assert any(
        expander.label == "Copy all extracted requirements"
        for expander in app.expander
    )


@patch("requests.post")
def test_frontend_analysis_tailoring_and_jd_extraction_coexist(
    mock_post: Mock,
) -> None:
    analysis_response = Mock()
    analysis_response.json.return_value = RESUME_ANALYSIS
    tailoring_response = Mock()
    tailoring_response.json.return_value = TAILORING
    extraction_response = Mock()
    extraction_response.json.return_value = JD_EXTRACTION
    mock_post.side_effect = [
        analysis_response,
        tailoring_response,
        extraction_response,
    ]
    app = AppTest.from_file(str(APP_PATH)).run()
    app.file_uploader[0].set_value(
        ("resume.pdf", _pdf_bytes(), "application/pdf")
    )
    app.text_area[0].set_value(PROFILE_VALUES["job_description"])

    app.button[0].click().run()
    app.button[1].click().run()
    app.button[2].click().run()

    assert not app.exception
    assert app.metric[0].value == "82%"
    headings = [heading.value for heading in app.subheader]
    assert "Structured JD Requirements" in headings
    assert "Resume Tailoring Recommendations" in headings
    assert any(code.value.startswith("Job title: AI Engineer") for code in app.code)


@patch("requests.post")
def test_frontend_requires_resume_upload(mock_post: Mock) -> None:
    app = AppTest.from_file(str(APP_PATH)).run()
    app.text_area[0].set_value(PROFILE_VALUES["job_description"])

    app.button[0].click().run()

    assert not app.exception
    mock_post.assert_not_called()
    assert app.error[0].value == (
        "Please upload a PDF resume before starting the analysis."
    )


@patch("requests.post")
def test_frontend_requires_resume_job_description(mock_post: Mock) -> None:
    app = AppTest.from_file(str(APP_PATH)).run()
    app.file_uploader[0].set_value(
        ("resume.pdf", _pdf_bytes(), "application/pdf")
    )
    app.text_area[0].set_value("   ")

    app.button[0].click().run()

    assert not app.exception
    mock_post.assert_not_called()
    assert app.error[0].value == (
        "Please enter a job description before analyzing your resume."
    )


@patch("requests.post")
def test_frontend_handles_resume_backend_error(mock_post: Mock) -> None:
    mock_post.side_effect = requests.HTTPError("secret backend detail")
    app = AppTest.from_file(str(APP_PATH)).run()
    app.file_uploader[0].set_value(
        ("resume.pdf", _pdf_bytes(), "application/pdf")
    )
    app.text_area[0].set_value(PROFILE_VALUES["job_description"])

    app.button[0].click().run()

    assert not app.exception
    assert app.error[0].value == (
        "The resume analysis could not be completed. Please try again."
    )
    assert "secret backend detail" not in app.error[0].value


@patch("requests.post")
def test_frontend_preserves_analysis_after_rate_limit(mock_post: Mock) -> None:
    analysis_response = Mock()
    analysis_response.json.return_value = RESUME_ANALYSIS
    rate_limit_response = Mock(status_code=429)
    mock_post.side_effect = [
        analysis_response,
        requests.HTTPError("internal detail", response=rate_limit_response),
    ]
    app = AppTest.from_file(str(APP_PATH)).run()
    app.file_uploader[0].set_value(
        ("resume.pdf", _pdf_bytes(), "application/pdf")
    )
    app.text_area[0].set_value(PROFILE_VALUES["job_description"])

    app.button[0].click().run()
    app.button[1].click().run()

    assert not app.exception
    assert app.error[0].value == (
        "Too many AI requests. Please wait a little and try again."
    )
    assert app.metric[0].value == "82%"
    assert "internal detail" not in app.error[0].value


@patch("requests.post")
def test_frontend_rejects_oversized_jd_without_request(mock_post: Mock) -> None:
    app = AppTest.from_file(str(APP_PATH)).run()
    app.text_area[0].set_value("x" * 20_001)

    app.button[2].click().run()

    assert not app.exception
    mock_post.assert_not_called()
    assert app.error[0].value == (
        "The job description is too long. Maximum length is 20,000 characters."
    )


@patch("requests.post")
def test_frontend_shows_friendly_oversized_resume_error(mock_post: Mock) -> None:
    oversized_response = Mock(status_code=413)
    mock_post.side_effect = requests.HTTPError(
        "internal detail", response=oversized_response
    )
    app = AppTest.from_file(str(APP_PATH)).run()
    app.file_uploader[0].set_value(
        ("resume.pdf", _pdf_bytes(), "application/pdf")
    )
    app.text_area[0].set_value(PROFILE_VALUES["job_description"])

    app.button[0].click().run()

    assert not app.exception
    assert app.error[0].value == "The resume PDF is too large. Maximum size is 5 MB."
    assert "internal detail" not in app.error[0].value
