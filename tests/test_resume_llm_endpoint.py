from io import BytesIO
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

from backend.main import app
from backend.services.llm_service import ProfileAnalysis, ScoreBreakdown

client = TestClient(app)


def create_pdf_bytes(text: str | None = None) -> bytes:
    pdf_buffer = BytesIO()
    pdf = canvas.Canvas(pdf_buffer)
    if text:
        pdf.drawString(72, 720, text)
    pdf.save()
    return pdf_buffer.getvalue()


@patch("backend.main.analyze_profile_with_gemini")
def test_analyze_resume_llm_returns_analysis_and_delegates(mock_analyze: Mock) -> None:
    resume_text = "Python developer with API experience"
    job_description = "Build production AI services."
    mock_analyze.return_value = ProfileAnalysis(
        match_score=80,
        score_breakdown=ScoreBreakdown(
            technical_skills=90,
            domain_experience=70,
            seniority=60,
            role_specific_requirements=80,
            education=100,
            communication_leadership=80,
        ),
        strengths=["Python"],
        skill_gaps=["LLM evaluation"],
        next_steps=["Build an evaluated LLM project"],
    )

    response = client.post(
        "/analyze-resume/llm",
        files={"file": ("resume.pdf", create_pdf_bytes(resume_text), "application/pdf")},
        data={"job_description": job_description},
    )

    assert response.status_code == 200
    assert response.json() == {
        "match_score": 80,
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
        "next_steps": ["Build an evaluated LLM project"],
        "resume_text": response.json()["resume_text"],
    }
    assert resume_text in response.json()["resume_text"]
    call = mock_analyze.call_args.kwargs
    assert resume_text in call["current_background"]
    assert call == {
        "current_background": call["current_background"],
        "target_role": "",
        "job_description": job_description,
        "skills": [],
        "project_experience": "",
    }


def test_analyze_resume_llm_rejects_non_pdf() -> None:
    response = client.post(
        "/analyze-resume/llm",
        files={"file": ("resume.txt", b"Not a PDF", "text/plain")},
        data={"job_description": "Build AI services."},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Uploaded file must be a PDF."}


def test_analyze_resume_llm_rejects_pdf_without_text() -> None:
    response = client.post(
        "/analyze-resume/llm",
        files={"file": ("blank.pdf", create_pdf_bytes(), "application/pdf")},
        data={"job_description": "Build AI services."},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "The PDF contains no extractable text."}


def test_analyze_resume_llm_rejects_empty_job_description() -> None:
    response = client.post(
        "/analyze-resume/llm",
        files={"file": ("resume.pdf", create_pdf_bytes("Python developer"), "application/pdf")},
        data={"job_description": "   "},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Job description must not be empty."}
