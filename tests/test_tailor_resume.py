from io import BytesIO
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

from backend.main import app
from backend.services.llm_service import (
    InvalidModelOutputError,
    ResumeTailoringRecommendations,
    tailor_resume_with_gemini,
)

client = TestClient(app)
PROFILE = {
    "current_background": "Backend developer",
    "target_role": "AI Engineer",
    "job_description": "Build production LLM services using Python and evaluation.",
    "skills": ["Python", "APIs"],
    "project_experience": "Built and deployed a production API.",
}
RECOMMENDATIONS = {
    "top_changes": ["Lead with Python production experience"],
    "skills_to_emphasize": ["Python", "API development"],
    "experiences_to_emphasize": ["Production API deployment"],
    "missing_keywords": ["LLM evaluation"],
    "bullet_rewrite_suggestions": [
        "Built and deployed a production API using Python."
    ],
    "overall_advice": ["Lead with production engineering experience."],
}


def create_pdf_bytes(text: str) -> bytes:
    pdf_buffer = BytesIO()
    pdf = canvas.Canvas(pdf_buffer)
    pdf.drawString(72, 720, text)
    pdf.save()
    return pdf_buffer.getvalue()


@patch("backend.main.tailor_resume_with_gemini")
def test_tailor_resume_endpoint_returns_structured_output(mock_tailor: Mock) -> None:
    mock_tailor.return_value = ResumeTailoringRecommendations(**RECOMMENDATIONS)

    response = client.post("/tailor-resume", json=PROFILE)

    assert response.status_code == 200
    assert response.json() == RECOMMENDATIONS
    mock_tailor.assert_called_once_with(
        current_background=PROFILE["current_background"],
        target_role=PROFILE["target_role"],
        job_description=PROFILE["job_description"],
        skills=PROFILE["skills"],
        project_experience=PROFILE["project_experience"],
    )


@patch("backend.main.tailor_resume_with_gemini")
def test_tailor_resume_endpoint_maps_invalid_output(mock_tailor: Mock) -> None:
    mock_tailor.side_effect = InvalidModelOutputError("raw model output")

    response = client.post("/tailor-resume", json=PROFILE)

    assert response.status_code == 502
    assert response.json() == {"detail": "The LLM service returned an invalid response."}


@patch("backend.main.tailor_resume_with_gemini")
def test_tailor_uploaded_resume_returns_reusable_extracted_text(
    mock_tailor: Mock,
) -> None:
    mock_tailor.return_value = ResumeTailoringRecommendations(**RECOMMENDATIONS)
    resume_text = "Python developer with API experience"

    response = client.post(
        "/tailor-resume/upload",
        files={"file": ("resume.pdf", create_pdf_bytes(resume_text), "application/pdf")},
        data={"job_description": PROFILE["job_description"]},
    )

    assert response.status_code == 200
    assert resume_text in response.json()["resume_text"]
    assert response.json()["top_changes"] == RECOMMENDATIONS["top_changes"]
    assert resume_text in mock_tailor.call_args.kwargs["current_background"]


@patch("backend.services.llm_service.genai.Client")
def test_tailor_resume_service_validates_output(mock_client: Mock, monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    live_client = mock_client.return_value.__enter__.return_value
    live_client.models.generate_content.return_value.text = (
        '{"top_changes":["Lead with Python experience"],'
        '"skills_to_emphasize":["Python"],'
        '"experiences_to_emphasize":["API deployment"],'
        '"missing_keywords":["LLM evaluation"],'
        '"bullet_rewrite_suggestions":["Built and deployed a production API."],'
        '"overall_advice":["Prioritize relevant experience."]}'
    )

    result = tailor_resume_with_gemini(**PROFILE)

    assert result.model_dump()["missing_keywords"] == ["LLM evaluation"]
    prompt = live_client.models.generate_content.call_args.kwargs["contents"]
    config = live_client.models.generate_content.call_args.kwargs["config"]
    assert config == {
        "response_mime_type": "application/json",
        "response_json_schema": ResumeTailoringRecommendations.model_json_schema(),
    }
    assert "Compare the candidate profile" in prompt
    assert "Do not invent or imply unsupported business impact" in prompt
    assert "3-5 highest-impact changes" in prompt
    assert "approximately 8-12 high-value missing keywords" in prompt
    assert "preserve every factual claim and metric" in prompt
    assert "supporting business strategy decisions" in prompt
    assert "if you have experience with..." in prompt
    assert "Return JSON only" in prompt


@patch("backend.services.llm_service.genai.Client")
def test_tailor_resume_service_rejects_invalid_output(
    mock_client: Mock, monkeypatch
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    live_client = mock_client.return_value.__enter__.return_value
    live_client.models.generate_content.return_value.text = '{"missing_keywords": []}'

    with pytest.raises(InvalidModelOutputError):
        tailor_resume_with_gemini(**PROFILE)
