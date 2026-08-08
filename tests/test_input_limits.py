from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from backend.config import MAX_BACKGROUND_CHARS, MAX_JD_CHARS, MAX_RESUME_SIZE_BYTES
from backend.main import app

client = TestClient(app)


@patch("backend.main.analyze_profile_with_gemini")
def test_oversized_resume_is_rejected_before_gemini(mock_analyze: Mock) -> None:
    response = client.post(
        "/analyze-resume/llm",
        files={
            "file": (
                "resume.pdf",
                b"x" * (MAX_RESUME_SIZE_BYTES + 1),
                "application/pdf",
            )
        },
        data={"job_description": "Build AI services."},
    )

    assert response.status_code == 413
    assert "no larger than" in response.json()["detail"]
    mock_analyze.assert_not_called()


@patch("backend.main.extract_jd_with_gemini")
def test_oversized_jd_is_rejected_before_gemini(mock_extract: Mock) -> None:
    response = client.post(
        "/extract-jd",
        json={"job_description": "x" * (MAX_JD_CHARS + 1)},
    )

    assert response.status_code == 422
    mock_extract.assert_not_called()


@patch("backend.main.analyze_profile_with_gemini")
def test_oversized_manual_background_is_rejected_before_gemini(
    mock_analyze: Mock,
) -> None:
    response = client.post(
        "/analyze-profile/llm",
        json={
            "current_background": "x" * (MAX_BACKGROUND_CHARS + 1),
            "target_role": "AI Engineer",
            "job_description": "Build AI services.",
            "skills": ["Python"],
            "project_experience": "Built an API.",
        },
    )

    assert response.status_code == 422
    mock_analyze.assert_not_called()

