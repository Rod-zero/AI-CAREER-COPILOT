from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.llm_service import (
    GeminiRequestError,
    InvalidModelOutputError,
    MissingAPIKeyError,
    ProfileAnalysis,
)

client = TestClient(app)
PROFILE = {
    "current_background": "Backend developer",
    "target_role": "AI Engineer",
    "job_description": "Build production LLM services using Python and APIs.",
    "skills": ["Python", "APIs"],
    "project_experience": "Built and deployed a production API.",
}


@patch("backend.main.analyze_profile_with_gemini")
def test_llm_endpoint_returns_profile_analysis(mock_analyze: Mock) -> None:
    mock_analyze.return_value = ProfileAnalysis(
        match_score=82,
        strengths=["Python", "API development"],
        skill_gaps=["LLM evaluation"],
        next_steps=["Build an evaluated LLM application"],
    )

    response = client.post("/analyze-profile/llm", json=PROFILE)

    assert response.status_code == 200
    assert response.json() == {
        "match_score": 82,
        "strengths": ["Python", "API development"],
        "skill_gaps": ["LLM evaluation"],
        "next_steps": ["Build an evaluated LLM application"],
    }
    mock_analyze.assert_called_once_with(
        current_background=PROFILE["current_background"],
        target_role=PROFILE["target_role"],
        job_description=PROFILE["job_description"],
        skills=PROFILE["skills"],
        project_experience=PROFILE["project_experience"],
    )


@pytest.mark.parametrize(
    ("service_error", "expected_status", "expected_detail"),
    [
        (
            MissingAPIKeyError("secret internal detail"),
            500,
            "LLM profile analysis is not configured.",
        ),
        (
            InvalidModelOutputError("raw model output"),
            502,
            "The LLM service returned an invalid response.",
        ),
        (
            GeminiRequestError("SDK failure detail"),
            502,
            "The LLM service is currently unavailable.",
        ),
    ],
)
@patch("backend.main.analyze_profile_with_gemini")
def test_llm_endpoint_maps_service_errors(
    mock_analyze: Mock,
    service_error: Exception,
    expected_status: int,
    expected_detail: str,
) -> None:
    mock_analyze.side_effect = service_error

    response = client.post("/analyze-profile/llm", json=PROFILE)

    assert response.status_code == expected_status
    assert response.json() == {"detail": expected_detail}


@patch("backend.main.analyze_profile_with_gemini")
def test_deterministic_endpoint_still_works_without_llm(mock_analyze: Mock) -> None:
    response = client.post("/analyze-profile", json=PROFILE)

    assert response.status_code == 200
    assert response.json()["match_score"] == 33
    mock_analyze.assert_not_called()
