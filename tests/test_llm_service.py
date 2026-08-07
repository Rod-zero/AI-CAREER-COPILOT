from datetime import date
from unittest.mock import Mock, patch

import pytest

from backend.services.llm_service import (
    GeminiRequestError,
    InvalidModelOutputError,
    MissingAPIKeyError,
    ScoreBreakdown,
    analyze_profile_with_gemini,
    calculate_weighted_score,
)


@patch("backend.services.llm_service.genai.Client")
def test_analyze_profile_returns_validated_analysis(mock_client: Mock, monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    live_client = mock_client.return_value.__enter__.return_value
    live_client.models.generate_content.return_value.text = (
        '{"match_score": 75, "strengths": ["Python"], '
        '"skill_gaps": ["Deployment"], "next_steps": ["Ship a project"], '
        '"score_breakdown": {"technical_skills": 80, "domain_experience": 40, '
        '"seniority": 20, "role_specific_requirements": 60, "education": 100, '
        '"communication_leadership": 70}}'
    )

    result = analyze_profile_with_gemini(
        current_background="Backend developer",
        target_role="AI Engineer",
        job_description="Build production AI services with Python and Kubernetes.",
        skills=["Python"],
        project_experience="Built an API",
    )

    assert result.model_dump() == {
        "match_score": 57,
        "score_breakdown": {
            "technical_skills": 80,
            "domain_experience": 40,
            "seniority": 20,
            "role_specific_requirements": 60,
            "education": 100,
            "communication_leadership": 70,
        },
        "strengths": ["Python"],
        "skill_gaps": ["Deployment"],
        "next_steps": ["Ship a project"],
    }
    mock_client.assert_called_once_with(api_key="test-key")
    mock_client.return_value.__enter__.assert_called_once_with()
    mock_client.return_value.__exit__.assert_called_once()
    call = live_client.models.generate_content.call_args
    assert call.kwargs["model"] == "test-model"
    assert call.kwargs["config"] == {"response_mime_type": "application/json"}
    assert "Backend developer" in call.kwargs["contents"]
    assert "Build production AI services with Python and Kubernetes." in call.kwargs[
        "contents"
    ]
    prompt = call.kwargs["contents"]
    assert "only explicitly stated candidate information as confirmed evidence" in prompt
    assert "Do not infer a specific platform, tool, cloud environment" in prompt
    assert "'Confirmed skill gap' only when the candidate explicitly states" in prompt
    assert "'Not evidenced in the supplied profile.'" in prompt
    assert "Absence of evidence is not proof" in prompt
    assert "required job qualifications more heavily than preferred ones" in prompt
    assert "missing preferred qualification should reduce the score only modestly" in prompt
    assert "Prioritize the most important requirements" in prompt
    assert "at most 5 strengths, 5 skill_gaps, and 5 next_steps" in prompt
    assert "no overlapping or repetitive bullets" in prompt
    assert "Order next_steps by highest expected impact" in prompt
    assert "directly to the candidate with an actionable verb" in prompt
    assert "Never write next_steps as questions" in prompt
    assert "concrete resume, portfolio, project, and learning actions" in prompt
    assert "Ground every strength, skill gap, and next step" in prompt
    assert "Do not invent experience, credentials, dates" in prompt
    assert f"The current date is {date.today().isoformat()}" in prompt
    assert "Do not infer that a degree is still in progress" in prompt
    assert "partial match / transferable experience" in prompt
    assert "hands-on RAG or LLM implementation" in prompt
    assert "core technical skills / ML / data science 30%" in prompt
    assert "relevant domain experience 20%" in prompt
    assert "seniority / years / scope of responsibility 20%" in prompt
    assert "governance, or production requirements 15%" in prompt
    assert "education / foundational qualifications 5%" in prompt
    assert "communication / leadership / collaboration" in prompt
    assert "Do not let one missing requirement dominate the score" in prompt
    assert "Major hard gaps" in prompt
    assert "resume-positioning improvements from genuine experience" in prompt
    assert "Score each dimension independently from 0 to 100" in prompt
    assert "Do not calculate an exact total duration" in prompt
    assert "score_breakdown (an object containing integer scores" in prompt
    assert "technical_skills, domain_experience, seniority" in prompt


def test_calculate_weighted_score() -> None:
    score_breakdown = ScoreBreakdown(
        technical_skills=80,
        domain_experience=40,
        seniority=20,
        role_specific_requirements=60,
        education=100,
        communication_leadership=70,
    )

    assert calculate_weighted_score(score_breakdown) == 57


def test_analyze_profile_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(MissingAPIKeyError, match="GEMINI_API_KEY"):
        analyze_profile_with_gemini(
            "Developer", "AI Engineer", "Python required.", ["Python"], "API"
        )


@patch("backend.services.llm_service.genai.Client")
@pytest.mark.parametrize(
    "model_output",
    [
        "not json",
        '{"match_score": 101, "strengths": [], "skill_gaps": [], "next_steps": []}',
        '{"match_score": 50, "strengths": []}',
        (
            '{"match_score": 50, "strengths": [], "skill_gaps": [], '
            '"next_steps": [], "score_breakdown": {"technical_skills": 101, '
            '"domain_experience": 50, "seniority": 50, '
            '"role_specific_requirements": 50, "education": 50, '
            '"communication_leadership": 50}}'
        ),
    ],
)
def test_analyze_profile_rejects_invalid_model_output(
    mock_client: Mock, monkeypatch, model_output: str
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    live_client = mock_client.return_value.__enter__.return_value
    live_client.models.generate_content.return_value.text = model_output

    with pytest.raises(InvalidModelOutputError):
        analyze_profile_with_gemini(
            "Developer", "AI Engineer", "Python required.", ["Python"], "API"
        )


@patch("backend.services.llm_service.genai.Client")
def test_analyze_profile_wraps_gemini_request_failures(
    mock_client: Mock, monkeypatch
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    live_client = mock_client.return_value.__enter__.return_value
    live_client.models.generate_content.side_effect = RuntimeError("timeout")

    with pytest.raises(GeminiRequestError, match="request failed"):
        analyze_profile_with_gemini(
            "Developer", "AI Engineer", "Python required.", ["Python"], "API"
        )
