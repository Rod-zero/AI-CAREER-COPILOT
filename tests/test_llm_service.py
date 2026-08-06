from unittest.mock import Mock, patch

import pytest

from backend.services.llm_service import (
    GeminiRequestError,
    InvalidModelOutputError,
    MissingAPIKeyError,
    analyze_profile_with_gemini,
)


@patch("backend.services.llm_service.genai.Client")
def test_analyze_profile_returns_validated_analysis(mock_client: Mock, monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    live_client = mock_client.return_value.__enter__.return_value
    live_client.models.generate_content.return_value.text = (
        '{"match_score": 75, "strengths": ["Python"], '
        '"skill_gaps": ["Deployment"], "next_steps": ["Ship a project"]}'
    )

    result = analyze_profile_with_gemini(
        current_background="Backend developer",
        target_role="AI Engineer",
        job_description="Build production AI services with Python and Kubernetes.",
        skills=["Python"],
        project_experience="Built an API",
    )

    assert result.model_dump() == {
        "match_score": 75,
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
    assert "confirmed skill gap" in prompt
    assert "requirement that is merely not evidenced" in prompt
    assert "directly to the candidate with an actionable verb" in prompt
    assert "Never write next_steps as questions" in prompt
    assert "concrete resume, portfolio, project, and learning actions" in prompt


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
