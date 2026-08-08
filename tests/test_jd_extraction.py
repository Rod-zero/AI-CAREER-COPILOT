import json
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.llm_service import (
    GeminiRequestError,
    InvalidModelOutputError,
    StructuredJobDescription,
    extract_jd_with_gemini,
)

client = TestClient(app)
JOB_DESCRIPTION = (
    "Senior AI Engineer. Requires 5+ years of Python experience and a bachelor's "
    "degree. Kubernetes is preferred. Lead delivery with cross-functional teams."
)
EXTRACTION = {
    "job_title": "AI Engineer",
    "seniority_level": "Senior",
    "responsibilities": ["Lead delivery with cross-functional teams"],
    "required_skills": ["Python"],
    "preferred_skills": ["Kubernetes"],
    "required_experience": ["5+ years of Python experience"],
    "preferred_experience": [],
    "education_requirements": ["Bachelor's degree"],
    "tools_and_technologies": ["Python", "Kubernetes"],
    "domain_knowledge": [],
    "soft_skills": ["Cross-functional collaboration"],
}


@patch("backend.main.extract_jd_with_gemini")
def test_extract_jd_endpoint_returns_validated_response(mock_extract: Mock) -> None:
    mock_extract.return_value = StructuredJobDescription(**EXTRACTION)

    response = client.post("/extract-jd", json={"job_description": JOB_DESCRIPTION})

    assert response.status_code == 200
    assert response.json() == EXTRACTION
    mock_extract.assert_called_once_with(JOB_DESCRIPTION)


@patch("backend.services.llm_service.genai.Client")
def test_extract_jd_separates_required_and_preferred(
    mock_client: Mock, monkeypatch
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    live_client = mock_client.return_value.__enter__.return_value
    live_client.models.generate_content.return_value.text = json.dumps(EXTRACTION)

    result = extract_jd_with_gemini(JOB_DESCRIPTION)

    assert result.required_skills == ["Python"]
    assert result.preferred_skills == ["Kubernetes"]
    assert result.preferred_experience == []
    assert result.domain_knowledge == []
    call = live_client.models.generate_content.call_args
    assert call.kwargs["config"] == {
        "response_mime_type": "application/json",
        "response_json_schema": StructuredJobDescription.model_json_schema(),
    }
    prompt = call.kwargs["contents"]
    assert "Do not invent, infer, or add requirements" in prompt
    assert "Preserve meaningful years-of-experience requirements" in prompt
    assert "do not turn every responsibility sentence into a skill" in prompt
    assert "Return an empty array" in prompt


@patch("backend.services.llm_service.genai.Client")
def test_extract_jd_accepts_empty_optional_categories(
    mock_client: Mock, monkeypatch
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    empty_extraction = {
        "job_title": None,
        "seniority_level": None,
        "responsibilities": [],
        "required_skills": [],
        "preferred_skills": [],
        "required_experience": [],
        "preferred_experience": [],
        "education_requirements": [],
        "tools_and_technologies": [],
        "domain_knowledge": [],
        "soft_skills": [],
    }
    live_client = mock_client.return_value.__enter__.return_value
    live_client.models.generate_content.return_value.text = json.dumps(empty_extraction)

    assert extract_jd_with_gemini("Join our team.").model_dump() == empty_extraction


@patch("backend.services.llm_service.genai.Client")
def test_extract_jd_rejects_invalid_model_output(
    mock_client: Mock, monkeypatch
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    live_client = mock_client.return_value.__enter__.return_value
    live_client.models.generate_content.return_value.text = '{"required_skills": []}'

    with pytest.raises(InvalidModelOutputError):
        extract_jd_with_gemini(JOB_DESCRIPTION)


@patch("backend.services.llm_service.genai.Client")
def test_extract_jd_wraps_gemini_request_failure(
    mock_client: Mock, monkeypatch
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    live_client = mock_client.return_value.__enter__.return_value
    live_client.models.generate_content.side_effect = RuntimeError("timeout")

    with pytest.raises(GeminiRequestError, match="request failed"):
        extract_jd_with_gemini(JOB_DESCRIPTION)


@patch("backend.main.extract_jd_with_gemini")
def test_extract_jd_endpoint_maps_invalid_output(mock_extract: Mock) -> None:
    mock_extract.side_effect = InvalidModelOutputError("raw output")

    response = client.post("/extract-jd", json={"job_description": JOB_DESCRIPTION})

    assert response.status_code == 502
    assert response.json() == {"detail": "The LLM service returned an invalid response."}


@patch("backend.main.extract_jd_with_gemini")
def test_extract_jd_endpoint_maps_request_failure(mock_extract: Mock) -> None:
    mock_extract.side_effect = GeminiRequestError("SDK detail")

    response = client.post("/extract-jd", json={"job_description": JOB_DESCRIPTION})

    assert response.status_code == 502
    assert response.json() == {"detail": "The LLM service is currently unavailable."}
