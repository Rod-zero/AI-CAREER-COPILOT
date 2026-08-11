from collections.abc import Callable
from unittest.mock import Mock, patch

import pytest
from google.genai import types

from backend.agent.career_agent import SYSTEM_INSTRUCTION, run_career_agent
from backend.services.llm_service import InvalidModelOutputError


CONTEXT = {
    "user_request": "Assess my fit and suggest resume improvements.",
    "current_background": "Backend developer",
    "target_role": "AI Engineer",
    "job_description": "Build production LLM services using Python.",
    "skills": ["Python", "APIs"],
    "project_experience": "Built a production API.",
}


def _tools_by_name(config: types.GenerateContentConfig) -> dict[str, Callable[..., object]]:
    assert config.tools is not None
    return {tool.__name__: tool for tool in config.tools if callable(tool)}


@patch("backend.agent.career_agent.genai.Client")
@patch("backend.agent.tools.tailor_resume_with_gemini")
@patch("backend.agent.tools.analyze_profile_with_gemini")
@patch("backend.agent.tools.extract_jd_with_gemini")
def test_agent_uses_model_selected_tools_and_returns_final_answer(
    mock_extract: Mock,
    mock_analyze: Mock,
    mock_tailor: Mock,
    mock_client: Mock,
    monkeypatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    live_client = mock_client.return_value.__enter__.return_value

    def simulate_automatic_function_calling(**kwargs) -> Mock:
        tools = _tools_by_name(kwargs["config"])
        tools["analyze_candidate_fit"](
            current_background=CONTEXT["current_background"],
            target_role=CONTEXT["target_role"],
            job_description=CONTEXT["job_description"],
            skills=CONTEXT["skills"],
            project_experience=CONTEXT["project_experience"],
        )
        tools["tailor_resume"](
            current_background=CONTEXT["current_background"],
            target_role=CONTEXT["target_role"],
            job_description=CONTEXT["job_description"],
            skills=CONTEXT["skills"],
            project_experience=CONTEXT["project_experience"],
        )
        return Mock(text="Emphasize your production Python work.")

    live_client.models.generate_content.side_effect = simulate_automatic_function_calling

    result = run_career_agent(**CONTEXT)

    assert result.final_answer == "Emphasize your production Python work."
    assert result.tools_used == ["analyze_candidate_fit", "tailor_resume"]
    assert result.tool_call_count == 2
    mock_extract.assert_not_called()
    mock_analyze.assert_called_once()
    mock_tailor.assert_called_once()
    call = live_client.models.generate_content.call_args
    assert call.kwargs["model"] == "gemini-2.5-flash"
    assert isinstance(call.kwargs["config"], types.GenerateContentConfig)
    assert call.kwargs["config"].system_instruction == SYSTEM_INSTRUCTION
    assert call.kwargs["config"].automatic_function_calling.maximum_remote_calls == 6


@patch("backend.agent.career_agent.genai.Client")
@patch("backend.agent.tools.extract_jd_with_gemini")
def test_agent_can_use_only_one_relevant_tool(
    mock_extract: Mock,
    mock_client: Mock,
    monkeypatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    live_client = mock_client.return_value.__enter__.return_value

    def simulate_single_tool_call(**kwargs) -> Mock:
        tools = _tools_by_name(kwargs["config"])
        tools["extract_job_requirements"](CONTEXT["job_description"])
        return Mock(text="The role primarily requires Python and LLM delivery.")

    live_client.models.generate_content.side_effect = simulate_single_tool_call

    result = run_career_agent(**CONTEXT)

    assert result.tools_used == ["extract_job_requirements"]
    assert result.tool_call_count == 1
    mock_extract.assert_called_once_with(CONTEXT["job_description"])


@patch("backend.agent.career_agent.genai.Client")
@patch("backend.agent.tools.analyze_profile_with_gemini")
def test_agent_propagates_underlying_tool_failure(
    mock_analyze: Mock,
    mock_client: Mock,
    monkeypatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    service_error = InvalidModelOutputError("invalid tool response")
    mock_analyze.side_effect = service_error
    live_client = mock_client.return_value.__enter__.return_value

    def simulate_failed_tool_call(**kwargs) -> Mock:
        tools = _tools_by_name(kwargs["config"])
        tools["analyze_candidate_fit"](
            current_background=CONTEXT["current_background"],
            target_role=CONTEXT["target_role"],
            job_description=CONTEXT["job_description"],
            skills=CONTEXT["skills"],
            project_experience=CONTEXT["project_experience"],
        )
        return Mock(text="unreachable")

    live_client.models.generate_content.side_effect = simulate_failed_tool_call

    with pytest.raises(InvalidModelOutputError) as exc_info:
        run_career_agent(**CONTEXT)

    assert exc_info.value is service_error
