"""Gemini-driven orchestration for the AI Career Copilot."""

import json
import os
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar, cast

from google import genai
from google.genai import types
from pydantic import BaseModel, ConfigDict

from backend.agent.tools import (
    analyze_candidate_fit,
    extract_job_requirements,
    tailor_resume,
)
from backend.services.llm_service import (
    GeminiRequestError,
    InvalidModelOutputError,
    MissingAPIKeyError,
)

SYSTEM_INSTRUCTION = (
    "Act as an AI Career Copilot. Use the available tools when they are useful "
    "for the user's request, and let the request determine which tools to call. "
    "Do not call irrelevant tools or assume that every tool must be used. Base all "
    "recommendations only on the supplied candidate and job evidence. Never "
    "fabricate resume experience, skills, employers, education, credentials, "
    "accomplishments, or metrics. After any useful tool calls, provide a concise "
    "final career recommendation that directly addresses the user's goal."
)

ToolCallable = TypeVar("ToolCallable", bound=Callable[..., Any])


class CareerAgentResult(BaseModel):
    """Inspectable result of one Career Agent run."""

    model_config = ConfigDict(extra="forbid")

    final_answer: str
    tools_used: list[str]
    tool_call_count: int


class _ToolInvocationRecorder:
    """Record automatic tool calls and retain errors swallowed by SDK AFC."""

    def __init__(self) -> None:
        self.tools_used: list[str] = []
        self.errors: list[Exception] = []

    def instrument(self, tool: ToolCallable) -> ToolCallable:
        @wraps(tool)
        def recorded_tool(*args: Any, **kwargs: Any) -> Any:
            self.tools_used.append(tool.__name__)
            try:
                return tool(*args, **kwargs)
            except Exception as exc:
                self.errors.append(exc)
                raise

        return cast(ToolCallable, recorded_tool)


def run_career_agent(
    user_request: str,
    current_background: str,
    target_role: str,
    job_description: str,
    skills: list[str],
    project_experience: str,
) -> CareerAgentResult:
    """Run one model-directed Career Agent turn with automatic tool calling."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise MissingAPIKeyError("GEMINI_API_KEY is not configured.")

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    context = {
        "user_request": user_request,
        "current_background": current_background,
        "target_role": target_role,
        "job_description": job_description,
        "skills": skills,
        "project_experience": project_experience,
    }
    prompt = (
        "Respond to the user request using the supplied career context. Select and "
        "call only the tools that materially help answer the request. Pass tool "
        "arguments exactly from the supplied context.\n\nCareer context:\n"
        f"{json.dumps(context, ensure_ascii=False)}"
    )

    recorder = _ToolInvocationRecorder()
    available_tools = [
        recorder.instrument(extract_job_requirements),
        recorder.instrument(analyze_candidate_fit),
        recorder.instrument(tailor_resume),
    ]
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        tools=available_tools,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            maximum_remote_calls=6,
        ),
    )

    try:
        with genai.Client(api_key=api_key) as client:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
    except Exception as exc:
        if recorder.errors:
            raise recorder.errors[0]
        raise GeminiRequestError("Gemini Career Agent request failed.") from exc

    if recorder.errors:
        raise recorder.errors[0]

    final_answer = response.text
    if not isinstance(final_answer, str) or not final_answer.strip():
        raise InvalidModelOutputError(
            "Gemini returned an empty Career Agent response."
        )

    return CareerAgentResult(
        final_answer=final_answer.strip(),
        tools_used=recorder.tools_used,
        tool_call_count=len(recorder.tools_used),
    )
