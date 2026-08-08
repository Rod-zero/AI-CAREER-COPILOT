"""Gemini-backed profile analysis service."""

import json
import logging
import os
from datetime import date

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, ConfigDict, Field, ValidationError

load_dotenv()

logger = logging.getLogger(__name__)


class LLMServiceError(RuntimeError):
    """Base exception for profile-analysis failures."""


class MissingAPIKeyError(LLMServiceError):
    """Raised when Gemini credentials are not configured."""


class InvalidModelOutputError(LLMServiceError):
    """Raised when Gemini returns malformed or unexpected data."""


class GeminiRequestError(LLMServiceError):
    """Raised when a request to Gemini fails."""


class ScoreBreakdown(BaseModel):
    """Validated dimension scores returned by Gemini."""

    model_config = ConfigDict(extra="forbid", strict=True)

    technical_skills: int = Field(ge=0, le=100)
    domain_experience: int = Field(ge=0, le=100)
    seniority: int = Field(ge=0, le=100)
    role_specific_requirements: int = Field(ge=0, le=100)
    education: int = Field(ge=0, le=100)
    communication_leadership: int = Field(ge=0, le=100)


class ProfileAnalysis(BaseModel):
    """Validated structure returned by Gemini."""

    model_config = ConfigDict(extra="forbid", strict=True)

    match_score: int = Field(ge=0, le=100)
    score_breakdown: ScoreBreakdown
    strengths: list[str]
    skill_gaps: list[str]
    next_steps: list[str]


class ResumeTailoringRecommendations(BaseModel):
    """Validated resume-tailoring structure returned by Gemini."""

    model_config = ConfigDict(extra="forbid", strict=True)

    top_changes: list[str]
    skills_to_emphasize: list[str]
    experiences_to_emphasize: list[str]
    missing_keywords: list[str]
    bullet_rewrite_suggestions: list[str]
    overall_advice: list[str]


class StructuredJobDescription(BaseModel):
    """Validated requirements extracted from a job description."""

    model_config = ConfigDict(extra="forbid", strict=True)

    job_title: str | None
    seniority_level: str | None
    responsibilities: list[str]
    required_skills: list[str]
    preferred_skills: list[str]
    required_experience: list[str]
    preferred_experience: list[str]
    education_requirements: list[str]
    tools_and_technologies: list[str]
    domain_knowledge: list[str]
    soft_skills: list[str]


def calculate_weighted_score(score_breakdown: ScoreBreakdown) -> int:
    """Calculate the overall score from the documented rubric weights."""
    return round(
        score_breakdown.technical_skills * 0.30
        + score_breakdown.domain_experience * 0.20
        + score_breakdown.seniority * 0.20
        + score_breakdown.role_specific_requirements * 0.15
        + score_breakdown.education * 0.05
        + score_breakdown.communication_leadership * 0.10
    )


def analyze_profile_with_gemini(
    current_background: str,
    target_role: str,
    job_description: str,
    skills: list[str],
    project_experience: str,
) -> ProfileAnalysis:
    """Analyze a candidate profile with Gemini and return validated output."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise MissingAPIKeyError("GEMINI_API_KEY is not configured.")

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    profile = {
        "current_background": current_background,
        "target_role": target_role,
        "job_description": job_description,
        "skills": skills,
        "project_experience": project_experience,
    }
    current_date = date.today().isoformat()
    prompt = (
        "Analyze the candidate primarily against the supplied job description. "
        "Identify its important required and preferred qualifications, then compare "
        "them only against evidence explicitly supplied in the candidate profile. "
        "Treat only explicitly stated candidate information as confirmed evidence. "
        "Ground every strength, skill gap, and next step in the supplied candidate "
        "profile and job description. Do not invent experience, credentials, dates, "
        "education status, or employment status. The current date is "
        f"{current_date}. Use it when interpreting dates. Do not infer that a degree "
        "is still in progress solely from a date range when its end date is in the "
        "past. If completion or employment status remains ambiguous, describe it as "
        "not evidenced instead of guessing. "
        "Do not infer a specific platform, tool, cloud environment, or depth of "
        "experience unless it is stated. Do not invent requirements unsupported by "
        "the job description. Use 'Confirmed skill gap' only when the candidate "
        "explicitly states that they lack the skill or experience. When candidate "
        "evidence for a qualification is absent, label it 'Not evidenced in the "
        "supplied profile.' Absence of evidence is not proof that the candidate "
        "lacks the qualification. Classify relevant evidence as a strong match, a "
        "partial match / transferable experience, or not evidenced / major gap. "
        "Credit adjacent experience as a partial match: for example, hands-on RAG "
        "or LLM implementation is relevant transferable experience for a GenAI "
        "technical leadership or Agentic AI requirement, but does not by itself "
        "prove leadership scope or Agentic AI experience. Use precise gap wording "
        "such as 'Not evidenced in the supplied resume...', 'Partial match: "
        "candidate has X, but the role requires Y...', or 'Major gap: ...'. "
        "Evaluate these weighted dimensions separately before assigning the overall "
        "match_score: core technical skills / ML / data science 30%; relevant domain "
        "experience 20%; seniority / years / scope of responsibility 20%; "
        "role-specific tools, systems, governance, or production requirements 15%; "
        "education / foundational qualifications 5%; and communication / leadership "
        "/ collaboration / other job requirements 10%. Score each dimension "
        "independently from 0 to 100 based only on evidence in the supplied profile "
        "and job description. Calculate a calibrated match_score from 0 to 100 using "
        "this rubric, "
        "weighting required job qualifications more heavily than preferred ones; a "
        "missing preferred qualification should reduce the score only modestly. "
        "Do not let one missing requirement dominate the score unless the job "
        "description makes it mandatory and central. Major hard gaps such as a "
        "required 5-7 years of professional experience, specific regulated-domain "
        "experience, or required leadership scope must materially reduce the score. "
        "Do not calculate an exact total duration of professional experience unless "
        "the supplied profile explicitly provides enough clear information. Prefer "
        "wording such as 'the supplied resume does not evidence the required 5-7 "
        "years of professional experience.' "
        "Prioritize the most important requirements from the job description. Return "
        "at most 5 strengths, 5 skill_gaps, and 5 next_steps, with no overlapping or "
        "repetitive bullets. Order next_steps by highest expected impact on the "
        "candidate's fit. Write every next_steps item directly to the candidate with "
        "an actionable verb such as "
        "build, add, learn, demonstrate, quantify, tailor, or document. Never write "
        "next_steps as questions or actions for a recruiter or interviewer. Prefer "
        "concrete resume, portfolio, project, and learning actions. Distinguish "
        "resume-positioning improvements from genuine experience or skill gaps that "
        "cannot be fixed through wording alone. Keep all results "
        "concise and actionable. Do not include or recommend hidden chain-of-thought "
        "or private reasoning. Return only JSON with exactly these fields: "
        "match_score (an integer from 0 to 100), strengths (an array of strings), "
        "skill_gaps (an array of strings), next_steps (an array of strings), and "
        "score_breakdown (an object containing integer scores from 0 to 100 for "
        "technical_skills, domain_experience, seniority, role_specific_requirements, "
        "education, and communication_leadership)."
        "\n\nCandidate profile and job description:\n"
        f"{json.dumps(profile, ensure_ascii=False)}"
    )

    try:
        with genai.Client(api_key=api_key) as client:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={"response_mime_type": "application/json"},
            )
    except Exception as exc:
        raise GeminiRequestError("Gemini profile analysis request failed.") from exc

    try:
        analysis = ProfileAnalysis.model_validate_json(response.text)
        return analysis.model_copy(
            update={"match_score": calculate_weighted_score(analysis.score_breakdown)}
        )
    except (TypeError, ValueError, ValidationError) as exc:
        raise InvalidModelOutputError(
            "Gemini returned invalid profile analysis output."
        ) from exc


def tailor_resume_with_gemini(
    current_background: str,
    target_role: str,
    job_description: str,
    skills: list[str],
    project_experience: str,
) -> ResumeTailoringRecommendations:
    """Generate validated, evidence-based resume-tailoring recommendations."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise MissingAPIKeyError("GEMINI_API_KEY is not configured.")

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    profile = {
        "current_background": current_background,
        "target_role": target_role,
        "job_description": job_description,
        "skills": skills,
        "project_experience": project_experience,
    }
    prompt = (
        "Compare the candidate profile against the supplied job description and "
        "provide concise, actionable resume-tailoring recommendations. Return 3-5 "
        "highest-impact changes in top_changes, ordered by impact. Recommend which "
        "existing skills and experiences should be emphasized. Identify important "
        "keywords from the job description that are missing from the current profile "
        "or resume content. Return approximately 8-12 high-value missing keywords, "
        "prioritizing basic requirements and strong transferable requirements; do not "
        "dump every absent term or include near-duplicate domain keywords. Suggest "
        "stronger resume bullet wording using only facts explicitly present in the "
        "candidate profile. Rewrites may improve wording and emphasize evidenced "
        "transferable skills, but must preserve every factual claim and metric. Do not "
        "invent or imply unsupported business impact, stakeholder impact, strategic "
        "decision-making, tools, domain experience, responsibilities, technologies, "
        "metrics, accomplishments, or credentials. Never add phrases such as "
        "'supporting business strategy decisions' unless explicitly evidenced. "
        "Use conditional wording such as 'if you have experience with...' for skills "
        "or experience that are not evidenced in the supplied profile. "
        "When the supplied information is too vague for a concrete rewrite, recommend "
        "what the candidate should clarify instead of fabricating details. Return JSON "
        "only, with exactly these fields, each containing an array of strings: "
        "top_changes, skills_to_emphasize, experiences_to_emphasize, missing_keywords, "
        "bullet_rewrite_suggestions, and overall_advice."
        "\n\nCandidate profile and job description:\n"
        f"{json.dumps(profile, ensure_ascii=False)}"
    )

    try:
        with genai.Client(api_key=api_key) as client:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": (
                        ResumeTailoringRecommendations.model_json_schema()
                    ),
                },
            )
    except Exception as exc:
        logger.warning(
            "Gemini resume tailoring request failed: %s",
            type(exc).__name__,
        )
        raise GeminiRequestError("Gemini resume tailoring request failed.") from exc

    try:
        return ResumeTailoringRecommendations.model_validate_json(response.text)
    except (TypeError, ValueError, ValidationError) as exc:
        if isinstance(exc, ValidationError):
            error_locations = [
                ".".join(str(part) for part in error["loc"])
                for error in exc.errors(include_input=False)
            ]
            logger.warning(
                "Gemini resume tailoring output failed schema validation at: %s",
                ", ".join(error_locations),
            )
        else:
            logger.warning(
                "Gemini resume tailoring output was not valid JSON: %s",
                type(exc).__name__,
            )
        raise InvalidModelOutputError(
            "Gemini returned invalid resume tailoring output."
        ) from exc


def extract_jd_with_gemini(job_description: str) -> StructuredJobDescription:
    """Extract validated, reusable requirements from a job description."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise MissingAPIKeyError("GEMINI_API_KEY is not configured.")

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    prompt = (
        "Extract structured requirements only from the supplied job description. "
        "Do not invent, infer, or add requirements that are not actually present. "
        "Distinguish required from preferred qualifications only when the job "
        "description provides that distinction; otherwise place explicitly stated "
        "minimum qualifications in required fields. Preserve meaningful years-of-"
        "experience requirements and qualifiers such as minimum, preferred, or "
        "equivalent experience. Normalize obvious duplicate wording without losing "
        "meaning. Keep responsibilities as responsibilities and do not turn every "
        "responsibility sentence into a skill. Use null for job_title or "
        "seniority_level when not evidenced. Return an empty array for every list "
        "category that is not evidenced. Keep entries concise. Return JSON only and "
        "conform exactly to the supplied response schema."
        "\n\nJob description:\n"
        f"{job_description}"
    )

    try:
        with genai.Client(api_key=api_key) as client:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": StructuredJobDescription.model_json_schema(),
                },
            )
    except Exception as exc:
        logger.warning("Gemini JD extraction request failed: %s", type(exc).__name__)
        raise GeminiRequestError("Gemini JD extraction request failed.") from exc

    try:
        return StructuredJobDescription.model_validate_json(response.text)
    except (TypeError, ValueError, ValidationError) as exc:
        if isinstance(exc, ValidationError):
            error_locations = [
                ".".join(str(part) for part in error["loc"])
                for error in exc.errors(include_input=False)
            ]
            logger.warning(
                "Gemini JD extraction output failed schema validation at: %s",
                ", ".join(error_locations),
            )
        else:
            logger.warning(
                "Gemini JD extraction output was not valid JSON: %s",
                type(exc).__name__,
            )
        raise InvalidModelOutputError(
            "Gemini returned invalid JD extraction output."
        ) from exc
