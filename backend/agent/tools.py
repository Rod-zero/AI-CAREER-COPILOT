"""Thin tool wrappers around existing career-analysis services."""

from backend.services.llm_service import (
    ProfileAnalysis,
    ResumeTailoringRecommendations,
    StructuredJobDescription,
    analyze_profile_with_gemini,
    extract_jd_with_gemini,
    tailor_resume_with_gemini,
)


def extract_job_requirements(job_description: str) -> StructuredJobDescription:
    """Provide structured job requirements to a future Career Agent."""
    return extract_jd_with_gemini(job_description)


def analyze_candidate_fit(
    current_background: str,
    target_role: str,
    job_description: str,
    skills: list[str],
    project_experience: str,
) -> ProfileAnalysis:
    """Provide a validated candidate-to-role fit analysis to a future Career Agent."""
    return analyze_profile_with_gemini(
        current_background=current_background,
        target_role=target_role,
        job_description=job_description,
        skills=skills,
        project_experience=project_experience,
    )


def tailor_resume(
    current_background: str,
    target_role: str,
    job_description: str,
    skills: list[str],
    project_experience: str,
) -> ResumeTailoringRecommendations:
    """Provide validated resume-tailoring guidance to a future Career Agent."""
    return tailor_resume_with_gemini(
        current_background=current_background,
        target_role=target_role,
        job_description=job_description,
        skills=skills,
        project_experience=project_experience,
    )
