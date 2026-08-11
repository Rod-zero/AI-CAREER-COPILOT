from unittest.mock import Mock, patch

from backend.agent.tools import (
    analyze_candidate_fit,
    extract_job_requirements,
    tailor_resume,
)
from backend.services.llm_service import (
    ProfileAnalysis,
    ResumeTailoringRecommendations,
    ScoreBreakdown,
    StructuredJobDescription,
)


PROFILE = {
    "current_background": "Backend developer",
    "target_role": "AI Engineer",
    "job_description": "Build production LLM services.",
    "skills": ["Python", "APIs"],
    "project_experience": "Built a production API.",
}


@patch("backend.agent.tools.extract_jd_with_gemini")
def test_extract_job_requirements_delegates_and_preserves_result(
    mock_extract: Mock,
) -> None:
    expected = StructuredJobDescription(
        job_title="AI Engineer",
        seniority_level=None,
        responsibilities=["Build LLM services"],
        required_skills=["Python"],
        preferred_skills=[],
        required_experience=[],
        preferred_experience=[],
        education_requirements=[],
        tools_and_technologies=["Python"],
        domain_knowledge=["Generative AI"],
        soft_skills=[],
    )
    mock_extract.return_value = expected

    result = extract_job_requirements(PROFILE["job_description"])

    assert result is expected
    mock_extract.assert_called_once_with(PROFILE["job_description"])


@patch("backend.agent.tools.analyze_profile_with_gemini")
def test_analyze_candidate_fit_delegates_and_preserves_result(
    mock_analyze: Mock,
) -> None:
    expected = ProfileAnalysis(
        match_score=80,
        score_breakdown=ScoreBreakdown(
            technical_skills=90,
            domain_experience=70,
            seniority=60,
            role_specific_requirements=80,
            education=100,
            communication_leadership=80,
        ),
        strengths=["Python"],
        skill_gaps=["LLM evaluation"],
        next_steps=["Build an evaluation project"],
    )
    mock_analyze.return_value = expected

    result = analyze_candidate_fit(**PROFILE)

    assert result is expected
    mock_analyze.assert_called_once_with(**PROFILE)


@patch("backend.agent.tools.tailor_resume_with_gemini")
def test_tailor_resume_delegates_and_preserves_result(mock_tailor: Mock) -> None:
    expected = ResumeTailoringRecommendations(
        top_changes=["Lead with Python experience"],
        skills_to_emphasize=["Python"],
        experiences_to_emphasize=["Production API work"],
        missing_keywords=["LLM evaluation"],
        bullet_rewrite_suggestions=["Built a production API using Python."],
        overall_advice=["Prioritize relevant experience."],
    )
    mock_tailor.return_value = expected

    result = tailor_resume(**PROFILE)

    assert result is expected
    mock_tailor.assert_called_once_with(**PROFILE)
