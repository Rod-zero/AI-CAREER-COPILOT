from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_partial_skill_match_score() -> None:
    response = client.post(
        "/analyze-profile",
        json={
            "current_background": "Operations analyst",
            "target_role": "Data Analyst",
            "skills": ["Python", "SQL", "Communication"],
            "project_experience": "Built a reporting dashboard that tracked weekly operational metrics.",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "match_score": 60,
        "strengths": [
            "Demonstrated skill: Python",
            "Demonstrated skill: Sql",
            "Hands-on project experience",
        ],
        "skill_gaps": ["Machine Learning", "Statistics"],
        "next_steps": [
            "Build a small project using Machine Learning.",
            "Build a small project using Statistics.",
            "Tailor your resume to highlight experience relevant to Data Analyst.",
        ],
    }


def test_full_skill_match_score() -> None:
    response = client.post(
        "/analyze-profile",
        json={
            "current_background": "Data analyst",
            "target_role": "Data Scientist",
            "skills": ["Python", "SQL", "Statistics", "Machine Learning", "Data Analysis"],
            "project_experience": (
                "Built and evaluated a forecasting model using production customer data."
            ),
        },
    )

    assert response.status_code == 200
    assert response.json()["match_score"] == 100


def test_empty_background_does_not_add_background_points() -> None:
    response = client.post(
        "/analyze-profile",
        json={
            "current_background": "",
            "target_role": "Data Analyst",
            "skills": ["Python", "SQL", "Statistics", "Machine Learning"],
            "project_experience": (
                "Built and evaluated a reporting pipeline using operational data."
            ),
        },
    )

    assert response.status_code == 200
    assert response.json()["match_score"] == 90


def test_short_project_experience_does_not_add_project_points() -> None:
    response = client.post(
        "/analyze-profile",
        json={
            "current_background": "Data analyst",
            "target_role": "Data Analyst",
            "skills": ["Python", "SQL", "Statistics", "Machine Learning"],
            "project_experience": "Built a dashboard.",
        },
    )

    assert response.status_code == 200
    assert response.json()["match_score"] == 90


def test_agentic_ai_engineer_uses_specific_skill_mapping() -> None:
    response = client.post(
        "/analyze-profile",
        json={
            "current_background": "Backend engineer",
            "target_role": "Agentic AI Engineer",
            "skills": ["Python", "LLM"],
            "project_experience": "Built a prototype agent.",
        },
    )

    assert response.status_code == 200
    assert response.json()["skill_gaps"] == [
        "Agent Orchestration",
        "Apis",
        "Git",
        "Langchain",
        "Llm Evaluation",
        "Machine Learning",
        "Rag",
        "Testing",
    ]


def test_data_scientist_uses_specific_skill_mapping() -> None:
    response = client.post(
        "/analyze-profile",
        json={
            "current_background": "Data analyst",
            "target_role": "Data Scientist",
            "skills": ["Python", "SQL"],
            "project_experience": "Analyzed customer retention trends.",
        },
    )

    assert response.status_code == 200
    assert response.json()["skill_gaps"] == [
        "Data Analysis",
        "Machine Learning",
        "Statistics",
    ]


def test_analyze_profile_validates_required_fields() -> None:
    response = client.post("/analyze-profile", json={"target_role": "Data Analyst"})

    assert response.status_code == 422
