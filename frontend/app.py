import os

import requests
import streamlit as st

st.set_page_config(page_title="AI Career Copilot", page_icon="🧭")

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
ANALYSIS_MODES = {
    "AI analysis": ("/analyze-profile/llm", "Gemini AI analysis"),
    "Rule-based analysis": ("/analyze-profile", "Rule-based analysis"),
}
SCORE_LABELS = {
    "technical_skills": "Technical skills",
    "domain_experience": "Domain experience",
    "seniority": "Seniority",
    "role_specific_requirements": "Role-specific requirements",
    "education": "Education",
    "communication_leadership": "Communication / leadership",
}


def display_score_breakdown(analysis: dict) -> None:
    score_breakdown = analysis.get("score_breakdown")
    if not score_breakdown:
        return

    st.subheader("Score breakdown")
    for field, label in SCORE_LABELS.items():
        st.metric(label, f"{score_breakdown[field]}%")


st.title("AI Career Copilot")
st.write("See how your experience lines up with your next role.")
st.subheader("Analyze your resume")

with st.form("resume-analysis-form"):
    resume_file = st.file_uploader("Upload your resume", type=["pdf"])
    resume_job_description = st.text_area(
        "Target job description",
        placeholder="Paste the job's responsibilities and qualifications.",
    )
    resume_submitted = st.form_submit_button("Analyze Resume", type="primary")

if resume_submitted and resume_file is None:
    st.error("Please upload a PDF resume before starting the analysis.")
elif resume_submitted and not resume_job_description.strip():
    st.error("Please enter a job description before analyzing your resume.")
elif resume_submitted:
    try:
        with st.spinner("Analyzing your resume with AI..."):
            response = requests.post(
                f"{API_BASE_URL}/analyze-resume/llm",
                files={
                    "file": (
                        resume_file.name,
                        resume_file.getvalue(),
                        resume_file.type,
                    )
                },
                data={"job_description": resume_job_description},
                timeout=60,
            )
            response.raise_for_status()
            resume_analysis = response.json()
            if not (
                isinstance(resume_analysis, dict)
                and isinstance(resume_analysis.get("match_score"), int)
                and isinstance(resume_analysis.get("score_breakdown"), dict)
                and all(
                    isinstance(resume_analysis["score_breakdown"].get(field), int)
                    for field in SCORE_LABELS
                )
                and all(
                    isinstance(resume_analysis.get(field), list)
                    for field in ("strengths", "skill_gaps", "next_steps")
                )
            ):
                raise ValueError
    except requests.Timeout:
        st.error("The resume analysis took too long. Please try again.")
    except requests.ConnectionError:
        st.error("Could not connect to the backend. Make sure the FastAPI server is running.")
    except requests.RequestException:
        st.error("The resume analysis could not be completed. Please try again.")
    except ValueError:
        st.error("The backend returned an invalid response.")
    else:
        st.metric("Match score", f"{resume_analysis['match_score']}%")
        display_score_breakdown(resume_analysis)

        st.subheader("Strengths")
        for strength in resume_analysis["strengths"]:
            st.markdown(f"- {strength}")

        st.subheader("Skill gaps")
        if resume_analysis["skill_gaps"]:
            for gap in resume_analysis["skill_gaps"]:
                st.markdown(f"- {gap}")
        else:
            st.write("No gaps identified in this analysis.")

        st.subheader("Next steps")
        for step in resume_analysis["next_steps"]:
            st.markdown(f"- {step}")

st.divider()
with st.expander("No resume? Enter profile manually"):
    with st.form("profile-analysis-form"):
        analysis_mode = st.radio(
            "Analysis mode",
            options=list(ANALYSIS_MODES),
            index=0,
            horizontal=True,
        )
        current_background = st.text_area(
            "Current background",
            placeholder="For example: 3 years in operations and customer support",
        )
        target_role = st.text_input("Target role", placeholder="For example: Data Analyst")
        job_description = st.text_area(
            "Job description",
            placeholder="Paste the job's responsibilities and qualifications.",
        )
        skills_text = st.text_input(
            "Skills (comma-separated)",
            placeholder="Python, SQL, communication",
        )
        project_experience = st.text_area(
            "Project experience",
            placeholder="Describe a relevant project and what you contributed.",
        )
        submitted = st.form_submit_button("Analyze Profile")

    if submitted and not job_description.strip():
        st.error("Please enter a job description before analyzing your profile.")
    elif submitted:
        payload = {
            "current_background": current_background,
            "target_role": target_role,
            "job_description": job_description,
            "skills": [skill.strip() for skill in skills_text.split(",") if skill.strip()],
            "project_experience": project_experience,
        }
        endpoint, result_label = ANALYSIS_MODES[analysis_mode]

        try:
            with st.spinner(f"Running {result_label.lower()}..."):
                response = requests.post(
                    f"{API_BASE_URL}{endpoint}", json=payload, timeout=60
                )
                response.raise_for_status()
                analysis = response.json()
        except requests.Timeout:
            st.error("The AI analysis took too long. Please try again.")
        except requests.ConnectionError:
            st.error(
                "Could not connect to the backend. Make sure the FastAPI server is running."
            )
        except requests.RequestException:
            st.error("The profile analysis could not be completed. Please try again.")
        except ValueError:
            st.error("The backend returned an invalid response.")
        else:
            st.info(f"Result source: {result_label}")
            st.metric("Match score", f"{analysis['match_score']}%")
            display_score_breakdown(analysis)

            st.subheader("Strengths")
            for strength in analysis["strengths"]:
                st.markdown(f"- {strength}")

            st.subheader("Skill gaps")
            if analysis["skill_gaps"]:
                for gap in analysis["skill_gaps"]:
                    st.markdown(f"- {gap}")
            else:
                st.write("No gaps identified in this first-pass analysis.")

            st.subheader("Next steps")
            for step in analysis["next_steps"]:
                st.markdown(f"- {step}")
