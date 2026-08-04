import os

import requests
import streamlit as st

st.set_page_config(page_title="AI Career Copilot", page_icon="🧭")

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")

st.title("AI Career Copilot")
st.write("See how your experience lines up with your next role.")

with st.form("profile-analysis-form"):
    current_background = st.text_area(
        "Current background",
        placeholder="For example: 3 years in operations and customer support",
    )
    target_role = st.text_input("Target role", placeholder="For example: Data Analyst")
    skills_text = st.text_input(
        "Skills (comma-separated)",
        placeholder="Python, SQL, communication",
    )
    project_experience = st.text_area(
        "Project experience",
        placeholder="Describe a relevant project and what you contributed.",
    )
    submitted = st.form_submit_button("Analyze Profile", type="primary")

if submitted:
    payload = {
        "current_background": current_background,
        "target_role": target_role,
        "skills": [skill.strip() for skill in skills_text.split(",") if skill.strip()],
        "project_experience": project_experience,
    }

    try:
        response = requests.post(f"{API_BASE_URL}/analyze-profile", json=payload, timeout=10)
        response.raise_for_status()
        analysis = response.json()
    except requests.ConnectionError:
        st.error("Could not connect to the backend. Make sure the FastAPI server is running.")
    except requests.RequestException as exc:
        st.error(f"The backend could not analyze the profile: {exc}")
    except ValueError:
        st.error("The backend returned an invalid response.")
    else:
        st.metric("Match score", f"{analysis['match_score']}%")

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

st.caption(f"Backend API: {API_BASE_URL}")
