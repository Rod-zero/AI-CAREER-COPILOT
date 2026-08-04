import os

import streamlit as st

st.set_page_config(page_title="AI Career Copilot", page_icon="🧭")

st.title("AI Career Copilot")
st.write("The project skeleton is ready.")
st.caption(f"Backend API: {os.getenv('API_BASE_URL', 'http://localhost:8000')}")
