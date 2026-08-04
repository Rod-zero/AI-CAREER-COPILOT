# AI Career Copilot

Initial project skeleton with a FastAPI backend and a separate Streamlit frontend.

## Setup (Windows PowerShell)

From the repository root, activate the existing virtual environment and install the project:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
```

Optionally copy the example environment file:

```powershell
Copy-Item .env.example .env
```

## Run the backend

```powershell
python -m uvicorn backend.main:app --reload
```

The health endpoint is available at `http://localhost:8000/health`.

## Run the frontend

In a second PowerShell window with the virtual environment activated:

```powershell
python -m streamlit run frontend/app.py
```

## Run tests

```powershell
python -m pytest
```
