from fastapi import FastAPI

app = FastAPI(title="AI Career Copilot API")


@app.get("/health")
def health() -> dict[str, str]:
    """Return the API health status."""
    return {"status": "ok"}
