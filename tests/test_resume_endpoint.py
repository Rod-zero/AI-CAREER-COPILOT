from io import BytesIO

from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

from backend.main import app

client = TestClient(app)


def create_pdf_bytes(text: str | None = None) -> bytes:
    pdf_buffer = BytesIO()
    pdf = canvas.Canvas(pdf_buffer)
    if text:
        pdf.drawString(72, 720, text)
    pdf.save()
    return pdf_buffer.getvalue()


def test_parse_resume_extracts_pdf_text() -> None:
    expected_text = "Hello AI Career Copilot"

    response = client.post(
        "/parse-resume",
        files={"file": ("resume.pdf", create_pdf_bytes(expected_text), "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json()["filename"] == "resume.pdf"
    assert expected_text in response.json()["text"]


def test_parse_resume_rejects_non_pdf() -> None:
    response = client.post(
        "/parse-resume",
        files={"file": ("resume.txt", b"Not a PDF", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Uploaded file must be a PDF."}


def test_parse_resume_rejects_pdf_without_extractable_text() -> None:
    response = client.post(
        "/parse-resume",
        files={"file": ("blank.pdf", create_pdf_bytes(), "application/pdf")},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "The PDF contains no extractable text."}
