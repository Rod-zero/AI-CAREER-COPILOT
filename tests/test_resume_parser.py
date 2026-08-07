from io import BytesIO

from reportlab.pdfgen import canvas

from backend.services.resume_parser import extract_text_from_pdf


def test_extract_text_from_pdf() -> None:
    expected_text = "Hello AI Career Copilot"
    pdf_buffer = BytesIO()
    pdf = canvas.Canvas(pdf_buffer)
    pdf.drawString(72, 720, expected_text)
    pdf.save()

    result = extract_text_from_pdf(pdf_buffer.getvalue())

    assert expected_text in result
