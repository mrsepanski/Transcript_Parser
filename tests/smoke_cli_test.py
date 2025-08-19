from pathlib import Path
from tempfile import TemporaryDirectory

from reportlab.pdfgen import canvas  # type: ignore


def _make_pdf(path: Path) -> None:
    c = canvas.Canvas(str(path))
    c.setFont("Helvetica", 12)
    c.drawString(72, 720, "Name: Test Student")
    c.drawString(72, 690, "MATH 101 Calculus I A")
    c.save()


def test_cli_smoke(capsys):
    from transcript_parser.parse_transcript import main

    with TemporaryDirectory() as td:
        pdf = Path(td) / "sample.pdf"
        _make_pdf(pdf)
        import sys

        sys.argv = ["transcript-parser", str(pdf), "--subjects", "math"]
        main()
        out = capsys.readouterr().out
        assert "Results for" in out
        assert "MATH 101" in out
