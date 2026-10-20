"""Build task2/adversarial_cv.pdf (Task 2 attack).

Idea: what a HUMAN sees and what a PDF TEXT EXTRACTOR reads are two different
layers of the same PDF.

* Visible layer: a picture (PNG) of an embellished CV. Pixels in an image are
  not text, so markitdown / pdfminer extract nothing from it.
* Text layer: Kelly Tsang's true CV, written with PDF text render mode 3
  ("invisible", the same mode OCR software uses for scanned documents).
  A human cannot see it, but every text extractor reads it.

Every verifier that reads the CV through `cv_text()` therefore only sees the
true claims, which match her LinkedIn/Facebook profiles, while the document a
person reads contains the false details below.

False / embellished details (visible to humans only):
  1. Degree upgraded: "MSc in Finance" (LinkedIn: BSc in Finance).
  2. Job start shifted: "Manager, AIA 2021 - Present" (LinkedIn: 2022).
  3. Title inflated: "Senior Manager, AIA" (LinkedIn: Manager).

Run:  pip install reportlab   (needs `pdftoppm` from poppler-utils)
      python3 task2/make_adversarial_cv.py
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

HERE = Path(__file__).resolve().parent
OUT = HERE / "adversarial_cv.pdf"


def cv_lines(fake: bool) -> list[tuple[str, str, float, float, float]]:
    """(text, font, size, x, y) for every line, in the true CV's layout."""
    title = "Senior Manager, AIA" if fake else "Manager, AIA"
    years = "2021 – Present" if fake else "2022 – Present"
    degree = "MSc in Finance" if fake else "BSc in Finance"
    bold, reg = "Helvetica-Bold", "Helvetica"
    return [
        ("Kelly Tsang", bold, 20, 63, 785),
        ("Finance Professional", reg, 11, 63, 767),
        ("Central, Hong Kong | Hometown: Sha Tin", reg, 10, 63, 752),
        ("Professional Experience", bold, 13, 63, 721),
        (f"{title}    {years}", bold, 10, 63, 700),
        ("- Led a team delivering finance projects on schedule.", reg, 10, 75, 686),
        ("- Managed stakeholders, budgets, and priorities.", reg, 10, 75, 672),
        ("Analyst, Standard Chartered    2019 – 2022", bold, 10, 63, 654),
        ("- Analysed finance data to support business decisions.", reg, 10, 75, 640),
        ("- Prepared reports and dashboards for stakeholders.", reg, 10, 75, 626),
        ("Analyst, HSBC    2016 – 2019", bold, 10, 63, 608),
        ("- Analysed finance data to support business decisions.", reg, 10, 75, 594),
        ("- Prepared reports and dashboards for stakeholders.", reg, 10, 75, 580),
        ("Education", bold, 13, 63, 555),
        (f"{degree}    Graduated 2016", bold, 10, 63, 534),
        ("HKUST", reg, 10, 63, 520),
        ("Skills", bold, 13, 63, 495),
        ("Financial Analysis, Excel, SQL, Risk Modeling", reg, 10, 63, 480),
    ]


def draw_lines(c: canvas.Canvas, lines, invisible: bool) -> None:
    for text, font, size, x, y in lines:
        t = c.beginText(x, y)
        t.setFont(font, size)
        if invisible:
            t.setTextRenderMode(3)  # invisible text: extracted, never painted
        t.textLine(text)
        c.drawText(t)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # 1. a normal PDF of the FAKE CV, then rasterise it to a PNG
        fake_pdf = tmp / "fake.pdf"
        c = canvas.Canvas(str(fake_pdf), pagesize=A4)
        draw_lines(c, cv_lines(fake=True), invisible=False)
        c.save()
        subprocess.run(["pdftoppm", "-r", "300", "-png", "-singlefile", str(fake_pdf), str(tmp / "fake")], check=True)

        # 2. final PDF = picture of the fake CV + invisible text of the true CV
        c = canvas.Canvas(str(OUT), pagesize=A4)
        c.setTitle("Kelly Tsang - CV")
        c.setAuthor("Kelly Tsang")
        width, height = A4
        c.drawImage(ImageReader(str(tmp / "fake.png")), 0, 0, width=width, height=height)
        draw_lines(c, cv_lines(fake=False), invisible=True)
        c.save()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
