# SPDX-License-Identifier: MIT
"""
Transcript parser CLI.

Strategy:
- Prefer extracting searchable text via pdfplumber.
- Only fall back to OCR (PaddleOCR) if the PDF text is too short
  or yields too few course-code hits, or if --prefer-ocr is set.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# ----------------------------- subject aliases ------------------------------

SUBJECT_ALIASES: dict[str, Sequence[str]] = {
    "math": ("MATH", "MAT", "MTH", "MA", "MATG"),
    "stat": ("STAT", "STA"),
    "cs": ("CS", "CSC", "CSCI", "CSE", "COSC"),
    "physics": ("PHYS", "PHY"),
    "chem": ("CHEM", "CHM"),
    "bio": ("BIOL", "BIO"),
    "econ": ("ECON", "ECN"),
    "engr": ("ENGR", "EGR"),
}


def expand_subjects(subjects: Sequence[str]) -> list[str]:
    out: list[str] = []
    for s in subjects:
        key = s.strip().lower()
        if key in SUBJECT_ALIASES:
            out.extend(SUBJECT_ALIASES[key])
        else:
            out.append(s.upper())
    # dedupe, preserve order
    seen: set[str] = set()
    uniq: list[str] = []
    for p in out:
        u = p.upper()
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq


# ------------------------------- utilities ----------------------------------


@dataclass
class ParseResult:
    file: str
    text_source: str  # 'pdf' or 'ocr'
    courses: list[str]
    notes: dict[str, str]


def _read_pdf_text(path: Path, max_pages: int | None = None) -> str:
    import pdfplumber  # type: ignore[import-not-found]

    text_parts: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        pages = pdf.pages
        if max_pages is not None:
            pages = pages[:max_pages]
        for page in pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)


def _ocr_pdf_text(path: Path, dpi: int = 300, lang: str = "en") -> str:
    # Deferred imports so we only pay the cost if/when needed
    from paddleocr import PaddleOCR  # type: ignore[import-not-found]
    from pdf2image import convert_from_path  # type: ignore[import-not-found]

    images = convert_from_path(str(path), dpi=dpi)
    ocr = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=False)
    lines: list[str] = []
    for img in images:
        # Convert PIL image to temp file-less ndarray format expected by PaddleOCR
        # PaddleOCR.ocr accepts ndarray directly.
        import numpy as np  # type: ignore[import-not-found]

        arr = np.array(img)
        result = ocr.ocr(arr, cls=True)
        for page in result:
            if page is None:
                continue
            for box in page:
                txt = box[1][0] if isinstance(box, (list, tuple)) and len(box) >= 2 else ""
                if txt:
                    lines.append(txt)
    return "\n".join(lines)


def _compile_course_regex(prefixes: Sequence[str]) -> re.Pattern[str]:
    # Accept PREFIX 123, PREFIX-123, PREFIX:123 with optional spaces
    # Word boundary on both ends of prefix
    prefix_alt = "|".join(re.escape(p) for p in prefixes)
    pattern = rf"\b(?:{prefix_alt})\s*[-:]?\s*(\d{{3,4}}[A-Za-z]?)\b"
    return re.compile(pattern, flags=re.IGNORECASE)


def _extract_courses_from_text(text: str, prefixes: Sequence[str]) -> list[str]:
    """Return pretty, per-course snippets from the text."""
    if not text.strip():
        return []

    pat = _compile_course_regex(prefixes)
    # Build a regex to find the *next* course start so we can truncate the snippet
    any_prefix_re = re.compile(
        rf"\b(?:{'|'.join(re.escape(p) for p in prefixes)})\s*[-:]?\s*\d{{3,4}}",
        flags=re.IGNORECASE,
    )

    lines: list[str] = text.splitlines()  # type: ignore[assignment]
    courses: list[str] = []

    for line in lines:
        for m in pat.finditer(line):
            # Base "SUBJ NUM"
            head = line[m.start() : m.end()]
            tail = line[m.end() :]

            # Drop a solitary single-letter column marker right after the code (e.g., " C ")
            tail = re.sub(r"^\s+[A-Z]\s+(?=\S)", " ", tail)

            # Truncate tail at the next course occurrence on the same line
            nxt = any_prefix_re.search(tail)
            if nxt:
                tail = tail[: nxt.start()]

            snippet = f"{head.upper()} — {tail.strip()}".rstrip(" -–—")
            snippet = re.sub(r"\s{2,}", " ", snippet)  # collapse excess spaces
            if snippet and snippet not in courses:
                courses.append(snippet)

    return courses


def parse_file(path: Path, subjects: Sequence[str], prefer_ocr: bool = False) -> ParseResult:
    prefixes = expand_subjects(subjects)

    text_source = "pdf"
    notes: dict[str, str] = {}
    text = ""
    try:
        text = _read_pdf_text(path)
    except Exception as e:  # pragma: no cover - safety
        notes["pdf_error"] = type(e).__name__
        text = ""

    courses = _extract_courses_from_text(text, prefixes) if text else []

    # Heuristics to decide OCR necessity
    need_ocr = prefer_ocr or (len(text) < 400 or len(courses) < 2)

    if need_ocr:
        try:
            ocr_text = _ocr_pdf_text(path)
            ocr_courses = _extract_courses_from_text(ocr_text, prefixes)
            # If OCR yields more, switch to it
            if len(ocr_courses) > len(courses):
                courses = ocr_courses
                text_source = "ocr"
            else:
                notes["ocr_unused"] = "pdf_better"
        except Exception as e:  # pragma: no cover - robustness
            notes["ocr_error"] = f"{type(e).__name__}: {e}"

    return ParseResult(file=path.name, text_source=text_source, courses=courses, notes=notes)


# ---------------------------------- CLI -------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Parse transcripts for course codes.")
    p.add_argument("inputs", nargs="+", help="PDF file(s) to parse")
    p.add_argument("--subjects", nargs="+", required=True, help="Subject areas (e.g., math stat cs)")
    p.add_argument("--out", help="Optional JSON output path")
    p.add_argument("--prefer-ocr", action="store_true", help="Force OCR even if PDF text looks OK")
    return p


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # Process the first input only for CLI output parity with prior behavior
    in_path = Path(args.inputs[0])
    result = parse_file(in_path, subjects=args.subjects, prefer_ocr=args.prefer_ocr)

    print(f"Results for {result.file}")
    if result.courses:
        for c in result.courses:
            print(f"  {c}")
    else:
        print(" [no course codes detected]")

    base = Path(result.file).name
    subj_str = ", ".join(s.lower() for s in args.subjects)
    print(f"Parsed {base} (subjects: {subj_str}; text_source={result.text_source})")

    if args.out:
        payload = {
            "file": result.file,
            "text_source": result.text_source,
            "courses": result.courses,
            "notes": result.notes,
        }
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
