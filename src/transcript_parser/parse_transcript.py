#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

# Lightweight deps; assume pdfplumber is available (declared in pyproject)
try:
    import pdfplumber  # type: ignore
except Exception:  # pragma: no cover - tests have this dep
    pdfplumber = None  # type: ignore


@dataclass
class ParseResult:
    courses: list[tuple[str, str]]
    text_source: str
    notes: dict


SUBJECT_ALIASES = {
    # core math family
    "math": ["MATH", "MAT", "MTH", "MA", "MATG"],
    "stat": ["STAT", "STA"],
    "cs": ["CS", "CSC", "CSCI", "CSE", "COSC"],
    "physics": ["PHYS", "PHY"],
    "chem": ["CHEM", "CHM"],
    "bio": ["BIOL", "BIO"],
    "econ": ["ECON", "ECN"],
    "engr": ["ENGR", "EGR"],
}


def expand_subjects(subjects: Iterable[str]) -> list[str]:
    expanded: list[str] = []
    for s in subjects:
        key = s.lower()
        if key in SUBJECT_ALIASES:
            expanded.extend(SUBJECT_ALIASES[key])
        else:
            expanded.append(s.upper())
    # unique-preserve order
    seen = set()
    uniq: list[str] = []
    for p in expanded:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


COURSE_PAT_CACHE: dict[tuple[str, ...], re.Pattern[str]] = {}


def _build_course_pattern(prefixes: list[str]) -> re.Pattern[str]:
    key = tuple(prefixes)
    if key in COURSE_PAT_CACHE:
        return COURSE_PAT_CACHE[key]
    # Accept PREFIX 123, PREFIX-123, PREFIX:123, optional letter suffix (e.g., 101A)
    # Word boundaries around prefix; allow multiple spaces
    pfx = r"(?:%s)" % "|".join(re.escape(p) for p in prefixes)
    pat = re.compile(rf"\b{pfx}\s*[-:\s]?\s*(\d{{3}}[A-Z]?)\b", re.IGNORECASE)
    COURSE_PAT_CACHE[key] = pat
    return pat


def _iter_lines(text: str) -> list[str]:
    # Normalize newlines and collapse CRLF
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    # Remove obviously empty lines
    return [ln for ln in lines if ln.strip()]


def extract_pdf_text(path: Path, max_pages: int | None = None) -> str:
    if pdfplumber is None:
        return ""
    text_parts: list[str] = []
    try:
        with pdfplumber.open(str(path)) as pdf:
            for i, page in enumerate(pdf.pages):
                if max_pages is not None and i >= max_pages:
                    break
                t = page.extract_text() or ""
                if t:
                    text_parts.append(t)
    except Exception:
        return ""
    return "\n".join(text_parts)


def find_courses_in_text(text: str, prefixes: list[str]) -> list[tuple[str, str]]:
    """
    Returns list of (code, snippet). Code is like 'MATH 101' using the
    matched prefix and number; snippet is a cleaned substring from the line
    containing the match.
    """
    pat = _build_course_pattern(prefixes)
    results: list[tuple[str, str]] = []
    lines: list[str] = _iter_lines(text)
    for ln in lines:
        # Work on an upper-cased copy for matching, but keep original for display
        upper_ln = ln.upper()
        matches = list(pat.finditer(upper_ln))
        if not matches:
            continue
        # If multiple matches in one line, emit one entry per match with the same snippet
        # but each has its own code (so test sees 'MATH 101' as separate item).
        # Clean snippet: trim excessive spaces
        cleaned = re.sub(r"\s+", " ", ln).strip()
        for m in matches:
            prefix = m.group(0)[: len(m.group(0)) - len(m.group(1))].strip()
            number = m.group(1).upper()
            # Normalize prefix in code to canonical uppercase of the matched prefix token
            # Extract the actual prefix token from the match span
            code_prefix = upper_ln[m.start() : m.start() + len(prefix)].strip()
            # Normalize to 'PREFIX 123'
            code = f"{code_prefix} {number}"
            results.append((code, cleaned))
    return results


def need_ocr(pdf_text: str, hits: int) -> bool:
    # Trigger OCR only if very short text OR no/little hits
    return (len(pdf_text) < 400) or (hits < 2)


def run_ocr_extract_text(path: Path) -> str:
    """
    Attempt OCR with PaddleOCR. Import lazily; if anything fails, return "".
    """
    try:
        from paddleocr import PaddleOCR  # type: ignore
        from pdf2image import convert_from_path  # type: ignore
    except Exception:
        return ""

    try:
        # Render pages to images (default dpi ~200 is OK for speed)
        images = convert_from_path(str(path))
        ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False)
        texts: list[str] = []
        for img in images:
            res = ocr.ocr(img, cls=True)
            # res is list[list[([x1,y1]..), (text, conf)]]
            lines: list[str] = []
            for page in res or []:
                for item in page or []:
                    txt = item[1][0] if item and item[1] else ""
                    if txt:
                        lines.append(txt)
            if lines:
                texts.append("\n".join(lines))
        return "\n".join(texts)
    except Exception:
        return ""


def parse_file(path: Path, subjects: list[str], prefer_ocr: bool = False) -> ParseResult:
    prefixes = expand_subjects(subjects)
    pdf_text = extract_pdf_text(path)
    courses = find_courses_in_text(pdf_text, prefixes)
    text_source = "pdf"

    if prefer_ocr or need_ocr(pdf_text, len(courses)):
        ocr_text = run_ocr_extract_text(path)
        if ocr_text:
            ocr_courses = find_courses_in_text(ocr_text, prefixes)
            # Prefer OCR only if it adds value
            if len(ocr_courses) > len(courses):
                courses = ocr_courses
                text_source = "ocr"

    return ParseResult(courses=courses, text_source=text_source, notes={})


def _dump_stdout(base: str, res: ParseResult, subjects: list[str]) -> None:
    print(f"Results for {base}")
    if res.courses:
        for code, snippet in res.courses:
            # Only print each course once per identical snippet
            print(f"  {code} — {snippet}")
    else:
        print(" [no course codes detected]")
    print(f"Parsed {base} (subjects: {', '.join(subjects)}; text_source={res.text_source})")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="PDF file to parse")
    ap.add_argument("--subjects", nargs="+", required=True, help="Subjects to extract (e.g., math stat)")
    ap.add_argument("--out", help="Optional JSON file to write results")
    ap.add_argument("--prefer-ocr", action="store_true", help="Force OCR even if PDF text is present")
    args = ap.parse_args(argv)

    path = Path(args.input)
    base = path.name

    res = parse_file(path, args.subjects, prefer_ocr=args.prefer_ocr)

    if args.out:
        payload = {
            "file": base,
            "subjects": args.subjects,
            "text_source": res.text_source,
            "courses": [{"code": c, "line": s} for c, s in res.courses],
            "notes": res.notes,
        }
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        _dump_stdout(base, res, args.subjects)


if __name__ == "__main__":
    main()
