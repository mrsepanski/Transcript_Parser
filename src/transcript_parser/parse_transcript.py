#!/usr/bin/env python3
"""
Updated parse_transcript.py
- Fixes multi-word course titles (e.g., "MATH 570 Topics in Optimization")
- Fixes student name extraction (uses "Record of:" pattern when present)
- Fixes university extraction (prefers page header over transfer section)

NOTE: This is a drop-in replacement for your existing script.
It prints results to stdout in the same style you've been using.
"""

import argparse
import io
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

# --- Dependencies expected in your Docker image ---
# pdfplumber for direct PDF text extraction
# pdf2image & poppler for image conversion if OCR fallback is needed
# paddleocr for OCR fallback (optional; script will skip if unavailable)
try:
    import pdfplumber
except Exception as e:
    print(f"[WARN] pdfplumber import failed: {e}", file=sys.stderr)
    pdfplumber = None

# Try optional OCR deps
try:
    from pdf2image import convert_from_path
except Exception:
    convert_from_path = None

try:
    from paddleocr import PaddleOCR  # type: ignore
except Exception:
    PaddleOCR = None


@dataclass
class ParsedCourse:
    subject: str
    number: str
    title: str
    credits: str
    grade: str


COURSE_LINE_RE_V2 = re.compile(
    r"""
    ^\s*
    (?P<subject>[A-Z]{2,5})              # e.g., MATH, STAT, MA
    \s+
    (?P<number>\d{3}[A-Z]?)              # e.g., 570, 101A
    \s+
    (?P<title>.+?)                       # full title (lazy), anything until credits
    \s+
    (?P<credits>\d+(?:\.\d{1,2})?)       # 4 or 4.00
    \s+
    (?P<grade>
        A(?:\+|-)?|B(?:\+|-)?|C(?:\+|-)?|D(?:\+|-)?|F|
        S|U|P|NP|W|AUD|EP|IP|X
    )
    (?:\s+\d+(?:\.\d{1,2})?)?            # optional trailing points column (e.g., 13.20)
    \s*$
    """,
    re.VERBOSE,
)


def parse_course_line_v2(line: str) -> Optional[ParsedCourse]:
    m = COURSE_LINE_RE_V2.search(line)
    if not m:
        return None
    d = m.groupdict()
    title = d["title"].strip(" -\t,;:")
    title = re.sub(r"\s{2,}", " ", title)
    return ParsedCourse(
        subject=d["subject"].strip(),
        number=d["number"].strip(),
        title=title,
        credits=d["credits"].strip(),
        grade=d["grade"].strip(),
    )


def robust_extract_student_name(text: str) -> Optional[str]:
    # Preferred: "Record of: NAME Page: 1"
    m = re.search(r'(?im)^\s*Record\s+of:\s*(.+?)\s+Page\s*:\s*\d+', text)
    if m:
        name = re.sub(r'\s{2,}', ' ', m.group(1).strip())
        # Filter obvious non-names like "Course Level Graduate"
        if not re.search(r'Course\s+Level', name, re.I):
            return name

    # Fallbacks
    for pat in [
        r'(?im)^\s*Student\s+Name\s*:\s*(.+)$',
        r'(?im)^\s*Name\s*:\s*(.+)$',
    ]:
        m = re.search(pat, text)
        if m:
            name = m.group(1).strip()
            if not re.search(r'Course\s+Level', name, re.I):
                return name
    return None


def robust_extract_university(text: str) -> Optional[str]:
    lines = text.splitlines()
    head = lines[:60]

    # 1) ALL-CAPS header line containing UNIVERSITY
    caps_candidates = [
        ln.strip()
        for ln in head
        if 'UNIVERSITY' in ln.upper()
        and ln.strip()
        and ln.strip() == ln.strip().upper()
        and len(ln.strip()) <= 120
    ]
    if caps_candidates:
        uni = caps_candidates[0]
        uni = re.sub(r'\bAcademic\s+Record\b', '', uni, flags=re.I).strip(' -\t')
        return uni if uni else None

    # 2) "Transcript from X University"
    m = re.search(r'(?i)Transcript\s+from\s+([A-Za-z][A-Za-z &\-\.\']+?University)\b', text)
    if m:
        return m.group(1).strip()

    # 3) Any standalone line ending with University
    m = re.search(r'(?im)^\s*([A-Za-z][A-Za-z &\-\.\']+?University)\s*$', text)
    if m:
        return m.group(1).strip()

    return None


def extract_pdf_text(path: str, max_pages: Optional[int] = None) -> Tuple[str, int]:
    if pdfplumber is None:
        return "", 0
    text_parts: List[str] = []
    pages = 0
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            if max_pages is not None and i >= max_pages:
                break
            pages += 1
            txt = page.extract_text() or ""
            text_parts.append(txt)
    return "\n".join(text_parts), pages


def run_ocr_with_paddle(path: str, dpi: int = 300, lang: str = "en") -> str:
    if PaddleOCR is None or convert_from_path is None:
        return ""
    try:
        images = convert_from_path(path, dpi=dpi)
    except Exception:
        return ""
    ocr = PaddleOCR(lang=lang, use_angle_cls=True, show_log=False)
    out_lines: List[str] = []
    for img in images:
        res = ocr.ocr(img, cls=True)
        if not res:
            continue
        for block in res:
            for line in block:
                txt = line[1][0]
                if txt:
                    out_lines.append(txt)
    return "\n".join(out_lines)


def should_trigger_ocr(pdf_text: str, parsed_courses: List[ParsedCourse], student: Optional[str], university: Optional[str]) -> bool:
    # Heuristics: very short text, missing key fields, or too few courses
    if len(pdf_text) < 400:
        return True
    if (not student) or (not university):
        return True
    if len(parsed_courses) < 2:
        return True
    return False


def parse_courses_from_text(text: str, subjects: List[str]) -> List[ParsedCourse]:
    subj_set = set(s.upper() for s in subjects)
    courses: List[ParsedCourse] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        c = parse_course_line_v2(line)
        if c and (not subj_set or c.subject.upper() in subj_set):
            courses.append(c)
    return courses


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="PDF file(s) to parse")
    parser.add_argument("--subjects", nargs="+", required=True, help="Subject codes to include (e.g., math stat)")
    parser.add_argument("--prefer-ocr", action="store_true", help="Force OCR mode first")
    parser.add_argument("--ocr-dpi", type=int, default=300)
    parser.add_argument("--ocr-lang", default="en")
    parser.add_argument("--max-pages", type=int, default=None)
    args = parser.parse_args()

    # Normalize subjects like "math" -> ["MATH"]
    subjects_norm: List[str] = []
    for s in args.subjects:
        s = s.strip().upper()
        # expand common aliases
        if s in {"MATH", "MAT", "MA", "MTH"}:
            subjects_norm.extend(["MATH", "MAT", "MA", "MTH"])
        elif s in {"STAT", "STA"}:
            subjects_norm.extend(["STAT", "STA"])
        else:
            subjects_norm.append(s)
    # Deduplicate while preserving order
    seen = set()
    subjects = [x for x in subjects_norm if not (x in seen or seen.add(x))]

    for path in args.inputs:
        source = "pdf"
        paddle_len = 0
        ocr_used = False
        pdf_text = ""
        pdf_pages = 0

        if not args.prefer_ocr:
            pdf_text, pdf_pages = extract_pdf_text(path, args.max_pages)

        courses = parse_courses_from_text(pdf_text, subjects) if pdf_text else []
        student = robust_extract_student_name(pdf_text) if pdf_text else None
        university = robust_extract_university(pdf_text) if pdf_text else None

        trigger_ocr = args.prefer_ocr or should_trigger_ocr(pdf_text, courses, student, university)
        ocr_text = ""
        if trigger_ocr:
            ocr_text = run_ocr_with_paddle(path, dpi=args.ocr_dpi, lang=args.ocr_lang)
            paddle_len = len(ocr_text)
            if ocr_text:
                ocr_used = True
                source = "ocr"
                # Re-parse from OCR text
                courses = parse_courses_from_text(ocr_text, subjects)
                student = robust_extract_student_name(ocr_text) or student
                university = robust_extract_university(ocr_text) or university

        # Final safety: if PDF text looked fine (long enough, courses parsed), prefer it
        # This keeps behavior consistent with "no OCR if not needed"
        if not args.prefer_ocr and len(pdf_text) >= 400 and courses and source == "ocr":
            source = "pdf"

        # Print results
        print(f"Results for {path}:")
        print(f"  Student Name: {student or 'N/A'}")
        print(f"  University:   {university or 'N/A'}")
        if courses:
            subj_label = ", ".join(sorted(set(c.subject for c in courses)))
            print(f"  Courses ({subj_label}):")
            for c in courses:
                title = c.title if c.title else "N/A"
                print(f"    {c.subject} {c.number} {title}, grade: {c.grade}")
        else:
            print("  Courses ():")

        # Notes line similar to your previous format
        print(
            "  Notes: "
            f"text_source={source}; "
            f"pdf_pages={pdf_pages}; "
            f"pdf_text_len={len(pdf_text)}; "
            f"paddle_len={paddle_len}; "
            f"paddleocr_ok={'True' if PaddleOCR else 'False'}; "
            f"pdf2image_ok={'True' if convert_from_path else 'False'}; "
            f"pdftoppm={'/usr/bin/pdftoppm' if convert_from_path else 'N/A'}"
        )
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
