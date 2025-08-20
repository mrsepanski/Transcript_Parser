#!/usr/bin/env python3
"""
parse_transcript.py (updated)
- Keeps public API: parse_course_line(line) -> dict or None
- Robust course-title parsing (handles multi-word titles before credits)
- Robust student name/university extraction
"""

import argparse
import re
import sys
from typing import Dict, List, Optional, Tuple

# Optional deps (the Docker image usually has these)
try:
    import pdfplumber
except Exception:  # pragma: no cover
    pdfplumber = None  # type: ignore

try:
    from pdf2image import convert_from_path
except Exception:  # pragma: no cover
    convert_from_path = None  # type: ignore

try:
    from paddleocr import PaddleOCR  # type: ignore
except Exception:  # pragma: no cover
    PaddleOCR = None  # type: ignore


# -------------------------
# Parsing helpers
# -------------------------

COURSE_LINE_RE = re.compile(
    r"""
    ^\s*
    (?P<subject>[A-Z]{2,5})              # e.g., MATH, STAT, MA
    \s+
    (?P<number>\d{3}[A-Z]?)              # e.g., 570, 101A
    \s+
    (?P<title>.+?)                       # full title (lazy) up to credits
    \s+
    (?P<credits>\d+(?:\.\d{1,2})?)       # 4 or 4.00
    \s+
    (?P<grade>
        A(?:\+|-)?|B(?:\+|-)?|C(?:\+|-)?|D(?:\+|-)?|F|
        S|U|P|NP|W|AUD|EP|IP|X
    )
    (?:\s+\d+(?:\.\d{1,2})?)?            # optional trailing points col (e.g., 13.20)
    \s*$
    """,
    re.VERBOSE,
)


def parse_course_line(line: str) -> Optional[Dict[str, str]]:
    """
    Return {'subject','number','title','credits','grade'} or None.
    Kept under the original public name for tests/typecheck.
    """
    m = COURSE_LINE_RE.search(line)
    if not m:
        return None
    d = m.groupdict()
    # Normalize
    d["subject"] = d["subject"].strip()
    d["number"] = d["number"].strip()
    title = d["title"].strip(" -\t,;:")
    title = re.sub(r"\s{2,}", " ", title)
    d["title"] = title
    d["credits"] = d["credits"].strip()
    d["grade"] = d["grade"].strip()
    return d


def robust_extract_student_name(text: str) -> Optional[str]:
    # Preferred: "Record of: NAME  Page: 1"
    m = re.search(r'(?im)^\s*Record\s+of:\s*(.+?)\s+Page\s*:\s*\d+', text)
    if m:
        name = re.sub(r'\s{2,}', ' ', m.group(1).strip())
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
    # ALL-CAPS header with UNIVERSITY
    caps = [
        ln.strip()
        for ln in head
        if 'UNIVERSITY' in ln.upper()
        and ln.strip()
        and ln.strip() == ln.strip().upper()
        and len(ln.strip()) <= 120
    ]
    if caps:
        uni = re.sub(r'\bAcademic\s+Record\b', '', caps[0], flags=re.I).strip(' -\t')
        if uni:
            return uni
    # "Transcript from X University"
    m = re.search(r'(?i)Transcript\s+from\s+([A-Za-z][A-Za-z &\-\.\']+?University)\b', text)
    if m:
        return m.group(1).strip()
    # Any standalone "X University" line
    m = re.search(r'(?im)^\s*([A-Za-z][A-Za-z &\-\.\']+?University)\s*$', text)
    if m:
        return m.group(1).strip()
    return None


# -------------------------
# Extraction pipeline
# -------------------------

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
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts), pages


def run_ocr_with_paddle(path: str, dpi: int = 300, lang: str = "en") -> str:
    if PaddleOCR is None or convert_from_path is None:
        return ""
    try:
        images = convert_from_path(path, dpi=dpi)
    except Exception:
        return ""
    ocr = PaddleOCR(lang=lang, use_angle_cls=True, show_log=False)
    out: List[str] = []
    for img in images:
        res = ocr.ocr(img, cls=True)
        if not res:
            continue
        for block in res:
            for line in block:
                txt = line[1][0]
                if txt:
                    out.append(txt)
    return "\n".join(out)


def should_trigger_ocr(pdf_text: str, courses: List[Dict[str, str]], student: Optional[str], university: Optional[str]) -> bool:
    if len(pdf_text) < 400:
        return True
    if (not student) or (not university):
        return True
    if len(courses) < 2:
        return True
    return False


def parse_courses_from_text(text: str, subjects: List[str]) -> List[Dict[str, str]]:
    subj_set = set(s.upper() for s in subjects)
    out: List[Dict[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        c = parse_course_line(line)
        if c and (not subj_set or c["subject"].upper() in subj_set):
            out.append(c)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="PDF file(s) to parse")
    ap.add_argument("--subjects", nargs="+", required=True)
    ap.add_argument("--prefer-ocr", action="store_true")
    ap.add_argument("--ocr-dpi", type=int, default=300)
    ap.add_argument("--ocr-lang", default="en")
    ap.add_argument("--max-pages", type=int, default=None)
    args = ap.parse_args()

    # Normalize/expand subjects
    expanded: List[str] = []
    for s in args.subjects:
        s = s.strip().upper()
        if s in {"MATH", "MAT", "MA", "MTH"}:
            expanded.extend(["MATH", "MAT", "MA", "MTH"])
        elif s in {"STAT", "STA"}:
            expanded.extend(["STAT", "STA"])
        else:
            expanded.append(s)
    # Dedup keep order
    seen, subjects = set(), []
    for x in expanded:
        if x not in seen:
            subjects.append(x); seen.add(x)

    for path in args.inputs:
        source = "pdf"
        paddle_len = 0
        pdf_text = ""
        pdf_pages = 0

        if not args.prefer_ocr:
            pdf_text, pdf_pages = extract_pdf_text(path, args.max_pages)

        courses = parse_courses_from_text(pdf_text, subjects) if pdf_text else []
        student = robust_extract_student_name(pdf_text) if pdf_text else None
        university = robust_extract_university(pdf_text) if pdf_text else None

        if args.prefer_ocr or should_trigger_ocr(pdf_text, courses, student, university):
            ocr_text = run_ocr_with_paddle(path, dpi=args.ocr_dpi, lang=args.ocr_lang)
            paddle_len = len(ocr_text)
            if ocr_text:
                source = "ocr"
                courses = parse_courses_from_text(ocr_text, subjects)
                student = robust_extract_student_name(ocr_text) or student
                university = robust_extract_university(ocr_text) or university

        # Prefer PDF when it looks good (no forced OCR)
        if not args.prefer_ocr and len(pdf_text) >= 400 and courses and source == "ocr":
            source = "pdf"

        # Output
        print(f"Results for {path}:")
        print(f"  Student Name: {student or 'N/A'}")
        print(f"  University:   {university or 'N/A'}")
        if courses:
            subj_label = ", ".join(sorted({c['subject'] for c in courses}))
            print(f"  Courses ({subj_label}):")
            for c in courses:
                title = c['title'] or "N/A"
                print(f"    {c['subject']} {c['number']} {title}, grade: {c['grade']}")
        else:
            print("  Courses ():")

        print("  Notes: "
              f"text_source={source}; "
              f"pdf_pages={pdf_pages}; "
              f"pdf_text_len={len(pdf_text)}; "
              f"paddle_len={paddle_len}; "
              f"paddleocr_ok={'True' if PaddleOCR else 'False'}; "
              f"pdf2image_ok={'True' if convert_from_path else 'False'}; "
              f"pdftoppm={'/usr/bin/pdftoppm' if convert_from_path else 'N/A'}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
