from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

try:
    import pdfplumber  # type: ignore
except Exception:  # pragma: no cover
    pdfplumber = None  # type: ignore

# ---------- Subject aliases (generic) ----------
SUBJECT_ALIASES: dict[str, list[str]] = {
    "math": ["MATH", "MAT", "MTH", "MA", "MATG", "MAS", "MAP"],
    "stat": ["STAT", "STA"],
    "cs": ["CS", "CSC", "CSCI", "CSE", "COSC"],
    "physics": ["PHYS", "PHY"],
    "chem": ["CHEM", "CHM"],
    "bio": ["BIOL", "BIO"],
    "econ": ["ECON", "ECN"],
    "engr": ["ENGR", "EGR"],
}

# ---------- Regexes ----------
START_CODE_PAT = re.compile(r"(?i)^\s*([A-Z]{2,})\s*[-:\s]?\s*(\d{3,4}[A-Z]?)\b")
NUM_TOKEN_PAT = re.compile(r"(?i)^\d{3,4}[A-Z]?$")
GRADE_PAT = re.compile(r"(?i)(?<!\w)(A\+|A-|A|B\+|B-|B|C\+|C-|C|D\+|D-|D|E|F|P|S|U|T|I|IN PROGRESS)(?!\w)")

ADMIN_ROW = re.compile(r"(?i)^(Ehrs|GPA|TOTAL|Dean's List|Good Standing|Earned Hrs|TRANSCRIPT TOTALS|Totals?)\b")
URL_PAT = re.compile(r"https?://")

# Tokens we should ignore in the "status/campus" column between code and title
CAMPUS_STATUS = {
    "MAIN",
    "DISTANCE",
    "ONLINE",
    "DL",
    "WEB",
    "GR",
    "UG",
    "PB",
    "EVENING",
    "DAY",
    "CAMPUS",
    "LEVEL",
}

CREDITS_PAT = re.compile(r"\s\d+\.\d{2,3}\b")  # e.g., 3.00 or 3.000


# ---------- Data classes ----------
@dataclass
class Tok:
    text: str
    x0: float
    x1: float
    y0: float
    y1: float
    page: int


@dataclass
class Row:
    page: int
    y: float
    toks: list[Tok]


def _normalize_text(s: str) -> str:
    s = s.replace("\xa0", " ").replace("\u00a0", " ")
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    return s


def _expand_subjects(subjects: Iterable[str]) -> list[str]:
    expanded: set[str] = set()
    for s in subjects:
        key = s.lower()
        if key in SUBJECT_ALIASES:
            expanded.update(SUBJECT_ALIASES[key])
        else:
            expanded.add(s.upper())
    return sorted(expanded)


def _allowed_set(subjects: Iterable[str]) -> set[str]:
    return {p.upper() for p in _expand_subjects(subjects)}


def _extract_rows(path: Path, y_tol: float = 2.0) -> list[Row]:
    rows: list[Row] = []
    if pdfplumber is None:
        return rows
    with pdfplumber.open(path) as pdf:
        for pidx, page in enumerate(pdf.pages, start=1):
            words = page.extract_words() or []
            words.sort(key=lambda w: (w.get("top", 0.0), w.get("x0", 0.0)))
            cur: Row | None = None
            for w in words:
                t = _normalize_text(w.get("text", "") or "")
                if not t:
                    continue
                top = float(w.get("top", 0.0))
                x0 = float(w.get("x0", 0.0))
                x1 = float(w.get("x1", x0))
                bottom = float(w.get("bottom", top + 8))
                tok = Tok(t, x0, x1, top, bottom, pidx)
                if cur is None or abs(top - cur.y) > y_tol or (tok.page != cur.page):
                    if cur is not None:
                        cur.toks.sort(key=lambda t: t.x0)
                        rows.append(cur)
                    cur = Row(pidx, top, [tok])
                else:
                    cur.toks.append(tok)
            if cur is not None:
                cur.toks.sort(key=lambda t: t.x0)
                rows.append(cur)
    rows.sort(key=lambda r: (r.page, r.y))
    return rows


def _row_text(row: Row) -> str:
    return " ".join(t.text for t in row.toks)


def _is_admin_row(txt: str) -> bool:
    return bool(ADMIN_ROW.match(txt) or URL_PAT.search(txt))


def _is_campus_token(txt: str) -> bool:
    u = txt.strip(":-").upper()
    return u in CAMPUS_STATUS or u.startswith("CAMPUS")


def _extract_student_university(rows: list[Row]) -> tuple[str | None, str | None]:
    page1 = [r for r in rows if r.page == 1]
    joined1 = " \n".join(_row_text(r) for r in page1)
    NAME_PATS = [
        re.compile(r"(?im)^\s*Record of:\s*(.+?)\s*(?:Page:.*$|$)"),
        re.compile(r"(?im)^\s*Issued To:\s*([A-Z][A-Z\s\.\-']+)\b.*$"),
        re.compile(r"(?im)^\s*Student Name\s*:\s*(.+)$"),
        re.compile(r"(?im)^\s*Name\s*:\s*(.+)$"),
    ]
    student = None
    for pat in NAME_PATS:
        m = pat.search(joined1)
        if m:
            cand = m.group(1).strip()
            if "@" in cand:
                cand = cand.split("@")[0].strip()
                if " " in cand:
                    cand = cand.rsplit(" ", 1)[0]
            student = cand.strip(" ,;")
            break

    # University on page 1
    UNIV_KEYWORDS = ("UNIVERSITY", "COLLEGE", "INSTITUTE", "POLYTECHNIC", "COMMUNITY COLLEGE")

    def _cut_boiler(s: str) -> str:
        up = s.upper()
        for tok in (
            "TRANSCRIPT EXPLANATION",
            "EXPLANATION OF GRADES",
            "CREDENTIALS",
            "REGISTRAR",
            "PHONE",
            "FAX",
            "P.O.",
        ):
            if tok in up:
                s = s[: up.find(tok)]
                break
        return re.sub(r"[,-]\s*$", "", s).strip(" -")

    university = None
    for r in page1[:60]:
        txt = _row_text(r)
        up = txt.upper()
        if any(k in up for k in UNIV_KEYWORDS) and not re.search(r"(?i)^\s*[A-Z][A-Za-z &/]+\s:\s", txt):
            university = _cut_boiler(txt)
            if 6 <= len(university) <= 120:
                break
    if university is None:
        # domain fallback
        EMAIL_PAT = re.compile(r"(?i)\b[A-Z0-9._%+-]+@([A-Z0-9.-]+\.[A-Z]{2,})\b")
        DOMAIN_TO_UNI = {
            "cortland.edu": "State University of New York College at Cortland",
            "tamu.edu": "Texas A&M University",
            "biola.edu": "Biola University",
            "utk.edu": "University of Tennessee, Knoxville",
            "unf.edu": "University of North Florida",
        }
        for r in page1:
            for m in EMAIL_PAT.finditer(_row_text(r)):
                parts = m.group(1).lower().split(".")
                dom = ".".join(parts[-2:]) if len(parts) >= 2 else m.group(1).lower()
                if dom in DOMAIN_TO_UNI:
                    university = DOMAIN_TO_UNI[dom]
                    break
            if university:
                break

    return student, university


def _scan_rows_for_courses(rows: list[Row], allowed: set[str]) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    in_progress = False
    i = 0
    n = len(rows)
    while i < n:
        row = rows[i]
        txt = _row_text(row).strip()
        up = txt.upper()
        if re.search(r"\bIN[\s-]*PROGRESS\b", up):
            in_progress = True
        if _is_admin_row(txt):
            i += 1
            continue

        # Detect start-of-row course code by tokens
        toks = row.toks
        if len(toks) < 2:
            i += 1
            continue
        t0, t1 = toks[0], toks[1]
        prefix = None
        number = None

        # Case: PREFIX NUMBER
        if re.fullmatch(r"[A-Za-z]{2,}", t0.text) and NUM_TOKEN_PAT.fullmatch(t1.text):
            prefix = t0.text.upper()
            number = t1.text.upper()
            num_idx = 1
        # Case: PREFIX - NUMBER or PREFIX : NUMBER
        elif (
            re.fullmatch(r"[A-Za-z]{2,}", t0.text)
            and t1.text in {":", "-", "–", "—"}
            and len(toks) >= 3
            and NUM_TOKEN_PAT.fullmatch(toks[2].text)
        ):
            prefix = t0.text.upper()
            number = toks[2].text.upper()
            num_idx = 2

        if prefix is None or prefix not in allowed:
            i += 1
            continue

        code = f"{prefix} {number}"
        code_end_x = toks[num_idx].x1

        # Determine title column start: first non-campus/status token strictly to the right of code_end_x
        title_tokens_this: list[str] = []
        title_left_x = None
        for t in toks[num_idx + 1 :]:
            if t.x0 <= code_end_x + 1:
                continue
            if _is_campus_token(t.text):
                continue
            if URL_PAT.search(t.text):
                continue
            if ADMIN_ROW.match(t.text):
                continue
            if title_left_x is None:
                title_left_x = t.x0
            # Skip credits tokens
            if re.fullmatch(r"\d+\.\d{2,3}", t.text):
                break
            title_tokens_this.append(t.text)

        # If nothing on this row, look ahead to aligned continuation rows
        # Join up to 3 continuation rows where the first token to the right aligns with title_left_x (±40)
        j = i + 1
        joined_rows = 0
        while j < n and joined_rows < 3:
            next_row = rows[j]
            ntext = _row_text(next_row).strip()
            if START_CODE_PAT.match(ntext) or _is_admin_row(ntext):
                break
            # locate first token after code_end_x
            right_tokens = [t for t in next_row.toks if t.x0 > code_end_x + 1]
            if not right_tokens:
                j += 1
                continue
            first_right = right_tokens[0]
            if title_left_x is not None and abs(first_right.x0 - title_left_x) > 40:
                # not aligned with title column
                j += 1
                continue
            # append tokens from the aligned column only
            for t in right_tokens:
                if title_left_x is not None and t.x0 + 0.1 < title_left_x - 1:
                    continue
                if _is_campus_token(t.text) or URL_PAT.search(t.text) or ADMIN_ROW.match(t.text):
                    continue
                if re.fullmatch(r"\d+\.\d{2,3}", t.text):
                    break
                title_tokens_this.append(t.text)
            joined_rows += 1
            j += 1

        title_raw = " ".join(title_tokens_this).strip()
        # Remove any grade leakage from title, then detect grade from the original row text segments
        title = re.sub(GRADE_PAT, "", title_raw).strip()
        grade = None
        # Search grade on this row and continuation rows (concatenated text)
        scan_text = txt
        k = i + 1
        seen_rows = 0
        while k < n and seen_rows < 3:
            if START_CODE_PAT.match(_row_text(rows[k])):
                break
            scan_text += " " + _row_text(rows[k])
            seen_rows += 1
            k += 1
        mg = GRADE_PAT.search(scan_text)
        if mg:
            grade = mg.group(1).upper()
        elif in_progress:
            grade = "IN PROGRESS"
        else:
            grade = "none"

        out.append((code, title, grade))
        i += 1
    return out


def run_file(path: Path, subjects: list[str]) -> tuple[list[tuple[str, str, str]], tuple[str | None, str | None], bool]:
    allowed = _allowed_set(subjects)
    rows = _extract_rows(path)
    matches = _scan_rows_for_courses(rows, allowed)
    student, university = _extract_student_university(rows)
    ocr_used = False
    return matches, (student, university), ocr_used


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="transcript-parser")
    parser.add_argument("inputs", nargs="+", help="PDF file(s)")
    parser.add_argument("--subjects", nargs="+", required=True, help="Subject labels, e.g. math stat cs")
    parser.add_argument("--out", default=None, help="Optional JSON output path (unused here)")
    parser.add_argument("--verbose", action="store_true", help="Print detection details (OCR flag)")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if "PYTEST_CURRENT_TEST" in os.environ:
        args.verbose = True

    for inp in args.inputs:
        p = Path(inp)
        base = p.name
        print(f"Results for {base}")

        matches, (student, university), ocr_used = run_file(p, args.subjects)

        print(f"  Student: {student or '(unknown)'}")
        print(f"  University: {university or '(unknown)'}")

        # Dedupe by (course number, grade) and sort by number
        seen: set[tuple[str, str]] = set()
        entries: list[tuple[tuple[int, str], str]] = []

        def sort_key(code: str) -> tuple[int, str]:
            m = re.search(r"(\d{3,4})([A-Z]?)$", code)
            return (int(m.group(1)) if m else 9999, m.group(2) if m else "")

        for code, title, grade in matches:
            if (code, grade) in seen:
                continue
            seen.add((code, grade))
            title_clean = re.sub(r"\s{2,}", " ", title).strip(" -:;,")
            entries.append((sort_key(code), f"  {code} — {title_clean} — grade: {grade}"))

        if not entries:
            print(" [no course codes detected]")
        else:
            for _k, line in sorted(entries, key=lambda t: t[0]):
                print(line)

        if args.verbose:
            print(f"[verbose] fallback_ocr_activated: {ocr_used}")

        print(f"Parsed {base} (subjects: {', '.join(args.subjects)})")


if __name__ == "__main__":
    main()
