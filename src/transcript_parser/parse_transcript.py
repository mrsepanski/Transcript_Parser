from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

# -------- Primary extractor (pdfplumber) --------
try:
    import pdfplumber  # type: ignore
except Exception:
    pdfplumber = None  # type: ignore


# -------- Optional OCR stack (PaddleOCR) --------
# NOTE: Never use deprecated `use_angle_cls`.
def _lazy_import_paddle():
    try:
        from paddleocr import PaddleOCR  # type: ignore
    except Exception:
        PaddleOCR = None  # type: ignore
    try:
        from pdf2image import convert_from_path  # type: ignore
    except Exception:
        convert_from_path = None  # type: ignore
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None  # type: ignore
    return PaddleOCR, convert_from_path, np


# ---------- Subject aliases ----------
SUBJECT_ALIASES = {
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
NUM_TOKEN_PAT = re.compile(r"(?i)^\d{3,4}[A-Z]?$")
GRADE_PAT = re.compile(
    r"(?i)(?<!\w)(A\+|A-|A|B\+|B-|B|C\+|C-|C|D\+|D-|D|E|F|P|S|U|T|I|IN PROGRESS)(?!\w)"
)
ADMIN_ROW = re.compile(
    r"(?i)^(Ehrs|GPA|TOTAL|Dean's List|Good Standing|Earned Hrs|TRANSCRIPT TOTALS|Totals?)\b"
)
URL_PAT = re.compile(r"https?://")
CREDITS_PAT = re.compile(r"\s\d+\.\d{2,3}\b")  # e.g., 3.00 or 3.000


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
    expanded = set()
    for s in subjects:
        key = s.lower()
        if key in SUBJECT_ALIASES:
            expanded.update(SUBJECT_ALIASES[key])
        else:
            expanded.add(s.upper())
    return sorted(expanded)


def _allowed_set(subjects: Iterable[str]) -> set[str]:
    return {p.upper() for p in _expand_subjects(subjects)}


def _extract_rows_pdfplumber(path: Path, y_tol: float = 2.0) -> list[Row]:
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


def _extract_rows_ocr(path: Path, dpi: int = 300, y_tol: float = 6.0) -> list[Row]:
    rows: list[Row] = []
    PaddleOCR, convert_from_path, np = _lazy_import_paddle()
    if PaddleOCR is None or convert_from_path is None:
        return rows

    try:
        # Minimal, stable init. No deprecated args.
        ocr = PaddleOCR(lang="en")  # type: ignore
    except Exception:
        return rows

    try:
        images = convert_from_path(str(path), dpi=dpi)
    except Exception:
        return rows

    rows_by_page: dict[int, list[Row]] = {}

    def push_token(page_idx: int, text: str, x0: float, y0: float, x1: float, y1: float):
        for r in rows_by_page.setdefault(page_idx, []):
            if abs(r.y - y0) <= y_tol:
                r.toks.append(Tok(text, x0, x1, y0, y1, page_idx))
                return
        rows_by_page[page_idx].append(Row(page_idx, y0, [Tok(text, x0, x1, y0, y1, page_idx)]))

    for pidx, pil_im in enumerate(images, start=1):
        try:
            im = pil_im.convert("RGB")
            arr = None
            if np is not None:
                try:
                    arr = np.array(im)  # type: ignore
                except Exception:
                    arr = None
            res = ocr.ocr(arr if arr is not None else im)  # type: ignore
        except Exception:
            continue

        if not res:
            continue
        rows_by_page.setdefault(pidx, [])
        for line in res[0]:
            try:
                box, (txt, _conf) = line
            except Exception:
                continue
            if not txt:
                continue
            xs = [pt[0] for pt in box]
            ys = [pt[1] for pt in box]
            x0, x1 = float(min(xs)), float(max(xs))
            y0, y1 = float(min(ys)), float(max(ys))
            push_token(pidx, _normalize_text(txt), x0, y0, x1, y1)

    for pidx in sorted(rows_by_page):
        page_rows = rows_by_page[pidx]
        for r in page_rows:
            r.toks.sort(key=lambda t: t.x0)
        page_rows.sort(key=lambda r: r.y)
        rows.extend(page_rows)

    rows.sort(key=lambda r: (r.page, r.y))
    return rows


def _row_text(row: Row) -> str:
    return " ".join(t.text for t in row.toks)


def _is_admin_row(txt: str) -> bool:
    return bool(ADMIN_ROW.match(txt) or URL_PAT.search(txt))


def _is_campus_token(txt: str) -> bool:
    u = txt.strip(":-").upper()
    return u in {
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


def _extract_student_university(rows: list[Row]) -> tuple[str | None, str | None]:
    page1 = [r for r in rows if r.page == 1]
    joined1 = " \n".join(_row_text(r) for r in page1)
    NAME_PATS = [
        re.compile(r"(?im)^\s*Record of:\s*(.+?)(?:\s*Page:.*$|$)"),
        re.compile(r"(?im)^\s*Issued To:\s*([A-Z][A-Z\s.\-']+)\b.*$"),
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
        if any(k in up for k in UNIV_KEYWORDS) and not re.search(
            r"(?i)^\s*[A-Z][A-Za-z &/]+\s:\s", txt
        ):
            cand = _cut_boiler(txt)
            if 6 <= len(cand) <= 120:
                university = cand
                break
    return student, university


def _scan_rows_for_courses(rows: list[Row], allowed: set[str]) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    i = 0
    n = len(rows)
    in_progress_seen = False  # restored flag: if we've seen "IN PROGRESS" anywhere
    while i < n:
        row = rows[i]
        txt = _row_text(row).strip()
        up = txt.upper()
        if "IN PROGRESS" in up:
            in_progress_seen = True
        if _is_admin_row(txt):
            i += 1
            continue

        toks = row.toks
        if len(toks) < 2:
            i += 1
            continue

        # prefix + number patterns
        prefix = None
        number = None
        num_idx = None

        if re.fullmatch(r"[A-Za-z]{2,}", toks[0].text) and NUM_TOKEN_PAT.fullmatch(toks[1].text):
            prefix = toks[0].text.upper()
            number = toks[1].text.upper()
            num_idx = 1
        elif (
            re.fullmatch(r"[A-Za-z]{2,}", toks[0].text)
            and toks[1].text in {":", "-", "–", "—"}
            and len(toks) >= 3
            and NUM_TOKEN_PAT.fullmatch(toks[2].text)
        ):
            prefix = toks[0].text.upper()
            number = toks[2].text.upper()
            num_idx = 2

        if prefix is None or prefix not in allowed or num_idx is None:
            i += 1
            continue

        code = f"{prefix} {number}"
        code_end_x = toks[num_idx].x1

        # title tokens on this row (to the right of code)
        title_tokens: list[str] = []
        title_left_x: float | None = None
        for t in toks[num_idx + 1 :]:
            if t.x0 <= code_end_x + 1:
                continue
            if _is_campus_token(t.text) or URL_PAT.search(t.text) or ADMIN_ROW.match(t.text):
                continue
            if re.fullmatch(r"\d+\.\d{2,3}", t.text):  # credits marker
                break
            if title_left_x is None:
                title_left_x = t.x0
            title_tokens.append(t.text)

        # continuation: look ahead up to 3 rows; alignment tolerance ±40 (original baseline)
        j = i + 1
        joined_rows = 0
        while j < n and joined_rows < 3:
            next_row = rows[j]
            ntext = _row_text(next_row).strip()
            if _is_admin_row(ntext):
                break
            ntoks = next_row.toks
            if (
                len(ntoks) >= 2
                and re.fullmatch(r"[A-Za-z]{2,}", ntoks[0].text)
                and NUM_TOKEN_PAT.fullmatch(ntoks[1].text)
            ):
                break

            right_tokens = [t for t in ntoks if t.x0 > code_end_x + 1]
            if not right_tokens:
                j += 1
                continue
            first_right = right_tokens[0]
            if title_left_x is not None and abs(first_right.x0 - title_left_x) > 40:
                j += 1
                continue
            for t in right_tokens:
                if title_left_x is not None and t.x0 + 0.1 < title_left_x - 1:
                    continue
                if _is_campus_token(t.text) or URL_PAT.search(t.text) or ADMIN_ROW.match(t.text):
                    continue
                if re.fullmatch(r"\d+\.\d{2,3}", t.text):
                    break
                title_tokens.append(t.text)
            joined_rows += 1
            j += 1

        title_raw = " ".join(title_tokens).strip()
        title = re.sub(GRADE_PAT, "", title_raw).strip(" -:;,")

        # grade: original behavior that yielded IN PROGRESS for certain rows
        # scan only a small window (this row + next 3), otherwise defer to in_progress_seen
        scan_text = txt
        k = i + 1
        seen_rows = 0
        while k < n and seen_rows < 3:
            if (
                len(rows[k].toks) >= 2
                and re.fullmatch(r"[A-Za-z]{2,}", rows[k].toks[0].text)
                and NUM_TOKEN_PAT.fullmatch(rows[k].toks[1].text)
            ):
                break
            scan_text += " " + _row_text(rows[k])
            seen_rows += 1
            k += 1
        mg = GRADE_PAT.search(scan_text)
        if mg:
            grade = mg.group(1).upper()
        elif "IN PROGRESS" in scan_text.upper() or in_progress_seen:
            grade = "IN PROGRESS"
        else:
            grade = "none"

        out.append((code, title, grade))
        i += 1

    return out


def _extract_rows(path: Path, prefer_ocr: bool = False) -> tuple[list[Row], bool]:
    """Try pdfplumber first; if no rows (or forced), try OCR fallback."""
    force_ocr = prefer_ocr or os.environ.get("TRANSCRIPT_FORCE_OCR", "").strip() == "1"
    rows_pdf: list[Row] = [] if force_ocr else _extract_rows_pdfplumber(path)
    if rows_pdf:
        return rows_pdf, False
    rows_ocr = _extract_rows_ocr(path)
    if rows_ocr:
        return rows_ocr, True
    return [], False


def run_file(
    path: Path, subjects: list[str], prefer_ocr: bool = False
) -> tuple[list[tuple[str, str, str]], tuple[str | None, str | None], bool]:
    allowed = _allowed_set(subjects)
    rows, ocr_used = _extract_rows(path, prefer_ocr=prefer_ocr)
    matches = _scan_rows_for_courses(rows, allowed)
    student, university = _extract_student_university(rows)
    return matches, (student, university), ocr_used


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="transcript-parser")
    parser.add_argument("inputs", nargs="+", help="PDF file(s)")
    parser.add_argument(
        "--subjects", nargs="+", required=True, help="Subject labels, e.g. math stat cs"
    )
    parser.add_argument("--out", default=None, help="Optional JSON output path (unused here)")
    parser.add_argument("--verbose", action="store_true", help="Print detection details (OCR flag)")
    parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="Force PaddleOCR fallback even if pdfplumber succeeds",
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    for inp in args.inputs:
        p = Path(inp)
        base = p.name
        print(f"Results for {base}")

        matches, (student, university), ocr_used = run_file(
            p, args.subjects, prefer_ocr=args.force_ocr
        )

        print(f"  Student: {student or '(unknown)'}")
        print(f"  University: {university or '(unknown)'}")

        # sort by numeric course number
        def sort_key(code: str):
            m = re.search(r"(\d{3,4})([A-Z]?)$", code)
            return (int(m.group(1)) if m else 9999, m.group(2) if m else "")

        seen = set()
        entries: list[tuple[tuple[int, str], str]] = []
        for code, title, grade in matches:
            if (code, grade) in seen:
                continue
            seen.add((code, grade))
            entries.append((sort_key(code), f"  {code} — {title} — grade: {grade}"))

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
