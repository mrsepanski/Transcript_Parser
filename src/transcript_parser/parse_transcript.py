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
    "math": ["MATH", "MAT", "MTH", "MA", "MATG", "MAS", "MAP", "STA", "STAT"],
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

# Strict letter grades A-F with optional +/-
GRADE_TOKEN_STRICT = re.compile(r"(?i)^(A|B|C|D|F)([+\-\u2212])?$")
# Status/loose grades (avoid unless no strict grade found)
GRADE_TOKEN_LOOSE = re.compile(r"(?i)^(P|S|U|I|T)$")
INPROG_PAT = re.compile(r"(?i)\bIN\s+PROGRESS\b")

# Row-level admin/heading detectors
ADMIN_ROW = re.compile(
    r"(?i)^(Ehrs|GPA|TOTAL|Dean's List|Good Standing|Earned Hrs|TRANSCRIPT TOTALS|Totals?)\b"
)
SEMESTER_HEADING = re.compile(
    r"(?i)^(FALL|SPRING|SUMMER|WINTER|AUTUMN|JAN|MAY|AUGUST)\s+(SEMESTER|TERM|SESSION|QUARTER)\b"
)
CUMULATIVE_HEADING = re.compile(r"(?i)^(CUMULATIVE|SUMMARY)\b")
END_OF_TRANSCRIPT = re.compile(r"(?i)END OF TRANSCRIPT")
URL_PAT = re.compile(r"https?://")

# Token-level "stop" markers that should not appear inside titles
STOP_TOKENS = {
    "EARNED",
    "EARNED:",
    "CUMULATIVE",
    "QPTS",
    "GPA",
    "ATT:",
    "ATT",
    "CREDITS",
    "SEMESTER",
    "TERM",
    "SESSION",
    "QUARTER",
    "GRADUATE",
    "UNDERGRADUATE",
    "GRAD",
    "END",
    "TRANSCRIPT",
    "EXPLANATION",
    "REGISTRAR",
    "REGISTRAR’S",
    "REGISTRAR'S",
    "OFFICE",
    # Transfer/placement markers that should stop title scanning
    "TCR",
    "TA",
    "TB",
    "TC",
    "TD",
    "TF",
}

# Additional "pre-title" tokens often appearing between code and title
# (campus, delivery mode, or institution codes). We skip these until actual title begins.
PRETITLE_TOKENS = {
    "MAIN",
    "CAMPUS",
    "CAMPUS-",
    "ONLINE",
    "REMOTE",
    "DISTANCE",
    "LEARNING",
    "DL",
    "HYBRID",
    "UNF",
    "UTK",
    "UNIV",
    "COL",
}

LEVEL_TOKEN = re.compile(r"(?i)^(UG|GR|G|U)$")  # program level markers, not grades
FLAG_SINGLE_LETTERS = {"C", "R", "H"}  # tiny flags columns


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
    s = s.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
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


def _extract_rows_pdfplumber(path: Path, y_tol: float = 3.2) -> list[Row]:
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


def _extract_rows(path: Path, prefer_ocr: bool = False) -> tuple[list[Row], bool]:
    force_ocr = prefer_ocr or os.environ.get("TRANSCRIPT_FORCE_OCR", "").strip() == "1"
    rows_pdf: list[Row] = [] if force_ocr else _extract_rows_pdfplumber(path)
    if rows_pdf:
        return rows_pdf, False
    rows_ocr = _extract_rows_ocr(path)
    if rows_ocr:
        return rows_ocr, True
    return [], False


def _row_text(row: Row) -> str:
    return " ".join(t.text for t in row.toks)


def _is_admin_row(txt: str) -> bool:
    up = txt.strip().upper()
    if ADMIN_ROW.match(txt) or URL_PAT.search(txt):
        return True
    if SEMESTER_HEADING.match(up) or CUMULATIVE_HEADING.match(up):
        return True
    if END_OF_TRANSCRIPT.search(up):
        return True
    if up.startswith("FROM:") or up.startswith("TO:"):
        return True
    return False


def _is_stop_token(t: str) -> bool:
    return (
        t.strip().upper().rstrip(":") in STOP_TOKENS or INPROG_PAT.fullmatch(t.strip()) is not None
    )


def _iter_code_pairs(toks: list[Tok]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    n = len(toks)
    k = 0
    while k < n - 1:
        if re.fullmatch(r"[A-Za-z]{2,}", toks[k].text) and NUM_TOKEN_PAT.fullmatch(
            toks[k + 1].text
        ):
            pairs.append((k, k + 1))
            k += 2
            continue
        if (
            k + 2 < n
            and re.fullmatch(r"[A-Za-z]{2,}", toks[k].text)
            and toks[k + 1].text in {":", "-", "–", "—"}
            and NUM_TOKEN_PAT.fullmatch(toks[k + 2].text)
        ):
            pairs.append((k, k + 2))
            k += 3
            continue
        k += 1
    return pairs


def _cut_university(s: str) -> str:
    up = s.upper()
    # Hard stop tokens commonly appearing after the university name
    hard_tokens = (
        "TRANSCRIPT",
        "EXPLANATION",
        "GRADE",
        "GRADES",
        "REGISTRAR",
        "OFFICE",
        "PHONE",
        "TEL",
        "FAX",
        "EMAIL",
        "P.O.",
        "PO BOX",
        "BOX",
        "WWW",
        "HTTP",
    )
    cutpoints: list[int] = []
    for tok in hard_tokens:
        idx = up.find(tok)
        if idx != -1:
            cutpoints.append(idx)
    # Also cut at the start of typical US phone numbers like (607) 753-4702 or 607-753-4702
    m_phone = re.search(r"\(?\d{3}\)?[\s\-]\d{3}[\s\-]\d{4}", s)
    if m_phone:
        cutpoints.append(m_phone.start())
    if cutpoints:
        s = s[: min(cutpoints)]
    return s.strip(" -,:;")


def _clean_student_name(s: str | None) -> str | None:
    """Remove trailing Banner-style IDs in parentheses, emails, and stray punctuation."""
    if not s:
        return s
    # Drop email if present
    if "@" in s:
        s = s.split("@", 1)[0]
    # Remove trailing parenthetical containing any digits (e.g., (730000018,T02302164))
    s = re.sub(r"\s*\((?=[^)]*[0-9])[^)]*\)\s*$", "", s)
    # Normalize spaces and strip commas/semicolons
    s = re.sub(r"\s{2,}", " ", s).strip(" ,;")
    return s or None


def _extract_student_university(rows: list[Row]) -> tuple[str | None, str | None]:
    window = [r for r in rows if r.page in (1, 2, 3, 4)]
    joined = " \n".join(_row_text(r) for r in window)

    NAME_PATS = [
        re.compile(r"(?im)^\s*Record of:\s*(.+?)(?:\s*Page:.*$|$)"),
        re.compile(r"(?im)^\s*Issued To:\s*([A-Z][A-Z\s.\-']+)\b.*$"),
        re.compile(r"(?im)^\s*Student Name\s*:\s*(.+)$"),
        re.compile(r"(?im)^\s*Name\s*:\s*(.+)$"),
        re.compile(
            r"(?im)^\s*([A-Z][A-Za-z'.\-]+,\s+[A-Z][A-Za-z'.\-]+)\s+\d{2,3}[- ]?\d{2}[- ]?\d{4}\b"
        ),
    ]
    student = None
    for pat in NAME_PATS:
        m = pat.search(joined)
        if m:
            cand = m.group(1).strip()
            if "@" in cand:
                cand = cand.split("@")[0].strip()
                if " " in cand:
                    cand = cand.rsplit(" ", 1)[0]
            student = cand.strip(" ,;")
            break

    def is_label_value(s: str) -> bool:
        return bool(re.search(r"^[A-Z][A-Za-z &/]+\s:\s", s))

    candidates: list[str] = []
    for r in window:
        txt = _row_text(r).strip()
        up = txt.upper()
        if "INSTITUTION INFORMATION CONTINUED" in up or is_label_value(txt):
            continue
        if re.search(r"(?i)\b(UNIVERSITY|COLLEGE|INSTITUTE|POLYTECHNIC|COMMUNITY COLLEGE)\b", up):
            cut = _cut_university(txt)
            if 6 <= len(cut) <= 200:
                candidates.append(cut)

    full_name = None
    base = next((c for c in candidates if "STATE UNIVERSITY OF NEW YORK" in c.upper()), None)
    addon = next(
        (c for c in candidates if "COLLEGE AT" in c.upper() or c.upper().endswith("CORTLAND")), None
    )
    if base and addon and addon not in base:
        full_name = (base + " " + addon).strip()

    def score(s: str) -> int:
        up = s.upper()
        sc = 0
        if "STATE UNIVERSITY OF NEW YORK" in up:
            sc += 4
        if "COLLEGE AT" in up:
            sc += 3
        if "CORTLAND" in up:
            sc += 2
        if "UNIVERSITY" in up:
            sc += 2
        if "COLLEGE" in up:
            sc += 1
        if 10 <= len(up) <= 90:
            sc += 1
        return sc

    university = None
    if full_name:
        university = full_name
    elif candidates:
        candidates.sort(key=score, reverse=True)
        university = candidates[0]

    # Final cleanup for student (strip trailing IDs, emails, etc.)
    student = _clean_student_name(student)

    return student, university


def _looks_like_title_word(s: str) -> bool:
    """Heuristic: a real title word often contains lowercase or is a long alpha token."""
    if any(ch.islower() for ch in s):
        return True
    # e.g., 'CRYPTOGRAPHY' (all caps) should still be okay
    return s.isalpha() and len(s) >= 5


def _clean_title_prefix(s: str) -> str:
    # Remove common catalog prefixes like 'ST:', 'TOPICS:', etc.
    s = re.sub(r"(?i)^(ST|SPECIAL\s+TOPICS|SELECTED\s+TOPICS|TOPICS)\s*[:\-]\s*", "", s).strip()
    return s


def _title_column_x_per_page(rows: list[Row]) -> dict[int, float]:
    """
    Detect the x0 of the 'Title' column header per page, if present.
    Returns a dict mapping page -> x0 for 'Title' token.
    """
    title_x: dict[int, float] = {}
    for r in rows:
        txt = _row_text(r).upper()
        if "SUBJECT" in txt and "TITLE" in txt and "GRADE" in txt:
            # look for the specific token 'Title' to get its x0
            for t in r.toks:
                if t.text.strip().upper() == "TITLE":
                    title_x[r.page] = t.x0
                    break
    return title_x


def _scan_rows_for_courses(rows: list[Row], allowed: set[str]) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    n = len(rows)
    in_progress_seen_anywhere = False

    # Precompute title header x0 per page (used to filter out Campus/Level columns)
    title_x_map = _title_column_x_per_page(rows)

    def is_any_course_row(r: Row) -> bool:
        return bool(_iter_code_pairs(r.toks))

    i = 0
    while i < n:
        row = rows[i]
        txt = _row_text(row).strip()
        up = txt.upper()
        if INPROG_PAT.search(up):
            in_progress_seen_anywhere = True
        if _is_admin_row(txt):
            i += 1
            continue

        toks = row.toks
        if len(toks) < 2:
            i += 1
            continue

        pairs = _iter_code_pairs(toks)
        used_any = False
        title_min_x = title_x_map.get(row.page, None)

        for pi, num_idx in pairs:
            prefix = toks[pi].text.upper()
            if prefix not in allowed:
                continue
            used_any = True
            number = toks[num_idx].text.upper()
            code = f"{prefix} {number}"
            code_x0 = toks[pi].x0
            code_end_x = toks[num_idx].x1

            title_tokens: list[str] = []
            title_left_x: float | None = None
            credits_x0: float | None = None

            next_pair_x0 = None
            for p2, _n2 in pairs:
                if p2 > pi:
                    next_pair_x0 = toks[p2].x0
                    break

            def is_in_progress_phrase(ntoks: list[Tok], idx: int) -> bool:
                return (
                    idx + 1 < len(ntoks)
                    and ntoks[idx].text.strip().upper() == "IN"
                    and ntoks[idx + 1].text.strip().upper() == "PROGRESS"
                )

            # ----- same row title scan -----
            right = toks[num_idx + 1 :]
            started_title = False
            for t_idx, t in enumerate(right):
                if t.x0 <= code_end_x + 1:
                    continue
                if next_pair_x0 is not None and t.x0 >= next_pair_x0 - 1:
                    break
                if t.text.strip().upper() in FLAG_SINGLE_LETTERS:
                    continue
                up_tok = t.text.strip().upper()
                if (
                    _is_stop_token(t.text)
                    or LEVEL_TOKEN.fullmatch(t.text)
                    or URL_PAT.search(t.text)
                    or is_in_progress_phrase(right, t_idx)
                ) and started_title:
                    break
                if started_title and (
                    GRADE_TOKEN_STRICT.fullmatch(t.text) or GRADE_TOKEN_LOOSE.fullmatch(t.text)
                ):
                    break
                # Skip campus/mode/level *before* the title begins
                if not started_title and (
                    up_tok in PRETITLE_TOKENS or LEVEL_TOKEN.fullmatch(t.text)
                ):
                    continue
                # Enforce a minimum x0 at the Title header, if known
                if title_min_x is not None and t.x0 < title_min_x - 1:
                    continue
                # detect credits position
                if re.fullmatch(r"\d+\.\d{2,3}", t.text):
                    credits_x0 = t.x0
                    break
                if title_left_x is None:
                    title_left_x = t.x0
                if not started_title:
                    started_title = _looks_like_title_word(t.text)
                    if not started_title and up_tok in {"ST", "ST:"}:
                        continue
                title_tokens.append(t.text)

            # ----- stitch next rows: keep tokens aligned with Title column -----
            j = i + 1
            joined_rows = 0
            while j < n and joined_rows < 3:
                next_row = rows[j]
                ntext = _row_text(next_row).strip()
                if _is_admin_row(ntext) or is_any_course_row(next_row):
                    break
                ntoks = next_row.toks
                right_tokens = [t for t in ntoks if t.x0 > code_end_x + 1]
                # Filter to Title column (if known)
                if title_min_x is not None:
                    right_tokens = [t for t in right_tokens if t.x0 >= title_min_x - 1]
                # Keep only tokens aligned with the initial title column
                aligned = []
                if title_left_x is not None:
                    for t in right_tokens:
                        if abs(t.x0 - title_left_x) <= 24.0:
                            aligned.append(t)
                if not aligned:
                    break
                if title_min_x is not None:
                    right_tokens = [t for t in right_tokens if t.x0 >= title_min_x - 1]
                if not right_tokens:
                    j += 1
                    continue
                first_right = right_tokens[0]
                if title_left_x is not None and abs(first_right.x0 - title_left_x) > 40:
                    break
                stop_hit = False
                for k, t in enumerate(aligned):
                    if t.text.strip().upper() in FLAG_SINGLE_LETTERS:
                        continue
                    if (
                        _is_stop_token(t.text)
                        or LEVEL_TOKEN.fullmatch(t.text)
                        or URL_PAT.search(t.text)
                        or GRADE_TOKEN_STRICT.fullmatch(t.text)
                        or GRADE_TOKEN_LOOSE.fullmatch(t.text)
                    ):
                        stop_hit = True
                        break
                    if k + 1 < len(right_tokens):
                        a = t.text.strip().upper()
                        b = right_tokens[k + 1].text.strip().upper()
                        if a == "IN" and b == "PROGRESS":
                            stop_hit = True
                            break
                    if re.fullmatch(r"\d+\.\d{2,3}", t.text):
                        credits_x0 = credits_x0 or t.x0
                        stop_hit = True
                        break
                    title_tokens.append(t.text)
                if stop_hit:
                    break
                joined_rows += 1
                j += 1

            title = " ".join(title_tokens).strip(" -:;,")
            title = _clean_title_prefix(title)

            def strict_grade_right_of_credits(start_row: Row) -> str | None:
                if credits_x0 is None:
                    return None
                region_tokens = [
                    t
                    for t in start_row.toks
                    if t.x0 > credits_x0 - 1 and not LEVEL_TOKEN.fullmatch(t.text)
                ]
                for t in region_tokens:
                    if GRADE_TOKEN_STRICT.fullmatch(t.text):
                        return t.text.upper().replace("\u2212", "-")
                joined = " ".join(t.text for t in region_tokens).upper().replace("\u2212", "-")
                if "IN PROGRESS" in joined:
                    return "IN PROGRESS"
                # regex variant that preserves +/- even with spaces
                m = re.search(r"(?<!\w)(A|B|C|D|F)\s*([+\-])?(?!\w)", joined)
                if m:
                    return (m.group(1) + (m.group(2) or "")).upper()
                for t in region_tokens:
                    if GRADE_TOKEN_LOOSE.fullmatch(t.text):
                        return t.text.upper()
                return None

            def grade_left_of_code(start_row: Row) -> str | None:
                left_tokens = [t for t in start_row.toks if t.x1 < code_x0 - 1]
                ignore = {
                    "GRD",
                    "GR",
                    "CR",
                    "CRED",
                    "CRED:",
                    "CREDIT",
                    "CREDITS",
                    "PTS",
                    "R",
                    "HRS",
                    "HOURS",
                }
                for t in left_tokens:
                    if t.text.strip().upper().rstrip(":") in ignore:
                        continue
                    if GRADE_TOKEN_STRICT.fullmatch(t.text):
                        return t.text.upper().replace("\u2212", "-")
                joined_left = " ".join(t.text for t in left_tokens).upper().replace("\u2212", "-")
                if "IN PROGRESS" in joined_left:
                    return "IN PROGRESS"
                m = re.search(r"(?<!\w)(A|B|C|D|F)\s*([+\-])?(?!\w)", joined_left)
                if m:
                    return (m.group(1) + (m.group(2) or "")).upper()
                for t in left_tokens:
                    if GRADE_TOKEN_LOOSE.fullmatch(t.text):
                        return t.text.upper()
                return None

            grade = strict_grade_right_of_credits(row) or grade_left_of_code(row)
            if not grade:
                window = (
                    " ".join(_row_text(r) for r in rows[i : i + 3]).upper().replace("\u2212", "-")
                )
                # preserve +/- even with spaces
                if "IN PROGRESS" in window:
                    grade = "IN PROGRESS"
                else:
                    m = re.search(r"(?<!\w)(A|B|C|D|F)\s*([+\-])?(?!\w)", window)
                    grade = (m.group(1) + (m.group(2) or "")).upper() if m else "none"

            out.append((code, title, grade))

        # Cross-row stitch (only if no allowed pair used on this row)
        if not used_any:
            if i + 1 < n and not _is_admin_row(_row_text(rows[i + 1])):
                last = toks[-1]
                if re.fullmatch(r"[A-Za-z]{2,}", last.text):
                    ntoks = rows[i + 1].toks
                    if ntoks and NUM_TOKEN_PAT.fullmatch(ntoks[0].text):
                        prefix = last.text.upper()
                        number = ntoks[0].text.upper()
                        if prefix in allowed:
                            code = f"{prefix} {number}"
                            code_end_x = ntoks[0].x1
                            credits_x0 = None
                            title_tokens = []
                            started_title = False
                            x_title_left: float | None = None
                            title_min_x = title_x_map.get(rows[i + 1].page, None)
                            for t in ntoks[1:]:
                                if t.x0 <= code_end_x + 1:
                                    continue
                                up_tok = t.text.strip().upper()
                                if t.text.strip().upper() in FLAG_SINGLE_LETTERS:
                                    continue
                                if (
                                    _is_stop_token(t.text)
                                    or LEVEL_TOKEN.fullmatch(t.text)
                                    or URL_PAT.search(t.text)
                                ):
                                    break
                                if title_min_x is not None and t.x0 < title_min_x - 1:
                                    continue
                                if re.fullmatch(r"\d+\.\d{2,3}", t.text):
                                    credits_x0 = t.x0
                                    break
                                if x_title_left is None:
                                    x_title_left = t.x0
                                if not started_title:
                                    started_title = _looks_like_title_word(t.text)
                                    if not started_title and up_tok in {"ST", "ST:"}:
                                        continue
                                title_tokens.append(t.text)
                            title = " ".join(title_tokens).strip(" -:;,")
                            title = _clean_title_prefix(title)
                            grade = None
                            if credits_x0 is not None:
                                reg = [
                                    t
                                    for t in ntoks
                                    if t.x0 > credits_x0 - 1 and not LEVEL_TOKEN.fullmatch(t.text)
                                ]
                                for t in reg:
                                    if GRADE_TOKEN_STRICT.fullmatch(t.text):
                                        grade = t.text.upper().replace("\u2212", "-")
                                        break
                                if (
                                    not grade
                                    and "IN PROGRESS" in " ".join(t.text for t in reg).upper()
                                ):
                                    grade = "IN PROGRESS"
                            out.append((code, title, grade or "none"))

        i += 1

    if in_progress_seen_anywhere:
        out = [(c, t, g if g != "none" else "IN PROGRESS") for (c, t, g) in out]
    return out


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

        # Final presentation cleanup for student (safety net)
        student = _clean_student_name(student)

        print(f"  Student: {student or '(unknown)'}")
        print(f"  University: {university or '(unknown)'}")

        def sort_key(code: str):
            m = re.search(r"(\d{3,4})([A-Z]?)$", code)
            return (int(m.group(1)) if m else 9999, m.group(2) if m else "")

        seen = set()
        entries: list[tuple[tuple[int, str], str]] = []
        for code, title, grade in matches:
            key = (code, title, grade)
            if key in seen:
                continue
            seen.add(key)
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
