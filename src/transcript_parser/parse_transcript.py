#!/usr/bin/env python3
import os
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("PADDLE_LOG_LEVEL", "ERROR")
os.environ.setdefault("FLAGS_use_mkldnn", "0")

import argparse
import json
import re
import shutil
from typing import Dict, List, Optional, Tuple
from difflib import get_close_matches, SequenceMatcher

import pdfplumber
from pdf2image import convert_from_path
from paddleocr import PaddleOCR  # type: ignore

SUBJECT_VARIANTS: Dict[str, List[str]] = {
    "MATH": ["MATH", "MAT", "MA", "MTH"],
    "STAT": ["STAT", "STA"],
    "PHYS": ["PHYS", "PHY", "PHYSICS"],
    "CS":   ["CS", "CPSC", "CSCI", "COMP", "COMPSCI", "CSC"],
}

LETTER_GRADE_PAT = re.compile(r"^(?:[ABCDF][+-]?|P|S|U|CR|NC|NCR|IP|IN|W|WP|WF|AU|NR|H|S/U)$", re.IGNORECASE)
NUMERIC_GRADE_PAT = re.compile(r"^(?:\d{2,3}(?:\.\d+)?)$")

COURSE_PAT = re.compile(r"\b([A-Z]{2,6})\s*[- ]?\s*(\d{2,4}[A-Z]?)\b")
COURSE_ANCHORED_PAT = re.compile(r"^\s*([A-Z]{2,6})\s*[- ]?\s*(\d{2,4}[A-Z]?)\b")

UNIV_PHRASE_PAT = re.compile(r"\b([A-Z][A-Za-z&.'-]+(?:\s+[A-Z][A-Za-z&.'-]+){0,5}\s+(?:University|College|Institute))\b")

NAME_LINE_PATS = [
    re.compile(r"^\s*Name\s*[:\-–]?\s*(?P<name>.+?)\s*$", re.IGNORECASE),
    re.compile(r"^\s*Student\s+Name\s*[:\-–]?\s*(?P<name>.+?)\s*$", re.IGNORECASE),
    re.compile(r"^\s*NAME\s*[:\-–]?\s*(?P<name>.+?)\s*$", re.IGNORECASE),
]
NAME_BETWEEN_PAT = re.compile(r"Name\s*[:\-–]?\s*(?P<name>[A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){1,5})", re.IGNORECASE | re.DOTALL)

ROMAN_PAT = re.compile(r"^(?:I|II|III|IV|V|VI|VII|VIII|IX|X)$", re.IGNORECASE)

MATH_COURSE_CANON = [
    "Calculus I","Calculus II","Calculus III",
    "Linear Algebra",
    "Abstract Algebra","Abstract Algebra I","Abstract Algebra II",
    "Algebraic Structures","Algebraic Structures I","Algebraic Structures II",
    "Real Analysis","Real Analysis I","Real Analysis II","Real Variables",
    "Complex Analysis",
    "Differential Equations","Ordinary Differential Equations","Partial Differential Equations",
    "Numerical Methods","Numerical Analysis",
    "Discrete Mathematics",
    "Transition to Abstract Math","Transition to Advanced Mathematics",
    "Combinatorics",
    "Probability","Mathematical Probability",
    "Statistics","Mathematical Statistics",
    "Topology","Graph Theory",
    "Number Theory",
    "Mathematical Modeling",
    "Trigonometry","College Algebra"
]

def extract_pdf_text(path: str, max_pages: Optional[int]) -> Tuple[str, int]:
    chunks: List[str] = []
    with pdfplumber.open(path) as pdf:
        pages = pdf.pages[:max_pages] if max_pages else pdf.pages
        for p in pages:
            chunks.append(p.extract_text() or "")
    return "\n".join(chunks), len(pages)

def _pre_resize_images(images, max_side: int = 3600):
    resized = []
    for im in images:
        w, h = im.size
        side = max(w, h)
        if side > max_side:
            scale = max_side / float(side)
            new_w, new_h = int(w * scale), int(h * scale)
            resized.append(im.resize((new_w, new_h)))
        else:
            resized.append(im)
    return resized

def _init_paddleocr(lang: str = "en", use_gpu: bool = False):
    # Initialize PaddleOCR (avoid the deprecated angle classification flag entirely).
    try:
        return PaddleOCR(lang=lang, use_textline_orientation=True, use_gpu=use_gpu, show_log=False)
    except TypeError:
        # For older PaddleOCR versions lacking `use_textline_orientation`
        return PaddleOCR(lang=lang, use_gpu=use_gpu, show_log=False)
    except Exception:
        try:
            return PaddleOCR(lang=lang, use_gpu=use_gpu, show_log=False)
        except Exception:
            return None

def _paddle_text_from_result(res) -> str:
    lines: List[str] = []
    if not res:
        return ""
    try:
        for page in res:
            for item in page or []:
                if isinstance(item, list) and len(item) >= 2:
                    maybe = item[1]
                    if isinstance(maybe, (list, tuple)) and maybe:
                        txt = maybe[0]
                        if isinstance(txt, str):
                            lines.append(txt)
    except Exception:
        pass
    if not lines and isinstance(res, (list, tuple)):
        try:
            for item in res:
                if isinstance(item, (list, tuple)) and item and isinstance(item[0], str):
                    lines.append(item[0])
        except Exception:
            pass
    return "\n".join(lines)

def ocr_pdf_with_paddle(path: str, lang: str = "en", use_gpu: bool = False, dpi: int = 300) -> str:
    try:
        pages = convert_from_path(path, dpi=dpi)
    except Exception:
        return ""
    ocr = _init_paddleocr(lang=lang, use_gpu=use_gpu)
    if ocr is None:
        return ""
    pages = _pre_resize_images(pages, max_side=3600)
    texts: List[str] = []
    for im in pages:
        try:
            res = ocr.ocr(im, cls=True)
        except Exception:
            try:
                res = ocr.predict([im])
            except Exception:
                res = None
        texts.append(_paddle_text_from_result(res))
    return "\n".join(texts)

def _clean_person_tokens(raw: str) -> Optional[str]:
    tokens = raw.split()
    out: List[str] = []
    for tok in tokens:
        pure = re.sub(r"[^A-Za-z'\-]", "", tok)
        if not pure:
            continue
        if re.fullmatch(r"[A-Z][a-z]+(?:'[A-Za-z]+)?", pure):
            out.append(pure)
        elif re.fullmatch(r"[A-Z]\.?", tok):
            out.append(tok[0])
        elif pure.isupper() and len(pure) <= 2:
            out.append(pure)
        else:
            break
        if len(out) >= 6:
            break
    if 2 <= len(out) <= 6:
        return " ".join(out)
    return None

def parse_name(text: str) -> Optional[str]:
    text_sp = text.replace("\u00A0", " ")
    lines = [ln for ln in text_sp.splitlines() if ln.strip()]
    top = lines[:150]
    for line in top:
        for pat in NAME_LINE_PATS:
            m = pat.search(line)
            if m:
                nm = _clean_person_tokens(m.group("name"))
                if nm:
                    return nm
    m = NAME_BETWEEN_PAT.search(" ".join(top))
    if m:
        nm = _clean_person_tokens(m.group("name"))
        if nm:
            return nm
    for ln in top[:60]:
        t = re.sub(r"[^A-Za-z'\- ]", " ", ln).strip()
        if not t:
            continue
        if 2 <= len(t.split()) <= 6 and t == t.title():
            if not re.search(r"\b(University|College|Institute|Official|Transcript|Registrar|Campus|Address)\b", t, re.I):
                nm = _clean_person_tokens(t)
                if nm:
                    return nm
    return None

def parse_university(text: str) -> Optional[str]:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    for scope in (lines[:120], lines):
        for line in scope:
            m_iter = list(UNIV_PHRASE_PAT.finditer(line))
            if m_iter:
                phrase = m_iter[0].group(1)
                return " ".join(w if (w.isupper() and len(w) <= 3) else w.capitalize() for w in phrase.split())
        for line in scope:
            if "UNIVERSITY" in line.upper() and line.strip() == line.upper() and len(line.split()) <= 5:
                return line.title().strip()
    return None

def _is_term_token(tok: str) -> bool:
    t = tok.strip().strip(",.;:").lower()
    if t in {"fall","spring","summer","winter","fa","sp","su","wi"}:
        return True
    months = {"january","february","march","april","may","june","july","august","september","october","november","december"}
    if t in months:
        return True
    if re.fullmatch(r"(19|20)\d{2}", t):
        return True
    return False

def _is_credit_or_meta(tok: str) -> bool:
    t = tok.strip().strip(",.;:").lower()
    if t in {"cr","credit","credits","unit","units","hrs","hour","hours","gpa","quality","points","attempted","earned","total"}:
        return True
    if re.fullmatch(r"\d+(?:\.\d+)?", t):
        return True
    return False

def _detect_grade(tokens: List[str]) -> Optional[str]:
    last: Optional[str] = None
    for raw in tokens:
        tok = raw.strip(",.;:()[]{}|\"'")
        test = tok.rstrip(".:,;)]}")
        if LETTER_GRADE_PAT.match(test):
            last = test.upper()
        elif NUMERIC_GRADE_PAT.match(test):
            try:
                val = float(test)
                if 60.0 <= val <= 100.0:
                    last = test
            except ValueError:
                pass
    return last

def _normalize_phrase(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z ]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"(.)\1{2,}", r"\1\1", s)
    return s

def _fuzzy_fix_course_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    parts = name.split()
    suffix = ""
    if parts and ROMAN_PAT.fullmatch(parts[-1]):
        suffix = " " + parts[-1].upper()
        base = " ".join(parts[:-1])
    else:
        base = name

    base_norm = _normalize_phrase(base)
    if not base_norm:
        return name

    candidates = get_close_matches(base_norm.title(), MATH_COURSE_CANON, n=1, cutoff=0.78)
    if candidates:
        return candidates[0] + suffix

    tokens = base_norm.split()
    canon_tokens = sorted({t for phrase in MATH_COURSE_CANON for t in phrase.lower().split()}, key=len, reverse=True)
    chosen: List[str] = []
    for tok in tokens:
        best = None
        best_ratio = 0.0
        for ct in canon_tokens:
            r = SequenceMatcher(None, tok, ct).ratio()
            if r > best_ratio:
                best_ratio = r
                best = ct
        if best and best_ratio >= 0.65:
            if not chosen or chosen[-1] != best:
                chosen.append(best)
    if chosen:
        phrase = " ".join(w.capitalize() for w in chosen)
        snap = get_close_matches(phrase, MATH_COURSE_CANON, n=1, cutoff=0.72)
        if snap:
            phrase = snap[0]
        return phrase + suffix

    return name

def _extract_course_name_and_grade(segment: str) -> Tuple[Optional[str], Optional[str]]:
    seg = re.sub(r"^[\s\-–—:|.,;/]+", "", segment)
    raw_tokens = seg.split()
    grade = _detect_grade(raw_tokens)

    name_tokens: List[str] = []
    for raw in raw_tokens:
        tok_clean = raw.strip(",.;:()[]{}|\"'")
        test = tok_clean.rstrip(".:,;)]}")
        if grade and test.upper() == str(grade).upper():
            break
        if _is_term_token(tok_clean) or _is_credit_or_meta(tok_clean):
            break
        if tok_clean and re.search(r"[A-Za-z]", tok_clean):
            name_tokens.append(tok_clean)

    while name_tokens and not re.search(r"[A-Za-z]", name_tokens[-1]):
        name_tokens.pop()

    name = " ".join(name_tokens).strip() if name_tokens else None
    if name and len(name) < 2:
        name = None
    name = _fuzzy_fix_course_name(name)
    return name, grade

def parse_courses(text: str, subject_variants: Dict[str, List[str]]) -> List[Tuple[str, Optional[str], Optional[str]]]:
    lines = text.splitlines()
    results: List[Tuple[str, Optional[str], Optional[str]]] = []

    i = 0
    while i < len(lines):
        raw_line = lines[i]
        line = raw_line.strip()
        if not line:
            i += 1
            continue

        matches = list(COURSE_PAT.finditer(line))
        if not matches:
            i += 1
            continue

        for m_index, m in enumerate(matches):
            subj = m.group(1).upper()
            num  = m.group(2).upper()
            code = f"{subj} {num}"

            seg_end = matches[m_index + 1].start() if m_index + 1 < len(matches) else len(line)
            segment = line[m.end():seg_end]

            appended = 0
            j = i + 1
            while j < len(lines) and appended < 2 and len(segment.strip()) < 6:
                next_line = lines[j].strip()
                if not next_line or COURSE_ANCHORED_PAT.match(next_line):
                    break
                cut = next_line
                tokens = cut.split()
                trimmed: List[str] = []
                for t in tokens:
                    if _is_term_token(t) or _is_credit_or_meta(t):
                        break
                    trimmed.append(t)
                if trimmed:
                    segment += " " + " ".join(trimmed)
                    appended += 1
                    j += 1
                else:
                    break

            name, grade = _extract_course_name_and_grade(segment)
            results.append((code, name, grade))

        i += 1

    seen = set()
    out: List[Tuple[str, Optional[str], Optional[str]]] = []
    for c in results:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out

def pick_text(pdf_text: str, paddle_text: str) -> Tuple[str, str]:
    if len(pdf_text) < 100 and paddle_text:
        return paddle_text, "paddle"
    def score(txt: str) -> Tuple[int, int, int]:
        nm = 1 if parse_name(txt) else 0
        crs = len(parse_courses(txt, SUBJECT_VARIANTS))
        return (nm, crs, min(len(txt)//1000, 50))
    candidates = [("pdf", pdf_text), ("paddle", paddle_text)]
    best = max(candidates, key=lambda kv: score(kv[1]))
    return best[1], best[0]

def main():
    parser = argparse.ArgumentParser(description="Extract transcript details (auto OCR; Paddle fallback)")
    parser.add_argument("inputs", nargs="+", help="PDF files to process")
    parser.add_argument("--subjects", nargs="+", required=True, help="Subject codes to search (e.g., math stat)")
    parser.add_argument("--out", help="Optional path to write JSON output")
    parser.add_argument("--ocr-dpi", type=int, default=300, help="OCR DPI for pdf2image (default: 300)")
    parser.add_argument("--ocr-lang", type=str, default="en", help="PaddleOCR language (default: en)")
    parser.add_argument("--ocr-gpu", action="store_true", help="Use GPU for PaddleOCR when available")
    parser.add_argument("--max-pages", type=int, default=None, help="Max pages to read for speed (optional)")
    parser.add_argument("--debug-ocr", type=str, default=None, help="Folder to dump raw OCR texts (optional)")
    args = parser.parse_args()

    req = {s.upper() for s in args.subjects}
    variants = {k: v for k, v in SUBJECT_VARIANTS.items() if k in req}
    for s in req:
        variants.setdefault(s, [s])

    results = []
    for path in args.inputs:
        pdf_text, pdf_pages = extract_pdf_text(path, args.max_pages)
        paddle_text = ocr_pdf_with_paddle(path, lang=args.ocr_lang, use_gpu=args.ocr_gpu, dpi=args.ocr_dpi)

        chosen_text, chosen_source = pick_text(pdf_text, paddle_text)

        nm = parse_name(chosen_text)
        univ = parse_university(chosen_text)
        all_courses = parse_courses(chosen_text, variants)

        allowed_prefixes = {v for arr in variants.values() for v in arr}
        filtered = [(c, n, g) for (c, n, g) in all_courses if c.split()[0] in allowed_prefixes]

        if args.debug_ocr:
            os.makedirs(args.debug_ocr, exist_ok=True)
            base = os.path.splitext(os.path.basename(path))[0]
            with open(os.path.join(args.debug_ocr, f"{base}.pdf.txt"), "w", encoding="utf-8") as f:
                f.write(pdf_text)
            if paddle_text:
                with open(os.path.join(args.debug_ocr, f"{base}.paddle.txt"), "w", encoding="utf-8") as f:
                    f.write(paddle_text)

        print(f"Results for {path}:")
        print(f"  Student Name: {nm or 'N/A'}")
        print(f"  University:   {univ or 'N/A'}")
        hdr = ", ".join(sorted(req))
        print(f"  Courses ({hdr}):")
        for code, name, grade in filtered:
            if grade and name:
                print(f"    {code} {name}, grade: {grade}")
            elif grade:
                print(f"    {code}, grade: {grade}")
            elif name:
                print(f"    {code} {name}")
            else:
                print(f"    {code}")

        notes = {
            "text_source": chosen_source,
            "pdf_pages": pdf_pages,
            "pdf_text_len": len(pdf_text),
            "paddle_len": len(paddle_text) if paddle_text else 0,
            "paddleocr_ok": bool(paddle_text),
            "pdf2image_ok": True,
            "pdftoppm": shutil.which("pdftoppm") or "not_found",
        }
        print("  Notes: " + "; ".join(f"{k}={v}" for k, v in notes.items()))

        results.append({
            "file": path,
            "name": nm,
            "university": univ,
            "courses": [{"code": c, "name": n, "grade": g} for c, n, g in filtered],
            "notes": notes,
        })

    if args.out:
        try:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Wrote JSON to {args.out}")
        except Exception as e:
            print(f"Failed to write JSON: {e}")

if __name__ == "__main__":
    main()
