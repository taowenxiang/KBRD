#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline 2: Heuristic "NER-like" + regex redaction -> human-readable generalization.

Goal:
- Output is human-readable (no [EMAIL] placeholders).
- Stronger than Baseline1 at: names + accommodation + unique identifying event details
  using heuristic triggers (NOT true ML NER).

Input JSONL: each line should include at least:
  - post_id
  - raw_text

Output JSONL: each line includes:
  - post_id
  - sanitized_text
  - method

Usage:
  python run_baseline2_heuristic_ner_friendly.py --in feedback_en.jsonl --out pred_baseline2.jsonl
"""

import argparse
import json
import re
from typing import Match

# ----------------------------
# Helpers
# ----------------------------

DAY = r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)"
TIME = r"\d{1,2}:\d{2}"
DASH = r"(?:-|–)"  # hyphen or en dash

def score_bucket(n: str) -> str:
    """Map a numeric score to a coarse, readable bucket."""
    try:
        x = float(n)
    except Exception:
        return "a score"
    if x < 0 or x > 100:
        return "a score"
    tens = int(x // 10) * 10
    if tens >= 100:
        return "a perfect score"
    if tens < 60:
        return "below 60"
    return f"in the {tens}s"

# ----------------------------
# Regex detectors (structured)
# ----------------------------

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\+?\d{1,3}-\d{3}-\d{4}\b")
SOCIAL_RE = re.compile(r"@\w{2,}\b")
STUDENT_ID_RE = re.compile(r"\b\d{8}\b")

SECTION_RE = re.compile(r"\bSection\s+[A-Za-z0-9]+\b")
CLASS_TIME_RE = re.compile(rf"\b(?:{DAY}(?:/{DAY})*\s+)?{TIME}(?:{DASH}{TIME})?\b")

# Dorm / dorm room
DORM_ROOM_RE = re.compile(r"\b(?:dorm\s+room|room)\s+\d{1,2}[A-Za-z]\b", re.IGNORECASE)
DORM_RE = re.compile(r"\b[A-Z][A-Za-z]+\s(?:Hall|Court|Tower|Dorm|House)\b")

# Classroom (broad-but-structured patterns; tuned to your synthetic generator style)
CLASSROOM_RE = re.compile(
    r"\b(?:"
    r"(?:[A-Z][a-z]+\s)?(?:Lab|Studio|Lecture|Auditorium|Classroom|Building|Center|Hall|Pod|Workshop)"
    r"(?:\s+[A-Za-z]+)?\s+[A-Za-z0-9-]+(?:\s+\d+)?"
    r"|Kiln\s+Lab\s+[A-Za-z0-9-]+"
    r"|Seminar\s+Room\s+[A-Za-z0-9-]+"
    r"|Seminar\s+Cove-\d+"
    r"|Greenhouse\s+Annex\s+\d+"
    r"|Workshop\s+[A-Za-z]+\-\d+"
    r"|Lantern\s+Room\s+\d+"
    r"|Room\s+[A-Za-z0-9-]+"
    r")\b"
)

# Score contexts: catches many phrasings (ended with/at, pulled a, got, scored, graded, finished with...)
SCORE_CTX_RE = re.compile(
    r"\b(?:ended with|ended at|finished with|ended up with|ended up at|came out with|walked away with|pulled a|got a|got an|got|scored|score(?:d)?|grade(?:d)?)\s+(\d{1,3})(?:\b|%)",
    re.IGNORECASE
)

# ----------------------------
# Heuristic "NER" for names
# ----------------------------

# 1) Context-based (higher precision)
NAME_CTX_RE = re.compile(r"\b(?:roommate|partner|teammate|classmate|friend)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
NAME_WITH_RE = re.compile(r"\bwith\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
NAME_SAID_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+(?:said|told|texted|dmed|thinks|thought|insisted)\b")

# 2) Generic capitalized bigram detector (lower precision; filtered)
CAP_BIGRAM_RE = re.compile(r"\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b")

# Filter out obvious non-person bigrams (places / schedule phrases)
PLACE_WORDS = {
    "Hall","Court","Tower","Dorm","House","Room","Lab","Studio","Lecture","Auditorium","Classroom",
    "Building","Center","Workshop","Seminar","Annex","Pod","Greenhouse","Kiln"
}
DAY_WORDS = {"Mon","Tue","Wed","Thu","Fri","Sat","Sun"}

def looks_like_place_bigram(w1: str, w2: str) -> bool:
    if w1 in DAY_WORDS or w2 in DAY_WORDS:
        return True
    if w1 in PLACE_WORDS or w2 in PLACE_WORDS:
        return True
    # e.g., "Student ID", "Section SEM1"
    if w1 in {"Student","Section","Dorm"}:
        return True
    return False

# ----------------------------
# Accommodation / unique-detail triggers
# ----------------------------

ACCOM_SENT_RE = re.compile(
    r"(?i)\b(?:accommodation request|accommodation note|requesting accommodation|i need an accommodation)\b[^.]*\."
)

# unique identifying detail tends to start with "I was the only one ..." in your generator
UNIQUE_SENT_RE = re.compile(
    r"(?i)\b(?:i was the only one|only one)\b[^.]*\."
)

# ----------------------------
# Main redaction
# ----------------------------

def redact_baseline2(text: str) -> str:
    out = text

    # --- accommodation / unique detail first (sentence-level) ---
    # Replace full sentence so the unique identifying content is removed.
    out = ACCOM_SENT_RE.sub("I have an accommodation request (details omitted).", out)
    out = UNIQUE_SENT_RE.sub("I had a specific personal detail (omitted).", out)

    # --- structured ---
    out = EMAIL_RE.sub("my school email", out)
    out = PHONE_RE.sub("my phone number", out)
    out = SOCIAL_RE.sub("my social handle", out)
    out = STUDENT_ID_RE.sub("my student ID", out)

    out = DORM_ROOM_RE.sub("my dorm room", out)
    out = DORM_RE.sub("my dorm", out)

    out = SECTION_RE.sub("the section", out)
    out = CLASSROOM_RE.sub("the classroom location", out)
    out = CLASS_TIME_RE.sub("the class time", out)

    # --- scores ---
    def _score_repl(m: Match) -> str:
        n = m.group(1)
        prefix = m.group(0)[: m.group(0).lower().rfind(n.lower())].rstrip()
        return prefix + " " + score_bucket(n)

    out = SCORE_CTX_RE.sub(_score_repl, out)

    # --- names (heuristic NER) ---
    # High-precision contexts
    out = NAME_CTX_RE.sub(lambda m: m.group(0).split()[0] + " a classmate", out)
    out = NAME_WITH_RE.sub("with a classmate", out)
    out = NAME_SAID_RE.sub(lambda m: "a classmate " + m.group(0).split()[-1], out)

    # Generic capitalized bigrams (filtered):
    # Replace remaining "First Last" patterns that are unlikely to be places.
    def _cap_bigram_repl(m: Match) -> str:
        w1, w2 = m.group(1), m.group(2)
        if looks_like_place_bigram(w1, w2):
            return m.group(0)
        return "a classmate"

    out = CAP_BIGRAM_RE.sub(_cap_bigram_repl, out)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input JSONL (gold or raw)")
    ap.add_argument("--out", dest="out", required=True, help="Output JSONL (pred)")
    args = ap.parse_args()

    n = 0
    with open(args.inp, "r", encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            raw = obj.get("raw_text", "")
            pid = obj.get("post_id", None)

            pred = {
                "post_id": pid,
                "sanitized_text": redact_baseline2(raw),
                "method": "baseline2_heuristic_ner_friendly",
            }
            fout.write(json.dumps(pred, ensure_ascii=False) + "\n")
            n += 1

    print(f"[OK] Wrote {n} lines to {args.out}")


if __name__ == "__main__":
    main()
