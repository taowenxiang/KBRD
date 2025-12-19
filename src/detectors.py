# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Candidate:
    type: str
    text: str
    start: Optional[int] = None
    end: Optional[int] = None
    confidence: float = 0.5
    note: str = ""

# Common regex detectors (lightweight, language-agnostic)
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\+?\d{1,3}[-\s]?\d{3}[-\s]?\d{4}\b")
SOCIAL_RE = re.compile(r"@\w{2,}\b")
STUDENT_ID_RE = re.compile(r"\b\d{8}\b")

DAY = r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)"
TIME = r"\d{1,2}:\d{2}"
DASH = r"(?:-|–)"
CLASS_TIME_RE = re.compile(rf"\b(?:{DAY}(?:/{DAY})*\s+)?{TIME}(?:\s*{DASH}\s*{TIME})?\b")

# Very lightweight classroom/location patterns seen in synthetic data
CLASSROOM_RE = re.compile(
    r"\b(?:"
    r"(?:[A-Z][a-z]+\s)?(?:Lab|Studio|Lecture|Auditorium|Classroom|Building|Center|Hall|Pod|Workshop|Seminar)"
    r"(?:\s+[A-Za-z]+)?\s+[A-Za-z0-9-]+(?:\s+\d+)?"
    r"|Greenhouse\s+Annex\s+\d+"
    r"|Seminar\s+Cove-\d+"
    r"|Room\s+[A-Za-z0-9-]+"
    r")\b"
)

DORM_RE = re.compile(r"\b[A-Z][A-Za-z]+\s(?:Hall|Court|Tower|Dorm|House)\b")
DORM_ROOM_RE = re.compile(r"\b(?:dorm\s+room|room)\s+\d{1,2}[A-Za-z]\b", re.IGNORECASE)

# Score patterns (contextual)
SCORE_CTX_RE = re.compile(
    r"\b(?:ended with|ended at|finished with|ended up with|ended up at|came out with|walked away with|pulled a|got a|got an|got|scored|score(?:d)?|grade(?:d)?|final(?:\s+score)?|midterm(?:\s+score)?|quiz(?:\s+score)?)\s*(?:was|:|=)?\s+(\d{1,3})(?:\b|%)",
    re.IGNORECASE
)

# Name heuristics (intentionally conservative; planner LLM handles the rest)
NAME_CTX_RE = re.compile(r"\b(?:roommate|partner|teammate|classmate|friend)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
NAME_WITH_RE = re.compile(r"\bwith\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")

def _find_all(regex: re.Pattern, text: str, typ: str, conf: float, note: str="") -> List[Candidate]:
    out: List[Candidate] = []
    for m in regex.finditer(text):
        out.append(Candidate(type=typ, text=m.group(0), start=m.start(), end=m.end(), confidence=conf, note=note))
    return out

def detect_candidates(raw_text: str) -> List[Candidate]:
    cands: List[Candidate] = []
    cands += _find_all(EMAIL_RE, raw_text, "email", 0.95)
    cands += _find_all(PHONE_RE, raw_text, "phone", 0.85)
    cands += _find_all(SOCIAL_RE, raw_text, "social_handle", 0.80)
    cands += _find_all(STUDENT_ID_RE, raw_text, "student_id", 0.90)
    cands += _find_all(DORM_ROOM_RE, raw_text, "dorm_room", 0.75)
    cands += _find_all(DORM_RE, raw_text, "dorm", 0.70)
    cands += _find_all(CLASS_TIME_RE, raw_text, "class_time", 0.70)
    cands += _find_all(CLASSROOM_RE, raw_text, "classroom", 0.70)

    # scores: store only the number, but keep match span
    for m in SCORE_CTX_RE.finditer(raw_text):
        cands.append(Candidate(type="exact_score", text=m.group(1), start=m.start(1), end=m.end(1), confidence=0.65, note="contextual_score"))

    # names in limited contexts
    for m in NAME_CTX_RE.finditer(raw_text):
        cands.append(Candidate(type="other_person_name", text=m.group(1), start=m.start(1), end=m.end(1), confidence=0.55, note="name_in_role_context"))
    for m in NAME_WITH_RE.finditer(raw_text):
        cands.append(Candidate(type="other_person_name", text=m.group(1), start=m.start(1), end=m.end(1), confidence=0.45, note="name_after_with"))

    # Deduplicate exact (type,text,start,end)
    seen = set()
    uniq: List[Candidate] = []
    for c in cands:
        key = (c.type, c.text, c.start, c.end)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq
