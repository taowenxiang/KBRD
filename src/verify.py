# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\+?\d{1,3}[-\s]?\d{3}[-\s]?\d{4}\b")
SOCIAL_RE = re.compile(r"@\w{2,}\b")
STUDENT_ID_RE = re.compile(r"\b\d{8}\b")

# naive "exact score" leak: remaining standalone 1-3 digit number in a score-ish context
SCORE_NUMBER_RE = re.compile(r"\b\d{1,3}\b")

@dataclass
class Finding:
    kind: str
    evidence: str

@dataclass
class VerifierResult:
    passed: bool
    findings: List[Finding]

def verify_output(
    sanitized_text: str,
    candidates: List[Dict[str, Any]],
    gold_pii: Optional[Dict[str, str]] = None,
    allow_numbers: bool = True,
) -> VerifierResult:
    """
    Conservative verifier:
    - checks for direct contact info patterns
    - checks for presence of any gold pii values (exact string) if provided
    - checks for candidate literal leaks for high-risk types
    """
    findings: List[Finding] = []

    # 1) regex-based direct patterns
    for pat, kind in [(EMAIL_RE, "email_pattern"), (PHONE_RE, "phone_pattern"), (SOCIAL_RE, "social_pattern"), (STUDENT_ID_RE, "student_id_pattern")]:
        m = pat.search(sanitized_text)
        if m:
            findings.append(Finding(kind=kind, evidence=m.group(0)))

    # 2) gold exact leaks (synthetic data)
    if gold_pii:
        for k, v in gold_pii.items():
            if not v:
                continue
            v = str(v)
            if v.strip() and v in sanitized_text:
                findings.append(Finding(kind=f"gold_leak:{k}", evidence=v))

    # 3) candidate literal leaks (esp. quasi-identifiers)
    high_risk_types = {
        "email","phone","student_id","social_handle",
        "class_time","classroom","dorm","dorm_room",
        "exact_score","other_person_name","unique_detail","accommodation_note",
    }
    for c in candidates:
        typ = c.get("type","")
        txt = c.get("text","")
        if typ in high_risk_types and txt and txt in sanitized_text:
            findings.append(Finding(kind=f"candidate_leak:{typ}", evidence=txt))

    # 4) optional: disallow raw-looking scores (numbers) if you want stricter privacy
    if not allow_numbers:
        # If any 2-3 digit numbers appear, flag (coarse, may over-flag)
        for m in SCORE_NUMBER_RE.finditer(sanitized_text):
            n = int(m.group(0))
            if 0 <= n <= 100:
                findings.append(Finding(kind="number_token", evidence=m.group(0)))

    return VerifierResult(passed=(len(findings)==0), findings=findings)
