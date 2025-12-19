# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from typing import Any, Dict, List

PLAN_SCHEMA_HINT = """
Return STRICT JSON (no markdown, no comments).
Top-level keys:
- version: "1.0"
- policy_strictness: integer 0-2
- entities: list of objects
- rewrite_goals: list[str]
- do_not_add: list[str]
- self_check: list[str]

Each entity object:
- type: one of [email, phone, student_id, social_handle, self_name, other_person_name,
               exact_score, other_person_score, class_time, classroom, dorm, dorm_room,
               accommodation_note, unique_detail, other]
- surface: the literal string in raw_text (or best guess)
- span: [start, end] or null if unsure
- risk: "high"|"med"|"low"
- action: "KEEP"|"GENERALIZE"|"DROP"|"PSEUDONYMIZE"
- replacement: the exact phrase to use in rewriting (human-readable, not bracket placeholders)
- generalization: {"style": "...", "details": "..."}  (short)
- rationale: short explanation
"""

def build_planner_messages(raw_text: str, course_kb: Dict[str, Any], candidates: List[Dict[str, Any]], policy_strictness: int) -> List[Dict[str, str]]:
    """
    Stage-1: produce a structured plan for "bold but meaning-preserving" rewriting using prior knowledge.
    """
    def to_jsonable(x):
      if isinstance(x, dict):
          return {k: to_jsonable(v) for k, v in x.items()}
      if isinstance(x, (list, tuple)):
          return [to_jsonable(v) for v in x]
      if hasattr(x, "item"):
          return x.item()
      if hasattr(x, "isoformat"):
          try:
              return x.isoformat()
          except Exception:
              return str(x)
      return x

    kb_json = json.dumps(to_jsonable(course_kb), ensure_ascii=False, indent=2)
    cand_json = json.dumps(candidates[:60], ensure_ascii=False, indent=2)  # cap to avoid huge prompts

    system = (
        "You are a privacy-preserving rewriting planner.\n"
        "Goal: create a structured plan to rewrite a student's course feedback into a HUMAN-READABLE version\n"
        "that preserves meaning and usefulness while reducing identifiability.\n"
        "You must NOT output the rewritten text here; output only a JSON plan.\n"
        "Never include bracket placeholders like [EMAIL]. Use natural phrases like 'my school email', 'a classmate', etc.\n"
        "Use the provided course prior knowledge to make GENERALIZATION MORE INFORMATIVE (calibrated), not less.\n"
        "Do NOT hallucinate facts not supported by raw_text or course_kb."
    )

    user = f"""
POLICY STRICTNESS = {policy_strictness} (0=bold/informative, 1=balanced, 2=conservative)

Course prior knowledge (course_kb):
{kb_json}

Candidate sensitive spans (may be incomplete/noisy):
{cand_json}

Raw feedback text:
{raw_text}

Planning requirements:
1) Identify ALL sensitive details (direct PII + quasi-identifiers + unique details).
2) Prefer **calibrated generalization** that preserves meaning.
   Examples (do NOT limit yourself to scores):
   - Scores: if course_kb.score_stats has q1/median/q3, map exact numbers to relative positions:
       * <= q1 -> "around the lower quartile (around Q1)"
       * between q1 and median -> "below the median (between Q1 and median)"
       * between median and q3 -> "above the median (between median and Q3)"
       * >= q3 -> "around the upper quartile (around Q3)"
     If stats missing, use coarse but still meaningful buckets (e.g., "mid-60s", "around passing").
   - Time: replace exact day/time with coarse windows (weekday morning/afternoon/evening), preserve constraints like
     "early", "late", "back-to-back", "conflict".
   - Location: replace building/room/dorm specifics with type-level ("a campus classroom", "campus housing") while preserving
     relevant attributes (far walk, noisy, cramped, etc.).
   - People: use stable role-based pseudonyms (e.g., "Classmate A", "Roommate A") and keep relationships.
   - Unique detail/accommodation: keep category + impact, drop uniquely identifying specifics.
3) Keep the complaint/suggestion/evidence and sentiment strength.
4) Output STRICT JSON following this schema:
{PLAN_SCHEMA_HINT}
"""
    return [{"role":"system","content":system},{"role":"user","content":user}]

def build_writer_messages(raw_text: str, course_kb: Dict[str, Any], plan_json: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Stage-2: produce the final rewritten text following the plan.
    """
    def to_jsonable(x):
      if isinstance(x, dict):
          return {k: to_jsonable(v) for k, v in x.items()}
      if isinstance(x, (list, tuple)):
          return [to_jsonable(v) for v in x]
      if hasattr(x, "item"):
          return x.item()
      if hasattr(x, "isoformat"):
          try:
              return x.isoformat()
          except Exception:
              return str(x)
      return x

    kb_json = json.dumps(to_jsonable(course_kb), ensure_ascii=False, indent=2)
    plan_str = json.dumps(plan_json, ensure_ascii=False, indent=2)

    system = (
        "You are a privacy-preserving rewriter.\n"
        "Rewrite the feedback into a HUMAN-READABLE version while preserving meaning and usefulness.\n"
        "You MUST follow the provided plan exactly.\n"
        "Do NOT add facts beyond raw_text + course_kb.\n"
        "Do NOT use bracket placeholders like [EMAIL].\n"
        "Return STRICT JSON with keys: sanitized_text, checks.\n"
        "checks is a list of self-check results like {\"name\":\"...\",\"passed\":true/false,\"note\":\"...\"}."
    )

    user = f"""
Course prior knowledge (course_kb):
{kb_json}

Plan (JSON):
{plan_str}

Raw feedback text:
{raw_text}

Now rewrite. Requirements:
- Apply entity replacements and actions from plan.
- Keep meaning: complaint, evidence, suggestion, sentiment.
- If plan says DROP something, remove it cleanly (no awkward gaps).
- Keep output concise but readable.

Output STRICT JSON only:
{{
  "sanitized_text": "...",
  "checks": [
    {{"name":"no_direct_contact_info","passed":true,"note":""}},
    {{"name":"no_exact_score_or_exact_time_location","passed":true,"note":""}},
    {{"name":"no_unique_identifying_detail","passed":true,"note":""}},
    {{"name":"no_hallucination","passed":true,"note":""}}
  ]
}}
"""
    return [{"role":"system","content":system},{"role":"user","content":user}]
