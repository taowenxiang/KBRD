# -*- coding: utf-8 -*-
from __future__ import annotations

import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

@dataclass
class CourseKBRow:
    term: str
    course_code: str
    course_name: str
    instructor: str
    assessment_structure: str
    has_midterm: Optional[bool]
    has_final: Optional[bool]
    avg_score: Optional[float]
    q1: Optional[float]
    median: Optional[float]
    q3: Optional[float]
    min_score: Optional[float]
    max_score: Optional[float]
    extra: Dict[str, Any]

    def to_prompt_dict(self) -> Dict[str, Any]:
        # Keep only fields that are safe and useful; avoid anything that could encode personal info.
        return {
            "term": self.term,
            "course_code": self.course_code,
            "course_name": self.course_name,
            "instructor": self.instructor,
            "assessment_structure": self.assessment_structure,
            "has_midterm": self.has_midterm,
            "has_final": self.has_final,
            "score_stats": {
                "avg": self.avg_score,
                "q1": self.q1,
                "median": self.median,
                "q3": self.q3,
                "min": self.min_score,
                "max": self.max_score,
            },
            # You can expose additional non-sensitive course metadata if present:
            "extra": {k: v for k, v in self.extra.items() if k not in {
                "notes",  # often free text; keep out by default
            }},
        }

class CourseKB:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        # Normalize columns
        for c in ["term", "course_code", "course_name"]:
            if c in self.df.columns:
                self.df[c] = self.df[c].fillna("").astype(str)
        if "course_code" in self.df.columns:
            self.df["course_code_norm"] = self.df["course_code"].str.upper().str.strip()
        if "term" in self.df.columns:
            self.df["term_norm"] = self.df["term"].str.strip()

    @staticmethod
    def load_csv(path: str) -> "CourseKB":
        df = pd.read_csv(path)
        return CourseKB(df)

    def retrieve(self, term: Optional[str], course_code: Optional[str], course_name: Optional[str]) -> Optional[CourseKBRow]:
        """
        Primary key: (term, course_code). Fallback: course_code only. Fallback: course_name contains.
        """
        term = (term or "").strip()
        course_code = (course_code or "").strip().upper()
        course_name = (course_name or "").strip()

        cand = self.df
        if term and "term_norm" in cand.columns:
            cand = cand[cand["term_norm"] == term]
        if course_code and "course_code_norm" in cand.columns:
            sub = cand[cand["course_code_norm"] == course_code]
            if len(sub) > 0:
                return self._row_from_df(sub.iloc[0])
        if course_code and "course_code_norm" in self.df.columns:
            sub = self.df[self.df["course_code_norm"] == course_code]
            if len(sub) > 0:
                return self._row_from_df(sub.iloc[0])
        if course_name and "course_name" in self.df.columns:
            # simple contains fallback
            sub = self.df[self.df["course_name"].str.contains(course_name, case=False, na=False)]
            if len(sub) > 0:
                return self._row_from_df(sub.iloc[0])
        return None

    def _row_from_df(self, r: pd.Series) -> CourseKBRow:
        def ffloat(x) -> Optional[float]:
            try:
                if pd.isna(x):
                    return None
                return float(x)
            except Exception:
                return None
        def fbool(x) -> Optional[bool]:
            if pd.isna(x):
                return None
            if isinstance(x, bool):
                return x
            s = str(x).strip().lower()
            if s in ("true","1","yes","y"):
                return True
            if s in ("false","0","no","n"):
                return False
            return None

        known = {
            "term","course_code","course_name","instructor","assessment_structure",
            "has_midterm","has_final","avg_score","q1","median","q3","min_score","max_score",
        }
        extra = {k: (None if pd.isna(r[k]) else r[k]) for k in r.index if k not in known and k != "course_code_norm" and k != "term_norm"}

        return CourseKBRow(
            term=str(r.get("term","")),
            course_code=str(r.get("course_code","")),
            course_name=str(r.get("course_name","")),
            instructor=str(r.get("instructor","")),
            assessment_structure=str(r.get("assessment_structure","")),
            has_midterm=fbool(r.get("has_midterm", None)),
            has_final=fbool(r.get("has_final", None)),
            avg_score=ffloat(r.get("avg_score", None)),
            q1=ffloat(r.get("q1", None)),
            median=ffloat(r.get("median", None)),
            q3=ffloat(r.get("q3", None)),
            min_score=ffloat(r.get("min_score", None)),
            max_score=ffloat(r.get("max_score", None)),
            extra=extra,
        )
