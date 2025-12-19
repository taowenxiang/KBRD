# -*- coding: utf-8 -*-
"""
llm.py (robust JSON mode + repair)

Fixes:
- Robust JSON extraction (raw_decode from first '{')
- Retry on API errors AND JSON parse errors
- Optional JSON repair pass (extra call) to reduce failure rate to near-zero
- Saves a snippet of bad outputs in logs for debugging (without dumping whole content)
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

logger = logging.getLogger("ours")


@dataclass
class LLMConfig:
    base_url: Optional[str]
    api_key: Optional[str]
    model: str
    temperature: float = 0.0
    max_tokens: int = 800
    timeout_s: int = 60
    max_retries: int = 2
    # If JSON parsing fails after retries, do one repair call
    enable_json_repair: bool = True
    # Max tokens for the repair call (keep small)
    repair_max_tokens: int = 800


class LLMClient:
    def __init__(self, config: LLMConfig):
        api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set (pass api_key or export OPENAI_API_KEY).")
        self.cfg = config
        if config.base_url:
            self.client = OpenAI(api_key=api_key, base_url=config.base_url, timeout=config.timeout_s)
        else:
            self.client = OpenAI(api_key=api_key, timeout=config.timeout_s)

    # -------- public APIs --------
    def chat_json(self, messages: List[Dict[str, str]], schema_hint: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Call chat.completions and parse a JSON object.
        Retries on:
          - API/network errors
          - JSON parse errors (model emitted non-JSON / truncated JSON)
        Optionally runs a final JSON repair pass to salvage malformed output.
        """
        last_err: Optional[Exception] = None
        last_content: str = ""

        for attempt in range(self.cfg.max_retries + 1):
            try:
                content = self._call(messages, max_tokens=self.cfg.max_tokens)
                last_content = content
                return self._parse_json_object(content)
            except Exception as e:
                last_err = e
                # log compactly
                snippet = (last_content or "")[:220].replace("\n", "\\n")
                logger.warning(
                    f"LLM chat_json error (attempt {attempt+1}/{self.cfg.max_retries}): {e} | "
                    f"content_snippet='{snippet}'"
                )
                if attempt < self.cfg.max_retries:
                    time.sleep(0.8 * (attempt + 1))
                    continue

        # Final repair attempt (extra call)
        if self.cfg.enable_json_repair and last_content:
            try:
                repaired = self._repair_to_json(last_content, schema_hint=schema_hint)
                return self._parse_json_object(repaired)
            except Exception as e:
                logger.warning(f"LLM json_repair failed: {e}")

        raise last_err or RuntimeError("Unknown LLM error")

    def chat_text(self, messages: List[Dict[str, str]]) -> str:
        """Plain text response (no JSON parsing)."""
        return self._call(messages, max_tokens=self.cfg.max_tokens)

    # -------- internal --------
    def _call(self, messages: List[Dict[str, str]], max_tokens: int) -> str:
        kwargs = dict(
            model=self.cfg.model,
            messages=messages,
            temperature=self.cfg.temperature,
            max_tokens=max_tokens,
            timeout=self.cfg.timeout_s,
        )

        # Try JSON mode if gateway supports it; if not, it may ignore.
        # We only include this when we expect JSON, but leaving it doesn't harm most gateways.
        # Some gateways may error; if so, remove this block.
        try:
            kwargs["response_format"] = {"type": "json_object"}
        except Exception:
            pass

        resp = self.client.chat.completions.create(**kwargs)
        return (resp.choices[0].message.content or "").strip()

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        t = (text or "").strip()
        if t.startswith("```"):
            # remove first fence line and last fence
            t = t.split("```", 1)[-1]
            if "```" in t:
                t = t.rsplit("```", 1)[0]
        return t.strip()

    @classmethod
    def _parse_json_object(cls, text: str) -> Dict[str, Any]:
        """
        Robust JSON object extraction:
          1) strip code fences
          2) try json.loads
          3) find first '{' and use JSONDecoder.raw_decode to parse first full JSON object (ignore trailing junk)
        """
        t = cls._strip_code_fences(text)

        # direct parse
        try:
            obj = json.loads(t)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        # raw_decode from first '{'
        start = t.find("{")
        if start == -1:
            raise ValueError(f"Could not find JSON object start. First 300 chars: {t[:300]}")

        sub = t[start:]
        dec = json.JSONDecoder()
        obj, _end = dec.raw_decode(sub)
        if not isinstance(obj, dict):
            raise ValueError(f"Expected JSON object but got {type(obj)}")
        return obj

    def _repair_to_json(self, broken: str, schema_hint: Optional[Dict[str, Any]] = None) -> str:
        """
        Ask the model to re-emit a VALID JSON object only.
        This is surprisingly effective at fixing:
          - unterminated strings
          - missing commas/braces
          - extra commentary around JSON
        """
        hint = ""
        if schema_hint:
            try:
                hint = json.dumps(schema_hint, ensure_ascii=False)
            except Exception:
                hint = str(schema_hint)

        sys_msg = (
            "You are a strict JSON repair tool.\n"
            "Given an input that may contain malformed JSON or extra text, output ONE valid JSON object ONLY.\n"
            "No markdown, no explanations, no code fences.\n"
            "If some fields are missing, infer reasonable defaults.\n"
        )
        if hint:
            sys_msg += f"\nTarget schema hint (example object): {hint}\n"

        user_msg = (
            "Repair the following into a valid JSON object.\n"
            "Return ONLY the JSON object.\n\n"
            f"{broken}"
        )

        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ]
        # keep repair output bounded
        return self._call(messages, max_tokens=min(self.cfg.repair_max_tokens, 1200))
