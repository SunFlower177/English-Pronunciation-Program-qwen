"""Small shared Qwen JSON helper for non-pronunciation modes."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from ..contracts import CaptSettings


def qwen_json(
    settings: CaptSettings,
    system: str,
    user: str,
    max_tokens: int = 1200,
) -> tuple[Optional[Dict[str, Any]], str]:
    if not settings.use_qwen_text:
        return None, ""
    if not settings.qwen_api_key:
        return None, "Qwen disabled: QWEN_API_KEY is not set."
    raw = qwen_chat_completion(
        settings,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
    )
    return parse_json_object(raw), raw


def qwen_chat_completion(settings: CaptSettings, messages: List[Dict[str, str]], max_tokens: int = 1200) -> str:
    import urllib.error
    import urllib.request

    payload: Dict[str, Any] = {
        "model": settings.qwen_model,
        "messages": messages,
        "temperature": settings.qwen_temperature,
        "response_format": {"type": "json_object"},
        "max_tokens": max_tokens,
    }
    url = settings.qwen_base_url.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {settings.qwen_api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")[:1000]
        raise RuntimeError(f"Qwen API HTTP {exc.code}: {detail}") from exc
    return data.get("choices", [{}])[0].get("message", {}).get("content", "")


def parse_json_object(raw: str) -> Optional[Dict[str, Any]]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
