from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config and overlay supported environment variables.

    The code intentionally avoids a hard dependency on hydra/omegaconf so that the
    reproduction remains simple for new students.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass

    cfg_path = Path(path or "configs/default.yaml")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    if yaml is None:
        raise RuntimeError("PyYAML is required to read config files. Install pyyaml.")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Environment overrides for secrets and frequently changed model options.
    cfg.setdefault("models", {})
    cfg["models"]["qwen_api_key"] = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if os.getenv("QWEN_BASE_URL"):
        cfg["models"]["qwen_base_url"] = os.getenv("QWEN_BASE_URL")
    if os.getenv("QWEN_MODEL"):
        cfg["models"]["qwen_model"] = os.getenv("QWEN_MODEL")
    if os.getenv("QWEN_BACKEND"):
        cfg["models"]["qwen_backend"] = os.getenv("QWEN_BACKEND")
    if os.getenv("TEXTPA_OUTPUT_DIR"):
        cfg.setdefault("paths", {})["output_dir"] = os.getenv("TEXTPA_OUTPUT_DIR")
    return cfg


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: str | Path, *, indent: int = 2) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(row: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def atomic_jsonl_write(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(p)


def normalize_text(text: str) -> str:
    """Normalize ASR/reference text for prompt display and lightweight comparisons."""
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"\s+", " ", text.strip())
    return text


def text_to_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+", normalize_text(text).lower())


def run_cmd(cmd: list[str], *, cwd: str | Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, check=check, text=True, capture_output=True)


def pearsonr_safe(x: Iterable[float], y: Iterable[float]) -> float:
    xs = np.asarray(list(x), dtype=float)
    ys = np.asarray(list(y), dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[mask]
    ys = ys[mask]
    if len(xs) < 2:
        return float("nan")
    x_centered = xs - np.mean(xs)
    y_centered = ys - np.mean(ys)
    denom = float(np.sqrt(np.sum(x_centered * x_centered) * np.sum(y_centered * y_centered)))
    if denom == 0:
        return float("nan")
    return float(np.sum(x_centered * y_centered) / denom)


def coerce_float(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    m = re.search(r"[-+]?\d*\.?\d+", str(value))
    if m:
        return float(m.group(0))
    return float("nan")


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_first_json_object(text: str) -> dict[str, Any]:
    """Extract and parse the first JSON object in an LLM response.

    Qwen usually follows JSON instructions, but this makes the reproduction robust to
    explanations or Markdown code fences.
    """
    text = strip_code_fences(text)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found in LLM response")
    depth = 0
    in_str = False
    esc = False
    for i, ch in enumerate(text[start:], start=start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    obj = json.loads(candidate)
                    if not isinstance(obj, dict):
                        raise ValueError("Parsed JSON is not an object")
                    return obj
    raise ValueError("Unclosed JSON object in LLM response")
