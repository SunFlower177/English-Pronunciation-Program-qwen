from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from .prompts import TextPAPrompt
from .utils import coerce_float, extract_first_json_object


class ChatClient(Protocol):
    def complete(self, prompt: TextPAPrompt) -> str: ...


@dataclass
class LLMResult:
    accuracy: float
    fluency: float
    reasoning: str
    raw_response: str
    valid_json: bool
    attempts: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "Accuracy": self.accuracy,
            "Fluency": self.fluency,
            "Reasoning": self.reasoning,
            "raw_response": self.raw_response,
            "valid_json": self.valid_json,
            "attempts": self.attempts,
        }


class OpenAICompatibleQwenClient:
    """Qwen API client using an OpenAI-compatible endpoint.

    Works with Alibaba Cloud Model Studio/DashScope compatible-mode endpoints and with
    a local vLLM server started by `vllm serve Qwen/Qwen2.5-7B-Instruct`.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None,
        base_url: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: float = 120.0,
        enable_thinking: bool | None = None,
    ) -> None:
        if not api_key:
            # vLLM local endpoints can use EMPTY; DashScope needs a real key.
            api_key = "EMPTY"
        try:
            api_key.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError(
                "QWEN_API_KEY contains non-ASCII characters. Replace any placeholder text with the real API key."
            ) from exc
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking

    def complete(self, prompt: TextPAPrompt) -> str:
        extra_body = None
        if self.enable_thinking is not None:
            extra_body = {"enable_thinking": self.enable_thinking}
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra_body=extra_body,
        )
        return resp.choices[0].message.content or ""


class LocalTransformersQwenClient:
    """Local Qwen inference using Hugging Face Transformers."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        temperature: float = 0.0,
        max_new_tokens: int = 512,
        device_map: str = "auto",
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map=device_map,
            trust_remote_code=True,
        )
        self.model.eval()
        self.model_id = model_id
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    @torch.inference_mode()
    def complete(self, prompt: TextPAPrompt) -> str:
        messages = [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        do_sample = self.temperature > 0
        generated = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            temperature=self.temperature if do_sample else None,
        )
        generated = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated)]
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]


class MockQwenClient:
    """Deterministic mock client for tests and pipeline debugging without LLM cost."""

    def complete(self, prompt: TextPAPrompt) -> str:
        h = hashlib.sha1((prompt.system + prompt.user).encode("utf-8")).hexdigest()
        # Use stable pseudo-scores in [1, 5].
        a = 1 + (int(h[:4], 16) / 0xFFFF) * 4
        f = 1 + (int(h[4:8], 16) / 0xFFFF) * 4
        return (
            '{"Accuracy": %.2f, "Fluency": %.2f, '
            '"Reasoning": "Mock response for pipeline testing; not a real pronunciation assessment."}'
            % (a, f)
        )


def make_qwen_client(cfg: dict[str, Any]) -> ChatClient:
    models = cfg.get("models", {})
    backend = models.get("qwen_backend", "openai_compatible")
    model = models.get("qwen_model", "qwen-plus")
    if backend == "mock":
        return MockQwenClient()
    if backend == "openai_compatible":
        return OpenAICompatibleQwenClient(
            model=model,
            api_key=models.get("qwen_api_key"),
            base_url=models.get("qwen_base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            temperature=float(models.get("temperature", 0.0)),
            max_tokens=int(models.get("max_tokens", 512)),
            enable_thinking=models.get("qwen_enable_thinking", False),
        )
    if backend == "local_transformers":
        return LocalTransformersQwenClient(
            model_id=model,
            temperature=float(models.get("temperature", 0.0)),
            max_new_tokens=int(models.get("max_tokens", 512)),
        )
    raise ValueError(f"Unknown qwen_backend={backend}")


def run_llm_with_json_retries(
    client: ChatClient,
    prompt: TextPAPrompt,
    max_retries: int = 5,
    sleep_s: float = 0.0,
) -> LLMResult:
    last_raw = ""
    for attempt in range(1, max_retries + 1):
        try:
            raw = client.complete(prompt)
            last_raw = raw
            obj = extract_first_json_object(raw)
            acc = coerce_float(obj.get("Accuracy"))
            flu = coerce_float(obj.get("Fluency"))
            reasoning = str(obj.get("Reasoning", ""))
            if not (1.0 <= acc <= 5.0 and 1.0 <= flu <= 5.0):
                raise ValueError(f"Scores outside 1-5 range: accuracy={acc}, fluency={flu}")
            return LLMResult(
                accuracy=acc,
                fluency=flu,
                reasoning=reasoning,
                raw_response=raw,
                valid_json=True,
                attempts=attempt,
            )
        except Exception as exc:
            if not last_raw:
                last_raw = f"{type(exc).__name__}: {exc}"
            if sleep_s:
                time.sleep(sleep_s)
            continue
    return LLMResult(
        accuracy=float("nan"),
        fluency=float("nan"),
        reasoning="",
        raw_response=last_raw,
        valid_json=False,
        attempts=max_retries,
    )
