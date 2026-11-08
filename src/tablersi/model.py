"""OpenAI-compatible clients with bounded calls and explicit token ledgers."""

import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import requests


def json_object(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("missing JSON object")
    return json.JSONDecoder().raw_decode(text[start:])[0]


@dataclass
class Budget:
    calls: int = 8
    input_tokens: int = 48000
    output_tokens: int = 8000
    db_calls: int = 24
    used_calls: int = 0
    used_input: int = 0
    used_output: int = 0
    used_db: int = 0
    unknown_usage: int = 0

    def allowed(self, prompt, max_output):
        # UTF-8 byte length is a conservative bound for byte-fallback tokenizers.
        return (
            self.used_calls < self.calls
            and self.used_input + len(prompt.encode()) <= self.input_tokens
            and self.used_output + max_output <= self.output_tokens
        )

    def charge(self, usage):
        if not usage:
            self.unknown_usage += 1
            return
        self.used_input += usage.get("prompt_tokens", 0)
        self.used_output += usage.get("completion_tokens", 0)

    def to_dict(self):
        return asdict(self)


class Client:
    def __init__(self, base_url, model, key_file=None, local=True, thinking_budget=0):
        self.base = base_url.rstrip("/")
        self.model = model
        self.local = local
        self.thinking_budget = thinking_budget
        self.key = (
            Path(key_file).read_text().strip()
            if key_file
            else ("" if local else os.environ.get("DEEPSEEK_API_KEY", ""))
        )

    def chat(
        self, messages, max_tokens=1200, seed=42, temperature=0.1, json_mode=False
    ):
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if self.local:
            payload.update(
                {
                    "seed": seed,
                    "chat_template_kwargs": {
                        "enable_thinking": self.thinking_budget > 0
                    },
                }
            )
            if self.thinking_budget:
                payload["thinking_token_budget"] = self.thinking_budget
        else:
            payload["thinking"] = {"type": "disabled"}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        start = time.monotonic()
        for attempt in range(3):
            r = requests.post(
                self.base + "/chat/completions",
                json=payload,
                headers=headers,
                timeout=180,
            )
            if r.status_code in [429, 500, 502, 503, 504] and attempt < 2:
                time.sleep(2**attempt)
                continue
            if r.status_code != 200:
                raise RuntimeError(f"Model HTTP {r.status_code}: {r.text[:300]}")
            data = r.json()
            m = data["choices"][0]["message"]
            return {
                "content": m.get("content") or "",
                "usage": data.get("usage", {}),
                "elapsed_s": time.monotonic() - start,
                "model": data.get("model"),
                "finish_reason": data["choices"][0].get("finish_reason"),
            }
        raise RuntimeError("model retry exhausted")
