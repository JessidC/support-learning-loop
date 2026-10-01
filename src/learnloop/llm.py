"""Two model clients behind the same Anthropic SDK.

The bot answers with DeepSeek (through its Anthropic-compatible endpoint). The judges run on a
different model family (Claude Haiku, or Gemini through a router), so the model that answers
never grades its own work.
Keys come only from the environment. Every call records tokens so a run can report its cost.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
from dataclasses import dataclass, field

import anthropic

# provider -> (base_url or None, key variable, default model, USD per million tokens in/out)
PROVIDERS = {
    "deepseek": ("https://api.deepseek.com/anthropic", "DEEPSEEK_API_KEY", "deepseek-flash", (0.30, 1.20)),
    "anthropic": (None, "ANTHROPIC_API_KEY", "claude-haiku-5-5", (0.10, 0.50)),
    # A router with an Anthropic-compatible endpoint; gives access to Gemini as an out-of-family judge.
    "orca": ("https://api.orcarouter.ai", "ORCAROUTER_API_KEY", "google/gemini-3.8-flash", (0.75, 3.75)),
}


def env_key(name: str) -> str | None:
    if os.environ.get(name):
        return os.environ[name]
    if sys.platform == "win32":  # a shell opened before the variable was set does not inherit it
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                return winreg.QueryValueEx(k, name)[0]
        except OSError:
            return None
    return None


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, i: int, o: int) -> None:
        with self.lock:
            self.calls += 1
            self.input_tokens += i
            self.output_tokens += o


class LLM:
    def __init__(self, provider: str, model: str | None = None):
        base_url, var, default_model, price = PROVIDERS[provider]
        key = env_key(var)
        if not key:
            raise RuntimeError(f"set {var} to use {provider}")
        self.provider, self.model, self.price = provider, model or default_model, price
        kwargs = {"api_key": key, "max_retries": 4, "timeout": 120.0}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = anthropic.Anthropic(**kwargs)
        self.usage = Usage()

    def cost(self) -> float:
        return (self.usage.input_tokens * self.price[0] + self.usage.output_tokens * self.price[1]) / 1e6

    def complete(self, system: str, user: str, max_tokens: int = 1500) -> str:
        params = {"model": self.model, "max_tokens": max_tokens, "system": system,
                  "messages": [{"role": "user", "content": user}]}
        # The current SDK no longer takes sampling parameters; third-party endpoints still read them.
        if self.provider == "deepseek":
            params["extra_body"] = {"temperature": 0, "thinking": {"type": "disabled"}}
        elif self.provider == "orca":
            params["extra_body"] = {"temperature": 0}
        else:
            # Haiku 5.5 thinks adaptively by default; a judge needs little of it.
            params["output_config"] = {"effort": "low"}
        r = self.client.messages.create(**params)
        self.usage.add(r.usage.input_tokens, r.usage.output_tokens)
        if r.stop_reason == "refusal":
            raise RuntimeError("model refused")
        return "".join(b.text for b in r.content if b.type == "text")

    def complete_json(self, system: str, user: str, max_tokens: int = 1500) -> dict:
        return parse_json(self.complete(system, user, max_tokens))


def parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON in model output: {text[:120]!r}")
    return json.loads(m.group(0))
