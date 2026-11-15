"""Minimal Ollama chat client (HTTP API, standard library only).

``OllamaClient.chat`` sends a non-streaming ``/api/chat`` request and returns the assistant's
message text plus Ollama's timing counters. ``format`` may be a JSON schema (structured output).
Anything with the same ``chat`` signature can stand in for it (tests use a fake).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ChatResult:
    content: str
    stats: dict[str, Any] = field(default_factory=dict)  # Ollama counters (durations in ns, token counts)


class ChatModel(Protocol):
    def chat(self, messages: list[dict[str, str]], format: dict[str, Any] | str | None = None) -> ChatResult:
        ...


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, model: str, url: str = "http://localhost:11434", options: dict[str, Any] | None = None,
                 timeout_s: float = 600.0, keep_alive: str = "15m") -> None:
        self.model = model
        self.url = url.rstrip("/")
        self.options = options or {}
        self.timeout_s = timeout_s
        self.keep_alive = keep_alive

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(self.url + path, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise OllamaError(f"Ollama {path} failed: HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')}") from exc
        except urllib.error.URLError as exc:
            raise OllamaError(f"cannot reach Ollama at {self.url}: {exc.reason}") from exc

    def chat(self, messages: list[dict[str, str]], format: dict[str, Any] | str | None = None) -> ChatResult:
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "stream": False,
                                   "options": self.options, "keep_alive": self.keep_alive}
        if format is not None:
            payload["format"] = format
        data = self._post("/api/chat", payload)
        stats = {k: data[k] for k in ("total_duration", "load_duration", "prompt_eval_count",
                                      "prompt_eval_duration", "eval_count", "eval_duration") if k in data}
        return ChatResult(data["message"]["content"], stats)

    def model_info(self) -> dict[str, Any]:
        """Name, digest and details of the model (for provenance)."""
        data = self._post("/api/show", {"model": self.model})
        return {"model": self.model, "details": data.get("details", {}),
                "modified_at": data.get("modified_at"), "digest": _digest(self.url, self.model)}


def _digest(url: str, model: str) -> str | None:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/api/tags", timeout=10) as resp:
            tags = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError):
        return None
    for m in tags.get("models", []):
        if m.get("name") in (model, f"{model}:latest") or m.get("model") in (model, f"{model}:latest"):
            return m.get("digest")
    return None
