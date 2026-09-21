"""An environment-configured client for a vision-capable compatible API."""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from importlib.resources import files
from typing import Any

logger = logging.getLogger(__name__)


def load_prompt(name: str, domain: str = "cook") -> str:
    from .domains import get_domain

    if name not in {"segmentation.txt", "revisit.txt", "qa.txt"}:
        raise ValueError("Unknown pipeline prompt")
    spec = get_domain(domain)
    return files("storm_real_pipeline").joinpath("prompts", spec.name, name).read_text(encoding="utf-8")


@dataclass(frozen=True)
class Settings:
    base_url: str
    api_key: str = field(repr=False)
    model: str
    timeout: float = 3600.0
    max_retries: int = 5
    max_tokens: int = 32768

    @classmethod
    def from_env(cls) -> "Settings":
        required = ("STORM_BASE_URL", "STORM_API_KEY", "STORM_MODEL")
        missing = [name for name in required if not os.environ.get(name, "").strip()]
        if missing:
            raise ValueError("Missing environment variables: " + ", ".join(missing))
        try:
            timeout = float(os.environ.get("STORM_TIMEOUT", "3600"))
            retries = int(os.environ.get("STORM_MAX_RETRIES", "5"))
            tokens = int(os.environ.get("STORM_MAX_TOKENS", "32768"))
        except ValueError:
            raise ValueError("Invalid numeric API configuration") from None
        if not math.isfinite(timeout) or timeout <= 0 or retries < 1 or tokens < 1:
            raise ValueError("API timeout, retry count and token limit must be positive")
        return cls(
            base_url=os.environ["STORM_BASE_URL"].strip(),
            api_key=os.environ["STORM_API_KEY"].strip(),
            model=os.environ["STORM_MODEL"].strip(),
            timeout=timeout,
            max_retries=retries,
            max_tokens=tokens,
        )


def _format_content(contents: list[Any]) -> list[dict[str, Any]]:
    result = []
    for item in contents:
        if isinstance(item, str):
            result.append({"type": "text", "text": item})
        elif isinstance(item, dict) and item.get("type") == "image_base64":
            result.append({"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + item["data"]
            }})
        elif isinstance(item, dict) and item.get("type") == "image_url" and "url" in item:
            result.append({"type": "image_url", "image_url": {"url": item["url"]}})
        elif isinstance(item, dict) and "type" in item:
            result.append(item)
        else:
            raise TypeError("Unsupported multimodal message content")
    return result


class VLMClient:
    """Send timestamped frames without embedding credentials in source files."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.model = self.settings.model
        self._client = None

    def call(
        self,
        system_prompt: str,
        contents: list[Any],
        *,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        thinking: str | None = None,
        disable_thinking: bool = False,
    ) -> str:
        from openai import OpenAI

        if self._client is None:
            self._client = OpenAI(
                base_url=self.settings.base_url,
                api_key=self.settings.api_key,
                timeout=self.settings.timeout,
                max_retries=0,
            )
        if disable_thinking:
            thinking = "disabled"
        if thinking not in {None, "enabled", "disabled"}:
            raise ValueError("thinking must be enabled, disabled, or None")
        request = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _format_content(contents)},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens or self.settings.max_tokens,
        }
        # Only enable this provider extension when the caller asks for it.
        if thinking is not None:
            request["extra_body"] = {"thinking": {"type": thinking}}
        for attempt in range(1, self.settings.max_retries + 1):
            try:
                response = self._client.chat.completions.create(**request)
                text = response.choices[0].message.content
                if not isinstance(text, str) or not text.strip():
                    raise RuntimeError("Empty model response")
                return text
            except Exception as error:
                status = getattr(error, "status_code", None)
                transient = (
                    status in {408, 409, 429}
                    or isinstance(status, int) and status >= 500
                    or type(error).__name__ in {"APIConnectionError", "APITimeoutError"}
                    or isinstance(error, RuntimeError)
                )
                if not transient or attempt == self.settings.max_retries:
                    # Avoid returning SDK payloads, internal addresses, or request headers.
                    raise RuntimeError(
                        f"VLM request failed ({type(error).__name__}, status={status})"
                    ) from None
                delay = min(30.0, 2.0 ** attempt)
                logger.warning("Transient VLM error; retry %s/%s in %.0fs", attempt,
                               self.settings.max_retries, delay)
                time.sleep(delay)
        raise RuntimeError("VLM request failed")
