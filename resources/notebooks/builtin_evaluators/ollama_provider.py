"""Minimal Ollama provider for notebook experiments."""
from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, Optional

import aiohttp

logger = logging.getLogger(__name__)


class OllamaProviderException(Exception):
    """Raised when the provider encounters an error."""


class OllamaProviderResponse:
    """Simple response wrapper."""

    def __init__(self, response: str) -> None:
        self.response = response.strip()


class OllamaProvider:
    """Asynchronous interface for interacting with a local Ollama server."""

    def __init__(self, model: str, llm_address: str | None = None, port: int = 11434, seed: int = 0) -> None:
        if llm_address is None:
            llm_address = "localhost"
        self._model = model
        self._base_url = f"http://{llm_address}:{int(port) + int(seed)}/api"
        random.seed(seed)
        self._seed = random.randint(42, 1024)
        self._session = aiohttp.ClientSession(
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )
        self._validate_task = asyncio.create_task(self._validate_model())

    async def _validate_model(self) -> None:
        async with self._session.get(f"{self._base_url}/tags") as resp:
            models = await resp.json()
            model_names = [m["name"] for m in models.get("models", [])]
            target = self._model if ":" in self._model else f"{self._model}:latest"
            if target not in model_names:
                raise OllamaProviderException(f"Model {self._model} not found. Available: {model_names}")

    async def generate(self, prompt: str, system_prompt: str | None = None, raw: bool = False, **extra: Any) -> OllamaProviderResponse | None:
        await self._validate_task
        url = f"{self._base_url}/generate"
        options = {"seed": self._seed, **extra}
        payload = {
            "model": self._model,
            "prompt": prompt,
            "raw": raw,
            "options": options,
            "stream": False,
        }
        if system_prompt is not None:
            payload["system"] = system_prompt
        async with self._session.post(url, json=payload) as resp:
            data = await resp.json()
            if "error" in data:
                raise OllamaProviderException(data.get("error"))
            return OllamaProviderResponse(data.get("response", ""))

    async def chat(self, messages: list[dict[str, str]], system_prompt: str | None = None, **extra: Any) -> OllamaProviderResponse | None:
        await self._validate_task
        url = f"{self._base_url}/chat"
        options = {"seed": self._seed, **extra}
        payload = {
            "model": self._model,
            "messages": messages,
            "options": options,
            "stream": False,
        }
        if system_prompt is not None:
            payload.setdefault("messages", []).insert(0, {"role": "system", "content": system_prompt})
        async with self._session.post(url, json=payload) as resp:
            data = await resp.json()
            if "error" in data:
                raise OllamaProviderException(data.get("error"))
            msg = data.get("message", {}).get("content", "")
            return OllamaProviderResponse(msg)

    def sync_generate(self, prompt: str, **extra: Any) -> OllamaProviderResponse | None:
        return asyncio.run(self.generate(prompt, **extra))

    async def close(self) -> None:
        await self._session.close()
        try:
            self._validate_task.cancel()
        except Exception:
            pass
