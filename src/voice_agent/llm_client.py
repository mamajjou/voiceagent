"""Qwen3.8-27B via llama.cpp OpenAI-compatible API."""
from __future__ import annotations
import time
import json
from dataclasses import dataclass, field
from typing import AsyncGenerator, List, Dict, Optional, Callable

import httpx
from openai import OpenAI, AsyncOpenAI

@dataclass
class LLMConfig:
    host: str = "127.0.0.1"
    port: int = 8081
    model: str = "Qwen3.8-27B"
    context: int = 8192
    system_prompt: str = "You are a helpful assistant."
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    presence_penalty: float = 1.5
    max_tokens: int = 256
    enable_thinking: bool = False
    stream: bool = True

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

class LLMClient:
    def __init__(self, cfg: LLMConfig, mock: bool = False):
        self.cfg = cfg
        self.mock = mock
        self._client = OpenAI(base_url=cfg.base_url, api_key="sk-no-key") if not mock else None
        self._async_client = AsyncOpenAI(base_url=cfg.base_url, api_key="sk-no-key") if not mock else None
        self.history: List[Dict[str, str]] = [{"role": "system", "content": cfg.system_prompt}]

    def reset(self):
        self.history = [{"role": "system", "content": self.cfg.system_prompt}]

    def add_user(self, text: str):
        self.history.append({"role": "user", "content": text})

    def add_assistant(self, text: str):
        self.history.append({"role": "assistant", "content": text})

    def _chat_kwargs(self) -> dict:
        kwargs = {
            "model": self.cfg.model,
            "messages": self.history,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "max_tokens": self.cfg.max_tokens,
            "stream": True,
            "presence_penalty": self.cfg.presence_penalty,
        }
        # Qwen thinking disable via chat_template_kwargs
        if not self.cfg.enable_thinking:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
            # also try top_k via extra
            # llama.cpp may expect top_k at top level
            kwargs["extra_body"]["top_k"] = self.cfg.top_k
        else:
            kwargs["extra_body"] = {"top_k": self.cfg.top_k}
        return kwargs

    async def stream_chat(self, on_token: Callable[[str], None], prompt_text: Optional[str] = None) -> str:
        """Stream response for last user message (or prompt_text if provided)."""
        if prompt_text is not None:
            # temporary single-turn without mutating history
            messages = self.history[:-1] + [{"role": "user", "content": prompt_text}] if self.history and self.history[-1]["role"]=="user" else self.history + [{"role": "user", "content": prompt_text}]
        else:
            messages = self.history

        if self.mock:
            return await self._mock_stream(on_token, messages)

        kwargs = self._chat_kwargs()
        kwargs["messages"] = messages
        start = time.monotonic()
        first_token_time = None
        full = ""
        # Use httpx streaming directly to capture timing and avoid OpenAI lib quirks
        # But use openai client streaming for simplicity
        try:
            stream = await self._async_client.chat.completions.create(**kwargs)
            async for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    if first_token_time is None:
                        first_token_time = time.monotonic()
                    full += delta
                    on_token(delta)
                if chunk.choices[0].finish_reason:
                    break
        except Exception as e:
            # fallback to httpx
            print(f"[llm] openai stream failed {e}, trying httpx")
            full = await self._httpx_stream(messages, on_token)
        return full

    async def _httpx_stream(self, messages, on_token) -> str:
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "top_k": self.cfg.top_k,
            "presence_penalty": self.cfg.presence_penalty,
            "max_tokens": self.cfg.max_tokens,
            "stream": True,
            "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False} if not self.cfg.enable_thinking else {}
        }
        full = ""
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", f"{self.cfg.base_url}/chat/completions", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        j = json.loads(data)
                        delta = j["choices"][0]["delta"].get("content") or ""
                        if delta:
                            full += delta
                            on_token(delta)
                    except:
                        continue
        return full

    async def _mock_stream(self, on_token, messages) -> str:
        # simple echo mock
        last_user = messages[-1]["content"] if messages else "hello"
        # Generate a plausible response
        reply = f"Mock response to: '{last_user[:80]}' — this is a placeholder from the mock LLM. Enable the real llama-server for full Qwen reasoning."
        for tok in reply.split():
            on_token(tok + " ")
            await asyncio.sleep(0.02)
        return reply

    def stream_chat_sync(self, on_token, prompt_text=None) -> str:
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as ex:
                fut = ex.submit(asyncio.run, self.stream_chat(on_token, prompt_text))
                return fut.result()
        else:
            return loop.run_until_complete(self.stream_chat(on_token, prompt_text))

    async def check_health(self) -> bool:
        if self.mock:
            return True
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(f"http://{self.cfg.host}:{self.cfg.port}/health")
                if r.status_code == 200:
                    return True
                # try /v1/models
                r = await client.get(f"{self.cfg.base_url}/models")
                return r.status_code == 200
        except:
            return False

import asyncio
