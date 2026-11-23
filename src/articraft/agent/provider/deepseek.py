"""Direct DeepSeek Chat Completions, including reasoning across tool turns."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any

import httpx

from articraft.agent.provider._chat import (
    _arguments_text,
    _int,
    _retry_delay,
    _retryable_status,
    _tools,
)
from articraft.errors import ModelError
from articraft.settings import Settings, get_settings

_API_URL = "https://api.deepseek.com/chat/completions"
logger = logging.getLogger(__name__)


class DeepSeekModel:
    def __init__(
        self, settings: Settings | None = None, *, client: httpx.AsyncClient | None = None
    ) -> None:
        self.config = settings or get_settings()
        if not (self.config.deepseek_api_key or "").strip():
            raise ModelError("DeepSeek credentials are required. Set DEEPSEEK_API_KEY.")
        if not self.config.deepseek_model.strip():
            raise ModelError(
                "DeepSeek model is required. Pass --model or set ARTICRAFT_DEEPSEEK_MODEL."
            )
        self._client = client

    @property
    def supports_images(self) -> bool:
        return self.config.deepseek_model in {
            "deepseek-flash",
            "deepseek-v4-flash",
            "deepseek-v4-flash-vision-exp",
        }

    @property
    def context_window_tokens(self) -> int:
        return self.config.deepseek_context_window_tokens

    async def query(
        self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        request = self._request(messages, self.config.deepseek_max_output_tokens)
        converted_tools = _tools(tools or [])
        if converted_tools:
            request["tools"] = converted_tools
        effort = self.config.deepseek_reasoning_effort
        request["thinking"] = {"type": "disabled" if effort == "none" else "enabled"}
        if effort != "none":
            request["reasoning_effort"] = effort
        result = await self._complete(request)
        if (
            not result["text"]
            and not result["tool_calls"]
            and not any(item["reasoning_content"] for item in result["provider_content"])
        ):
            raise ModelError("DeepSeek response did not contain text, reasoning, or tool calls")
        return result

    async def summarize_context(
        self, messages: list[dict[str, Any]], *, max_output_tokens: int
    ) -> dict[str, Any]:
        request = self._request(
            messages, min(max_output_tokens, 8192, self.config.deepseek_max_output_tokens)
        )
        request["thinking"] = {"type": "disabled"}
        result = await self._complete(request)
        if not result["text"]:
            raise ModelError("DeepSeek summary response did not contain text")
        return {key: result[key] for key in ("text", "token_usage", "cost")}

    def _request(self, messages: list[dict[str, Any]], max_tokens: int) -> dict[str, Any]:
        return {
            "model": self.config.deepseek_model,
            "messages": _messages(messages, include_images=self.supports_images),
            "max_tokens": max_tokens,
            "stream": False,
        }

    async def _complete(self, request: dict[str, Any]) -> dict[str, Any]:
        if self._client is None:
            self._client = httpx.AsyncClient()
        for attempt in range(1, self.config.deepseek_max_attempts + 1):
            print(
                f"[deepseek] Sending request: model={request['model']}, attempt={attempt}, "
                f"messages={len(request['messages'])}, tools={len(request.get('tools', []))}, "
                f"thinking={request['thinking']['type']}",
                file=sys.stderr,
                flush=True,
            )
            try:
                response = await self._client.post(
                    _API_URL,
                    headers={
                        "Authorization": f"Bearer {(self.config.deepseek_api_key or '').strip()}",
                        "Content-Type": "application/json",
                    },
                    json=request,
                    timeout=self.config.deepseek_request_timeout_seconds,
                )
            except httpx.TransportError as exc:
                if attempt == self.config.deepseek_max_attempts:
                    raise ModelError(
                        f"DeepSeek request failed: {type(exc).__name__}: {exc}"
                    ) from exc
                logger.warning("DeepSeek transport error, retrying: %s", exc)
                await asyncio.sleep(_retry_delay(attempt))
                continue
            if (
                _retryable_status(response.status_code)
                and attempt < self.config.deepseek_max_attempts
            ):
                delay = _retry_delay(attempt, response.headers.get("Retry-After"))
                logger.warning("DeepSeek HTTP %s, retrying in %.2fs", response.status_code, delay)
                await asyncio.sleep(delay)
                continue
            try:
                payload = response.json()
            except ValueError as exc:
                raise ModelError(
                    f"DeepSeek response was not valid JSON (HTTP {response.status_code})"
                ) from exc
            if not isinstance(payload, dict):
                raise ModelError("DeepSeek response was not an object")
            if response.is_error or payload.get("error"):
                error = payload.get("error")
                detail = error.get("message") if isinstance(error, dict) else error
                raise ModelError(
                    f"DeepSeek request failed (HTTP {response.status_code}): {detail or 'request failed'}"
                )
            result = _result(payload)
            print(
                f"[deepseek] Response received: tools={len(result['tool_calls'])}, "
                f"tokens={result['token_usage'].get('total_tokens', 0)}; feedback ready for agent",
                file=sys.stderr,
                flush=True,
            )
            return result
        raise AssertionError("retry loop did not return or raise")

    async def close(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()


def _content(content: Any, *, include_images: bool) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return json.dumps(content)
    parts: list[dict[str, Any]] = []
    for item in content:
        if item.get("type") == "input_text":
            parts.append({"type": "text", "text": str(item.get("text") or "")})
        elif item.get("type") == "input_image":
            if not include_images:
                raise ModelError("The selected DeepSeek model does not support images")
            image = {"url": item["image_url"]}
            if "detail" in item:
                image["detail"] = item["detail"]
            parts.append({"type": "image_url", "image_url": image})
    return parts


def _messages(messages: list[dict[str, Any]], *, include_images: bool) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        if message.get("type") == "function_call_output":
            converted.append(
                {
                    "role": "tool",
                    "tool_call_id": message["call_id"],
                    "content": _content(message.get("output"), include_images=include_images),
                }
            )
        elif message.get("role") in {"system", "user"}:
            converted.append(
                {
                    "role": message["role"],
                    "content": _content(message["content"], include_images=include_images),
                }
            )
        elif message.get("role") == "assistant":
            assistant: dict[str, Any] = {
                "role": "assistant",
                "content": message.get("content") or None,
            }
            if message.get("tool_calls"):
                assistant["tool_calls"] = [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": _arguments_text(call.get("arguments")),
                        },
                    }
                    for call in message["tool_calls"]
                ]
            # DeepSeek requires every prior reasoning field on requests with tools.
            # Keep it in provider_content in the agent record, separate from visible text.
            assistant["reasoning_content"] = ""
            for item in message.get("provider_content") or []:
                if item.get("type") == "deepseek_reasoning":
                    assistant["reasoning_content"] = item["reasoning_content"]
            converted.append(assistant)
    return converted


def _result(payload: dict[str, Any]) -> dict[str, Any]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ModelError("DeepSeek response did not contain a valid choice")
    choice = choices[0]
    if choice.get("finish_reason") == "length":
        raise ModelError(
            "DeepSeek output was truncated. Increase ARTICRAFT_DEEPSEEK_MAX_OUTPUT_TOKENS."
        )
    if choice.get("finish_reason") in {"content_filter", "insufficient_system_resource", "error"}:
        raise ModelError(f"DeepSeek response failed: {choice['finish_reason']}")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ModelError("DeepSeek response choice did not contain an assistant message")
    calls = []
    for call in message.get("tool_calls") or []:
        function = call.get("function", {})
        if not call.get("id") or not function.get("name"):
            raise ModelError("DeepSeek returned a function call without an id or name")
        calls.append(
            {
                "id": call["id"],
                "name": function["name"],
                "arguments": _arguments_text(function.get("arguments")),
            }
        )
    reasoning = message.get("reasoning_content")
    provider_content = (
        [{"type": "deepseek_reasoning", "reasoning_content": reasoning}]
        if isinstance(reasoning, str)
        else []
    )
    usage = payload.get("usage") or {}
    prompt = _int(usage.get("prompt_tokens"))
    completion = _int(usage.get("completion_tokens"))
    return {
        "text": message.get("content") or "",
        "tool_calls": calls,
        "provider_content": provider_content,
        "token_usage": {
            "input_tokens": prompt,
            "cached_input_tokens": _int(usage.get("prompt_cache_hit_tokens")),
            "output_tokens": completion,
            "total_tokens": _int(usage.get("total_tokens")) or prompt + completion,
        },
        # DeepSeek does not document a per-request monetary charge in this response.
        "cost": 0.0,
        "response": payload,
    }


__all__ = ["DeepSeekModel"]
