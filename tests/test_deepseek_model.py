from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest
from harness import GOOD_MAIN_PY, WarmEnvironment, run

from articraft import api
from articraft.agent import Agent
from articraft.agent.provider import DeepSeekModel, create_model
from articraft.agent.record import read_conversation
from articraft.errors import ModelError
from articraft.settings import Settings


def response(
    text: str | None = None,
    *,
    reasoning: str | None = None,
    calls: list[dict[str, Any]] | None = None,
    finish: str = "stop",
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {
                    "content": text,
                    "tool_calls": calls or [],
                    "reasoning_content": reasoning,
                },
                "finish_reason": finish,
            }
        ],
        "usage": usage or {},
    }


def function_call(name: str, arguments: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments),
        },
    }


def model_with_responses(
    replies: Sequence[dict[str, Any] | httpx.Response | BaseException],
    **settings: Any,
) -> tuple[DeepSeekModel, list[httpx.Request]]:
    requests: list[httpx.Request] = []
    pending = list(replies)

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        item = pending.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item if isinstance(item, httpx.Response) else httpx.Response(200, json=item)

    config = Settings(provider="deepseek", deepseek_api_key="deepseek-test", **settings)
    return DeepSeekModel(
        config, client=httpx.AsyncClient(transport=httpx.MockTransport(handle))
    ), requests


def body(request: httpx.Request) -> dict[str, Any]:
    return json.loads(request.content)


def test_deepseek_sends_direct_authenticated_request_and_accounts_for_cached_tokens(capsys):
    model, requests = model_with_responses(
        [
            response(
                "done",
                usage={
                    "prompt_tokens": 100,
                    "prompt_cache_hit_tokens": 70,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                },
            )
        ],
        deepseek_request_timeout_seconds=45,
    )
    tools = [{"type": "function", "name": "compile", "description": "Compile", "parameters": {}}]
    result = run(model.query([{"role": "user", "content": "build"}], tools=tools))
    request = requests[0]
    assert request.url == "https://api.deepseek.com/chat/completions"
    assert request.headers["Authorization"] == "Bearer deepseek-test"
    assert "HTTP-Referer" not in request.headers
    assert request.extensions["timeout"]["read"] == 45
    assert body(request) == {
        "model": "deepseek-flash",
        "messages": [{"role": "user", "content": "build"}],
        "max_tokens": 32768,
        "stream": False,
        "thinking": {"type": "enabled"},
        "reasoning_effort": "high",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "compile",
                    "description": "Compile",
                    "parameters": {},
                    "strict": False,
                },
            }
        ],
    }
    assert result["token_usage"] == {
        "input_tokens": 100,
        "cached_input_tokens": 70,
        "output_tokens": 20,
        "total_tokens": 120,
    }
    assert result["cost"] == 0.0
    logs = capsys.readouterr().err
    assert "[deepseek] Sending request" in logs
    assert "[deepseek] Response received" in logs
    assert "deepseek-test" not in logs
    client = model._client
    assert client is not None
    run(model.close())
    assert client.is_closed
    run(model.close())


@pytest.mark.parametrize("effort", ["none", "low", "high", "max"])
def test_deepseek_reasoning_effort_and_output_limit(effort):
    model, requests = model_with_responses(
        [response("done")],
        deepseek_reasoning_effort=effort,
        deepseek_max_output_tokens=4096,
    )
    run(model.query([{"role": "user", "content": "build"}]))
    request = body(requests[0])
    assert request["thinking"] == {"type": "disabled" if effort == "none" else "enabled"}
    assert request.get("reasoning_effort") == (None if effort == "none" else effort)
    assert request["max_tokens"] == 4096
    run(model.close())


def test_deepseek_preserves_reasoning_and_tool_ids_through_real_agent_loop(tmp_path, capsys):
    replies = [
        response(reasoning="private reasoning one"),
        response(
            "Writing the object.",
            reasoning="private reasoning two",
            calls=[
                function_call("write", {"path": "main.py", "content": GOOD_MAIN_PY}, "write-1"),
            ],
        ),
        response(
            reasoning="private reasoning three", calls=[function_call("compile", {}, "compile-1")]
        ),
        response("done", reasoning="private reasoning four"),
    ]
    model, requests = model_with_responses(replies)
    agent = Agent(model, WarmEnvironment(output_dir=tmp_path), max_turns=4)
    result = run(agent.run("a box", run_id="deepseek-box"))
    assert result["status"] == "success"
    history = body(requests[-1])["messages"]
    assistants = [message for message in history if message["role"] == "assistant"]
    assert [message["reasoning_content"] for message in assistants] == [
        "private reasoning one",
        "private reasoning two",
        "private reasoning three",
    ]
    assert assistants[0]["content"] is None
    assert assistants[1]["tool_calls"][0]["id"] == "write-1"
    assert [message["tool_call_id"] for message in history if message["role"] == "tool"] == [
        "write-1",
        "compile-1",
    ]
    assert "private reasoning" not in capsys.readouterr().err
    recorded = read_conversation(tmp_path / "deepseek-box" / "conversation.jsonl")
    assert any(
        item.get("provider_content")
        == [
            {
                "type": "deepseek_reasoning",
                "reasoning_content": "private reasoning one",
            }
        ]
        for item in recorded
    )


def test_deepseek_keeps_multiple_tool_calls_and_results_in_order():
    calls = [
        function_call("read", {"path": "a.py"}, "a"),
        function_call("read", {"path": "b.py"}, "b"),
    ]
    model, requests = model_with_responses(
        [response(reasoning="inspect both", calls=calls), response("done")]
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": "inspect"}]
    first = run(model.query(messages))
    messages.extend(
        [
            {
                "role": "assistant",
                "content": first["text"],
                "tool_calls": first["tool_calls"],
                "provider_content": first["provider_content"],
            },
            {"type": "function_call_output", "call_id": "a", "output": {"text": "A"}},
            {"type": "function_call_output", "call_id": "b", "output": "B"},
        ]
    )
    run(model.query(messages))
    history = body(requests[-1])["messages"]
    assert history[1]["tool_calls"] == calls
    assert history[2:] == [
        {"role": "tool", "tool_call_id": "a", "content": '{"text": "A"}'},
        {"role": "tool", "tool_call_id": "b", "content": "B"},
    ]
    run(model.close())


def test_deepseek_flash_converts_reference_and_tool_images():
    model, requests = model_with_responses([response("done")])
    image = {"type": "input_image", "image_url": "data:image/png;base64,AAAA", "detail": "original"}
    run(
        model.query(
            [
                {"role": "user", "content": [{"type": "input_text", "text": "Build this"}, image]},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "view", "name": "view_image", "arguments": "{}"}],
                },
                {
                    "type": "function_call_output",
                    "call_id": "view",
                    "output": [
                        {"type": "input_text", "text": "preview"},
                        image,
                    ],
                },
            ]
        )
    )
    history = body(requests[0])["messages"]
    expected = {"type": "image_url", "image_url": {"url": image["image_url"], "detail": "original"}}
    assert history[0]["content"][-1] == expected
    assert history[2]["content"] == [{"type": "text", "text": "preview"}, expected]
    assert history[1]["reasoning_content"] == ""
    run(model.close())


def test_deepseek_pro_rejects_images_before_network_call():
    model, requests = model_with_responses([], deepseek_model="deepseek-v4-pro")
    assert not model.supports_images
    with pytest.raises(ModelError, match="does not support images"):
        run(
            model.query(
                [{"role": "user", "content": [{"type": "input_image", "image_url": "data:test"}]}]
            )
        )
    assert not requests
    run(model.close())


def test_deepseek_summary_disables_thinking_and_does_not_mutate_history():
    model, requests = model_with_responses([response("checkpoint"), response("done")])
    messages = [{"role": "user", "content": "build"}]
    result = run(model.summarize_context(messages, max_output_tokens=100))
    summary = body(requests[0])
    assert summary["thinking"] == {"type": "disabled"}
    assert summary["max_tokens"] == 100
    assert "tools" not in summary and "reasoning_effort" not in summary
    assert result["text"] == "checkpoint"
    run(model.query(messages))
    assert body(requests[1])["messages"] == messages
    run(model.close())


def test_deepseek_compaction_keeps_recent_tool_reasoning(tmp_path):
    model, requests = model_with_responses(
        [
            response(
                "old work " * 2000,
                reasoning="old reasoning",
                calls=[
                    function_call("write", {"path": "main.py", "content": GOOD_MAIN_PY}, "write"),
                ],
            ),
            response(
                "recent work " * 8000,
                reasoning="recent reasoning",
                calls=[function_call("compile", {}, "compile")],
            ),
            response("checkpoint"),
            response("done"),
        ],
        deepseek_context_window_tokens=40000,
    )
    agent = Agent(model, WarmEnvironment(output_dir=tmp_path), max_turns=3)
    result = run(agent.run("a box", run_id="deepseek-compact"))
    assert result["status"] == "success"
    assert len(requests) == 4
    assert "tools" not in body(requests[2])
    assert "<compaction-checkpoint>" in str(body(requests[3])["messages"])
    recent = [m for m in body(requests[3])["messages"] if m["role"] == "assistant"][-1]
    assert recent["reasoning_content"] == "recent reasoning"
    assert recent["tool_calls"][0]["id"] == "compile"


@pytest.mark.parametrize("status", [429, 500, 503])
def test_deepseek_retries_transient_http_errors_and_honors_retry_after(status, monkeypatch):
    delays = []

    async def sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(asyncio, "sleep", sleep)
    model, requests = model_with_responses(
        [
            httpx.Response(status, headers={"Retry-After": "0.25"}),
            response("done"),
        ]
    )
    assert run(model.query([{"role": "user", "content": "build"}]))["text"] == "done"
    assert len(requests) == 2
    assert delays == [0.25]
    run(model.close())


def test_deepseek_does_not_retry_authentication_errors():
    model, requests = model_with_responses(
        [httpx.Response(401, json={"error": {"message": "Invalid API key"}})]
    )
    with pytest.raises(ModelError, match=r"DeepSeek request failed.*401.*Invalid API key"):
        run(model.query([{"role": "user", "content": "build"}]))
    assert len(requests) == 1
    run(model.close())


def test_deepseek_wraps_transport_error_after_bounded_retries(monkeypatch):
    async def sleep(_delay):
        pass

    monkeypatch.setattr(asyncio, "sleep", sleep)
    model, requests = model_with_responses(
        [httpx.ReadTimeout("read failed")] * 2, deepseek_max_attempts=2
    )
    with pytest.raises(ModelError, match="DeepSeek request failed: ReadTimeout"):
        run(model.query([{"role": "user", "content": "build"}]))
    assert len(requests) == 2
    run(model.close())


@pytest.mark.parametrize(
    "reply,match",
    [
        (httpx.Response(200, text="not JSON"), "not valid JSON"),
        (httpx.Response(200, json=[]), "not an object"),
        ({}, "valid choice"),
        ({"choices": [{}]}, "assistant message"),
        (response(), "did not contain text, reasoning, or tool calls"),
        (response(reasoning=""), "did not contain text, reasoning, or tool calls"),
        (response(finish="length"), "truncated.*ARTICRAFT_DEEPSEEK_MAX_OUTPUT_TOKENS"),
        (response(calls=[{"id": "bad", "function": {}}]), "without an id or name"),
    ],
)
def test_deepseek_rejects_invalid_or_truncated_results(reply, match):
    model, _ = model_with_responses([reply])
    with pytest.raises(ModelError, match=match):
        run(model.query([{"role": "user", "content": "build"}]))
    run(model.close())


def test_deepseek_settings_dotenv_and_factory(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(Settings.model_config, "env_file", ".env")
    for name in ("DEEPSEEK_API_KEY", "ARTICRAFT_PROVIDER", "ARTICRAFT_DEEPSEEK_MODEL"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / ".env").write_text(
        "DEEPSEEK_API_KEY=deepseek-key\nARTICRAFT_PROVIDER=deepseek\n"
        "ARTICRAFT_DEEPSEEK_MODEL=deepseek-flash\nARTICRAFT_DEEPSEEK_REASONING_EFFORT=low\n"
    )
    settings = Settings()
    assert settings.selected_model == "deepseek-flash"
    assert settings.selected_reasoning_effort == "low"
    assert settings.deepseek_api_key == "deepseek-key"
    assert api._missing_provider_settings(settings) == []
    model = create_model(settings)
    assert isinstance(model, DeepSeekModel)
    assert model.context_window_tokens == 1_000_000
    run(model.close())


def test_deepseek_requires_its_own_credentials_and_model():
    settings = Settings(provider="deepseek", deepseek_api_key=" ", openai_api_key="openai-key")
    assert api._missing_provider_settings(settings) == ["DEEPSEEK_API_KEY"]
    with pytest.raises(ModelError, match="DEEPSEEK_API_KEY"):
        DeepSeekModel(settings)
    settings = settings.model_copy(
        update={"deepseek_api_key": "deepseek-key", "deepseek_model": " "}
    )
    assert api._missing_provider_settings(settings) == ["ARTICRAFT_DEEPSEEK_MODEL or --model"]
    with pytest.raises(ModelError, match="ARTICRAFT_DEEPSEEK_MODEL"):
        DeepSeekModel(settings)
