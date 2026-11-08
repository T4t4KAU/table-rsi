from types import SimpleNamespace

import pytest

from tablersi.model import Client


@pytest.mark.parametrize("effort", [None, "high"])
def test_teacher_thinking_is_explicit_and_does_not_change_scorer_default(
    monkeypatch, effort
):
    captured = {}

    def post(url, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            status_code=200,
            json=lambda: {
                "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 4},
                "model": "fixture",
            },
        )

    monkeypatch.setattr("tablersi.model.requests.post", post)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    client = Client(
        "https://example.invalid",
        "fixture",
        local=False,
        remote_effort=effort,
        request_timeout=600,
    )
    response = client.chat([{"role": "user", "content": "test"}], json_mode=True)
    assert response["usage"]["completion_tokens"] == 4
    assert captured["json"]["thinking"]["type"] == ("enabled" if effort else "disabled")
    assert captured["timeout"] == 600
    if effort:
        assert captured["json"]["reasoning_effort"] == effort
        assert "temperature" not in captured["json"]
    else:
        assert "reasoning_effort" not in captured["json"]
        assert captured["json"]["temperature"] == 0.1


def test_remote_thinking_cannot_silently_change_local_solver():
    with pytest.raises(ValueError, match="local client"):
        Client("https://example.invalid", "fixture", local=True, remote_effort="high")
