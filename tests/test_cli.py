from types import SimpleNamespace

from research_assistant.cli import evaluate_check


def make_response(tool_name=None, arguments="{}", usage=(50, 10)):
    calls = None
    if tool_name:
        calls = [SimpleNamespace(function=SimpleNamespace(name=tool_name, arguments=arguments))]
    message = SimpleNamespace(tool_calls=calls, content=None)
    usage_obj = SimpleNamespace(prompt_tokens=usage[0], completion_tokens=usage[1]) if usage else None
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage_obj)


def test_check_passes_with_tool_call_and_usage():
    result = evaluate_check(make_response("count_papers", '{"topic": "RAG"}'))
    assert result["tool_calls"] and result["usage"]
    assert result["tool_args"] == {"topic": "RAG"}


def test_check_fails_without_tool_call():
    result = evaluate_check(make_response(None))
    assert not result["tool_calls"]


def test_check_fails_on_broken_arguments():
    result = evaluate_check(make_response("count_papers", "{not json"))
    assert not result["tool_calls"]


def test_check_fails_without_usage():
    result = evaluate_check(make_response("count_papers", '{"topic": "RAG"}', usage=None))
    assert not result["usage"]
