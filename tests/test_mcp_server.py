import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from research_assistant.agent.tools import ToolContext, prompt_hash
from research_assistant.ingest.models import PaperRecord
from research_assistant.mcp_server import (
    PROTOCOL_VERSIONS,
    McpServer,
    build_server,
    library_tools,
    project_root,
    serve,
    write_tools,
)
from research_assistant.parse.passages import SectionOut, paragraph_passages
from research_assistant.parse.structure import Paragraph
from research_assistant.retrieval.search import Searcher
from research_assistant.store.db import Library


@pytest.fixture
def ctx():
    with Library(":memory:") as lib:
        topic = lib.get_or_create_topic("speculative decoding")
        texts = [
            ("Speculative decoding with draft models", 2024, "limitations",
             "Speedup drops when the draft model disagrees with the target model. Rare tokens are the main cause."),
            ("Tree attention for verification", 2025, "method",
             "Tree-based verification checks many candidate continuations in one forward pass. It needs more memory."),
        ]
        for title, year, kind, text in texts:
            paper_id, _ = lib.upsert_paper(PaperRecord(title=title, sources=["openalex"], year=year, language="en"))
            lib.link_topic(paper_id, topic)
            section = SectionOut(kind.title(), kind, 1)
            section.passages = paragraph_passages(Paragraph(text, [(0, 3)]))
            lib.save_structure(paper_id, [section], status="ok", parser="pdf", version=1)
        yield ToolContext(library=lib, searcher=Searcher(lib), allow_add=False)


@pytest.fixture
def server(ctx):
    return McpServer(library_tools(ctx))


def sentence_id(ctx, startswith):
    return ctx.library.conn.execute("SELECT id FROM sentences WHERE text LIKE ?", (startswith + "%",)).fetchone()[0]


def request(server, method, params=None, request_id=1):
    message = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        message["params"] = params
    return server.handle(message)


def call_tool(server, name, arguments):
    return request(server, "tools/call", {"name": name, "arguments": arguments})["result"]


# --- протокол ---

def test_initialize_echoes_known_version_and_gives_instructions(server):
    result = request(server, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                            "clientInfo": {"name": "claude-code"}})["result"]
    assert result["protocolVersion"] == "2025-06-18"
    assert result["capabilities"] == {"tools": {"listChanged": False}}
    assert result["serverInfo"]["name"] == "research-assistant"
    assert "cite" in result["instructions"] and "not instructions" in result["instructions"]


def test_initialize_with_unknown_version_answers_latest(server):
    result = request(server, "initialize", {"protocolVersion": "1999-01-01"})["result"]
    assert result["protocolVersion"] == PROTOCOL_VERSIONS[-1]


def test_notifications_get_no_response(server):
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_ping_and_unknown_method(server):
    assert request(server, "ping")["result"] == {}
    assert request(server, "resources/list")["error"]["code"] == -32601


def test_invalid_request(server):
    assert server.handle({"id": 3, "method": "ping"})["error"]["code"] == -32600
    assert server.handle(["not", "a", "dict"])["error"]["code"] == -32600


def test_default_tools_are_read_only(server):
    tools = request(server, "tools/list")["result"]["tools"]
    names = {t["name"] for t in tools}
    assert names == {"library_overview", "find_papers", "search_library", "read_passage", "list_entities",
                     "compare_papers", "cite"}
    for tool in tools:
        assert tool["inputSchema"]["type"] == "object"
        assert tool["annotations"]["readOnlyHint"] is True


def test_schemas_of_own_agent_stay_untouched(ctx):
    before = prompt_hash()
    tools = library_tools(ctx)
    for tool in tools:  # правка схем MCP не должна протекать в SCHEMAS агента
        tool.input_schema["properties"]["injected"] = {"type": "string"}
    assert prompt_hash() == before


# --- инструменты ---

def test_search_then_cite_gives_exact_quote(ctx, server):
    found = call_tool(server, "search_library", {"query": "speedup draft model"})
    assert found["isError"] is False
    sid = sentence_id(ctx, "Speedup drops")
    assert f"[s{sid}]" in found["content"][0]["text"]

    cited = call_tool(server, "cite", {"sentence_ids": [f"s{sid}"]})
    text = cited["content"][0]["text"]
    assert cited["isError"] is False
    assert "«Speedup drops when the draft model disagrees with the target model.»" in text
    assert "Speculative decoding with draft models (2024)" in text
    assert "not returned by search" not in text


def test_cite_text_matches_passage_slice(ctx, server):
    """Инвариант цитат: текст цитаты — ровно срез фрагмента по char_start:char_end."""
    sid = sentence_id(ctx, "Rare tokens")
    row = ctx.library.conn.execute(
        "SELECT s.text, p.text AS passage, s.char_start, s.char_end FROM sentences s "
        "JOIN passages p ON p.id = s.passage_id WHERE s.id = ?", (sid,)).fetchone()
    assert row["text"] == row["passage"][row["char_start"]:row["char_end"]]
    text = call_tool(server, "cite", {"sentence_ids": [sid]})["content"][0]["text"]
    assert f"«{row['text']}»" in text


def test_cite_warns_about_unseen_and_drops_unknown(ctx, server):
    sid = sentence_id(ctx, "It needs more memory")
    result = call_tool(server, "cite", {"sentence_ids": [f"s{sid}", "s999999", "nonsense"]})
    text = result["content"][0]["text"]
    assert result["isError"] is False
    assert "not returned by search" in text
    assert "s999999" in text and "nonsense" in text and "do not cite" in text


def test_cite_only_unknown_ids_is_error(server):
    result = call_tool(server, "cite", {"sentence_ids": ["s999999"]})
    assert result["isError"] is True
    assert "no such sentence ids" in result["content"][0]["text"]


def test_tool_error_is_observation_not_crash(server):
    result = call_tool(server, "search_library", {"query": "  "})
    assert result["isError"] is True and "query is empty" in result["content"][0]["text"]
    result = call_tool(server, "search_library", {"query": "speedup", "bogus": 1})
    assert result["isError"] is True and "bad arguments" in result["content"][0]["text"]
    result = call_tool(server, "drop_database", {})
    assert result["isError"] is True and "unknown tool" in result["content"][0]["text"]


def test_unexpected_exception_is_reported(server, capsys):
    def boom(args):
        raise RuntimeError("disk is on fire")

    server.tools["library_overview"].run = boom
    result = call_tool(server, "library_overview", {})
    assert result["isError"] is True and "RuntimeError: disk is on fire" in result["content"][0]["text"]
    assert "Traceback" in capsys.readouterr().err


def test_overview_and_read_passage(ctx, server):
    overview = call_tool(server, "library_overview", {})["content"][0]["text"]
    assert "speculative decoding: 2 papers" in overview
    sid = sentence_id(ctx, "Tree-based")
    passage = call_tool(server, "read_passage", {"passage_id": f"s{sid}"})["content"][0]["text"]
    assert "Tree attention for verification" in passage


def test_write_tools_need_candidates_from_search(ctx):
    record = PaperRecord(title="New paper", sources=["arxiv"], year=2026, arxiv_id="2601.00001")
    added = []
    ctx.sources = [SimpleNamespace(name="arxiv")]
    ctx.add_papers_fn = lambda records, topic: added.append((records, topic)) or "Added 1"
    server = McpServer(write_tools(ctx))
    result = call_tool(server, "add_papers", {"ids": ["arxiv:2601.00001"], "topic": "t"})
    assert result["isError"] is True and "unknown ids" in result["content"][0]["text"]

    ctx.candidates[record.external_id] = record
    result = call_tool(server, "add_papers", {"ids": [record.external_id], "topic": "t"})
    assert result["isError"] is False and added and added[0][1] == "t"


# --- сборка и транспорт ---

def settings_stub(tmp_path, missing=()):
    return SimpleNamespace(embed_model=None, runs_dir=tmp_path / "runs", missing_llm_vars=lambda: list(missing))


def test_build_server_levels(ctx, tmp_path):
    server, close = build_server(settings_stub(tmp_path), ctx.library)
    assert "add_papers" not in server.tools and "ask_agent" not in server.tools
    close()
    with pytest.raises(ValueError, match="allow-model"):
        build_server(settings_stub(tmp_path, missing=["MODEL"]), ctx.library, allow_model=True)


def test_serve_over_byte_streams_with_log(server, tmp_path):
    from research_assistant.agent.loop import RunLog

    server.log = RunLog(tmp_path / "mcp.jsonl")
    lines = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": "search_library", "arguments": {"query": "ускорение speedup"}}},
    ]
    stdin = io.BytesIO(("\n".join(json.dumps(m, ensure_ascii=False) for m in lines) + "\n\nnot json\n").encode("utf-8"))
    stdout = io.BytesIO()
    serve(server, stdin, stdout)
    responses = [json.loads(line) for line in stdout.getvalue().decode("utf-8").splitlines()]
    assert [r.get("id") for r in responses] == [1, 2, None]
    assert responses[0]["result"]["protocolVersion"] == "2025-03-26"
    assert "Found" in responses[1]["result"]["content"][0]["text"]
    assert responses[2]["error"]["code"] == -32700
    events = [json.loads(line)["event"] for line in (tmp_path / "mcp.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events == ["initialize", "tool"]


def test_batch_request(server):
    stdin = io.BytesIO((json.dumps([{"jsonrpc": "2.0", "id": 1, "method": "ping"},
                                    {"jsonrpc": "2.0", "method": "notifications/initialized"}]) + "\n").encode())
    stdout = io.BytesIO()
    serve(server, stdin, stdout)
    assert json.loads(stdout.getvalue()) == [{"jsonrpc": "2.0", "id": 1, "result": {}}]


def test_leading_bom_is_ignored(server):
    stdin = io.BytesIO(b"\xef\xbb\xbf" + json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}).encode() + b"\n")
    stdout = io.BytesIO()
    serve(server, stdin, stdout)
    assert json.loads(stdout.getvalue()) == {"jsonrpc": "2.0", "id": 1, "result": {}}


def test_project_root(tmp_path):
    assert project_root(str(tmp_path)) == tmp_path.resolve()
    (tmp_path / "data").mkdir()
    assert project_root(cwd=tmp_path) == tmp_path
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    assert (project_root(cwd=elsewhere) / "pyproject.toml").exists()  # папка, откуда установлен пакет


def test_cli_has_mcp_command(capsys):
    from research_assistant.cli import main

    with pytest.raises(SystemExit):
        main(["mcp", "--help"])
    out = capsys.readouterr().out
    assert "--allow-model" in out and "--allow-write" in out and "--project-dir" in out


def test_skill_file_is_valid():
    skill = Path(__file__).parents[1] / "skills" / "research-library" / "SKILL.md"
    text = skill.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    head = text.split("---\n")[1]
    fields = dict(line.split(":", 1) for line in head.strip().splitlines() if ":" in line and not line.startswith(" "))
    assert fields["name"].strip() == "research-library"
    assert 50 < len(fields["description"].strip()) <= 1024
    for tool in ("search_library", "cite", "read_passage"):
        assert tool in text
