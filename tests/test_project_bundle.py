import importlib.util
import json
import sqlite3
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "make_project_bundle.py"


@pytest.fixture(scope="module")
def pb():
    spec = importlib.util.spec_from_file_location("make_project_bundle", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Фальшивые ключи собираются из частей: в тексте файла не должно быть строк, похожих на ключ (их ловит сборщик).
FAKE_KEY = "sk" + "-test-0123456789abcdefXYZ"


def make_project(root: Path, secret: str = FAKE_KEY):
    (root / "src").mkdir(parents=True)
    (root / "src" / "app.py").write_text("print('ok')\n", encoding="utf-8")
    (root / "AGENT.md").write_text("# AGENT\n", encoding="utf-8")
    (root / ".env.example").write_text("INCEPTION_API_KEY=\nOPENAI_API_KEY=\nEMBED_API_KEY=lm-studio\n", encoding="utf-8")
    (root / ".env").write_text(f"LLM_PROVIDER=inception\nINCEPTION_API_KEY={secret}\nDATA_DIR=data\n", encoding="utf-8")
    (root / "data" / "vectors").mkdir(parents=True)
    (root / "data" / "fulltext").mkdir()
    (root / "data" / "fulltext" / "1.pdf").write_bytes(b"%PDF")
    (root / "data" / "vectors" / "text-embedding-bge-m3.npz").write_bytes(b"npz")
    conn = sqlite3.connect(root / "data" / "library.sqlite")
    conn.executescript("CREATE TABLE papers(id); CREATE TABLE passages(id); CREATE TABLE sentences(id);"
                       "CREATE TABLE topics(id, name); INSERT INTO papers VALUES (1); INSERT INTO topics VALUES (1, 't');")
    conn.commit()
    conn.close()
    return ["src/app.py", "AGENT.md", ".env.example", ".env", "data/fulltext/1.pdf", "runs/x.jsonl"]


def test_bundle_contents_and_manifest(pb, tmp_path):
    files = make_project(tmp_path / "p")
    zip_path = pb.build(tmp_path / "p", tmp_path / "out", files=[f for f in files if f != "runs/x.jsonl"],
                        version="v1")
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("research-assistant/MANIFEST.json"))
        for name, meta in manifest["files"].items():
            assert pb.sha256_bytes(archive.read(f"research-assistant/{name}")) == meta["sha256"]
    assert {"research-assistant/AGENT.md", "research-assistant/.env.example", "research-assistant/src/app.py",
            "research-assistant/data/library.sqlite", "research-assistant/data/vectors/text-embedding-bge-m3.npz"} <= names
    assert "research-assistant/.env" not in names and not any("fulltext" in n for n in names)
    assert manifest["library"]["papers"] == 1 and manifest["start_here"] == "AGENT.md"


def test_only_secret_like_env_values_are_tracked(pb, tmp_path):
    env = tmp_path / ".env"
    env.write_text(f"INCEPTION_API_KEY={FAKE_KEY}\nEMBED_MODEL=text-embedding-bge-m3\nEMBED_API_KEY=lm-studio\n"
                   "# OLD_API_KEY=commented-out-value\n", encoding="utf-8")
    assert pb.env_secrets(env) == [FAKE_KEY]


def test_refuses_when_env_value_leaks_into_a_file(pb, tmp_path):
    secret = "inc-secret-value-987654321"
    files = make_project(tmp_path / "p", secret=secret)
    (tmp_path / "p" / "src" / "notes.md").write_text(f"ключ: {secret}\n", encoding="utf-8")
    with pytest.raises(pb.BundleError, match="значение из .env"):
        pb.build(tmp_path / "p", tmp_path / "out", files=[*files, "src/notes.md"])


def test_key_patterns_and_empty_assignments(pb):
    assert pb.find_secrets("INCEPTION_API_KEY=\nOPENAI_API_KEY=\n") == []  # пустые значения — не ключи
    assert pb.find_secrets("EMBED_API_KEY=lm-studio") == []
    assert pb.find_secrets("OPENAI_API_KEY=real-looking-value") and pb.find_secrets("token " + FAKE_KEY)
    assert pb.forbidden([".env", ".env.local", ".env.example", "data/fulltext/a.pdf", "data/library.sqlite",
                         "data/vectors/x.npz", "runs/a.jsonl", "src/a.py"]) == [
        ".env", ".env.local", "data/fulltext/a.pdf", "runs/a.jsonl"]


def test_no_library_and_missing_agent_md(pb, tmp_path):
    files = make_project(tmp_path / "p")
    zip_path = pb.build(tmp_path / "p", tmp_path / "out", files=files, include_library=False)
    with zipfile.ZipFile(zip_path) as archive:
        assert not any(n.endswith(".sqlite") for n in archive.namelist())
    (tmp_path / "p" / "AGENT.md").unlink()
    with pytest.raises(pb.BundleError, match="AGENT.md"):
        pb.build(tmp_path / "p", tmp_path / "out2", files=[f for f in files if f != "AGENT.md"])
