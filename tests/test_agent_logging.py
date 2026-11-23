from __future__ import annotations

from harness import WarmEnvironment, calls, run_scenario, text, tool_call


def test_terminal_tool_logs_identify_files_failures_and_call_ids(tmp_path, capsys) -> None:
    artifacts = run_scenario(
        "a box",
        [
            calls(tool_call("write", {"path": "notes.txt", "content": "UNLOGGED_FILE_CONTENT"})),
            calls(
                tool_call("read", {"path": "notes.txt"}, call_id="read-good"),
                tool_call("read", {"path": "missing.txt"}, call_id="read-bad"),
            ),
            calls(tool_call("compile")),
            text("Built the box."),
        ],
        env=WarmEnvironment(output_dir=tmp_path),
        max_turns=4,
    )

    assert artifacts.record.status == "success"
    logs = capsys.readouterr().err
    assert "Tool started: read notes.txt (read-good)" in logs
    assert "Tool started: read missing.txt (read-bad)" in logs
    assert "Tool finished: read notes.txt (read-good, completed," in logs
    assert "Tool finished: read missing.txt (read-bad, error," in logs
    assert "Tool error: read missing.txt (read-bad):" in logs
    assert "No such file or directory" in logs
    assert "Tool started: write notes.txt" in logs
    assert "UNLOGGED_FILE_CONTENT" not in logs
    assert "Tool started: compile workspace" in logs
    assert "feedback added to conversation" in logs


def test_malformed_tool_arguments_are_reported_without_breaking_logging(tmp_path, capsys) -> None:
    malformed = tool_call("read", call_id="bad-arguments")
    malformed["arguments"] = "{"
    artifacts = run_scenario(
        "a box",
        [calls(malformed), calls(tool_call("compile")), text("Built the box.")],
        env=WarmEnvironment(output_dir=tmp_path),
        max_turns=3,
    )

    assert artifacts.record.status == "success"
    logs = capsys.readouterr().err
    assert "Tool error: read (bad-arguments):" in logs
    assert "Expecting property name" in logs
