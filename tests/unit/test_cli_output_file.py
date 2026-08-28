from __future__ import annotations

from academic_literature_rag.cli import write_demo_output


def test_write_demo_output_prints_to_stdout_when_no_file_is_provided(
    capsys,
) -> None:
    write_demo_output(
        output="demo result",
        output_file=None,
    )

    captured = capsys.readouterr()

    assert captured.out == "demo result\n"
    assert captured.err == ""


def test_write_demo_output_writes_to_file(
    tmp_path,
    capsys,
) -> None:
    output_file = tmp_path / "demo_output.md"

    write_demo_output(
        output="# Demo Result\n",
        output_file=output_file,
    )

    captured = capsys.readouterr()

    assert captured.out == ""
    assert captured.err == ""
    assert output_file.read_text(encoding="utf-8") == "# Demo Result\n"


def test_write_demo_output_creates_parent_directories(
    tmp_path,
) -> None:
    output_file = tmp_path / "reports" / "nested" / "demo_output.json"

    write_demo_output(
        output='{"status": "ok"}',
        output_file=output_file,
    )

    assert output_file.exists()
    assert output_file.read_text(encoding="utf-8") == '{"status": "ok"}'


def test_write_demo_output_overwrites_existing_file(
    tmp_path,
) -> None:
    output_file = tmp_path / "demo_output.txt"
    output_file.write_text(
        "old content",
        encoding="utf-8",
    )

    write_demo_output(
        output="new content",
        output_file=output_file,
    )

    assert output_file.read_text(encoding="utf-8") == "new content"