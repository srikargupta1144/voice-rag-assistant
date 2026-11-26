"""
File: tests/unit/test_eval_scripts.py

Description:
    Unit tests for the evaluation scripts: the CI quality gate (eval.gate), the README
    report writer (eval.report), the retrieval evaluation (eval.retrieval_eval) and the
    answer-quality evaluation (eval.answer_eval). Every test uses temporary result files,
    a temporary README, a fake retriever, a fake judge and fake settings, so no model is
    loaded and the real README.md and eval/results folder are never touched.

Libraries used:
    __future__ - standard library; postponed evaluation of type annotations
    json - standard library; writes and reads result files
    sys - standard library; sets command-line arguments for main()
    pytest - third party; fixtures, parametrisation and expected failures
    common.types - project; Chunk and Hit records used by the fake retriever
    eval.answer_eval - project; judge scoring helpers, summarize, run and main under test
    eval.dataset - project; the Question record
    eval.gate - project; check and main under test
    eval.report - project; Markdown section builders and main under test
    eval.retrieval_eval - project; evaluate and main under test
    rag.citations - project; the Citation record used by the fake pipeline
    rag.pipeline - project; the AnswerResult record returned by the fake pipeline

Created by: Srikar Gupta
Created date: 2026-11-26
Updated by: Srikar Gupta
Updated date: 2026-11-26

Change log:
    2026-11-26  Srikar Gupta  First version.
"""

from __future__ import annotations

import json
import sys

import pytest

import eval.answer_eval as ae
import eval.gate as gate_mod
import eval.report as report_mod
import eval.retrieval_eval as re_mod
from common.types import Chunk, Hit
from eval.dataset import Question
from rag.citations import Citation
from rag.pipeline import AnswerResult

# ---------------------------------------------------------------------------------------
# eval.gate
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("results", "baseline", "tolerance", "ok", "text"),
    [
        ({"m": 0.9}, {"m": 0.9}, 0.0, True, "gate passed: m 0.9 >= baseline 0.9"),
        ({"m": 0.95}, {"m": 0.9}, 0.0, True, "gate passed"),
        ({"m": 0.89}, {"m": 0.9}, 0.0, False, "gate FAILED: m 0.89 < baseline 0.9"),
        ({"m": 0.85}, {"m": 0.9}, 0.05, True, "gate passed"),
        ({"m": 0.8}, {"m": 0.9}, 0.05, False, "(tolerance 0.05)"),
        ({"m": 0.0}, {"m": 0.0}, 0.0, True, "gate passed"),
        ({}, {"m": 0.9}, 0.0, False, "gate FAILED: results have no m"),
        ({"m": None}, {"m": 0.9}, 0.0, False, "results have no m"),
        ({"m": 0.1}, {}, 0.0, True, "gate: baseline has no m; passing"),
        ({}, {"m": None}, 0.0, True, "baseline has no m"),
    ],
)
def test_gate_check_cases(results, baseline, tolerance, ok, text):
    """
    Function name: test_gate_check_cases
    Purpose: Check the pass or fail result and the message of the gate for equal, higher and
        lower values, a tolerance, zero values and missing values on either side.
    Parameters:
        results (dict): Results document.
        baseline (dict): Baseline document.
        tolerance (float): Allowed drop below the baseline.
        ok (bool): Expected outcome.
        text (str): Text that the message must contain.
    Returns:
        None: The test passes or fails through assertions.
    """
    passed, msg = gate_mod.check(results, baseline, "m", tolerance)
    assert passed is ok
    assert text in msg


def _write_json(path, data):
    """
    Function name: _write_json
    Purpose: Write a dictionary to a JSON file.
    Parameters:
        path (Path): Target file.
        data (dict): Document to write.
    Returns:
        None: The file is written.
    """
    path.write_text(json.dumps(data), encoding="utf-8")


def _run_gate(monkeypatch, capsys, *args):
    """
    Function name: _run_gate
    Purpose: Run eval.gate.main with the given command-line arguments and capture its output.
    Parameters:
        monkeypatch (pytest.MonkeyPatch): Sets sys.argv.
        capsys (pytest.CaptureFixture): Captures the printed message.
        args (str): Command-line arguments after the program name.
    Returns:
        tuple[int, str]: The exit code and the printed text.
    """
    monkeypatch.setattr(sys, "argv", ["gate", *args])
    code = gate_mod.main()
    return code, capsys.readouterr().out


def test_gate_main_passes_with_exit_code_zero(tmp_path, monkeypatch, capsys):
    """
    Function name: test_gate_main_passes_with_exit_code_zero
    Purpose: Check that main returns 0 and prints the pass message when the results meet the
        baseline.
    Parameters:
        tmp_path (Path): Temporary directory.
        monkeypatch (pytest.MonkeyPatch): Sets sys.argv.
        capsys (pytest.CaptureFixture): Captures the printed message.
    Returns:
        None: The test passes or fails through assertions.
    """
    _write_json(tmp_path / "r.json", {"recall_at_5": 1.0})
    _write_json(tmp_path / "b.json", {"recall_at_5": 0.9})
    code, out = _run_gate(
        monkeypatch,
        capsys,
        "--results",
        str(tmp_path / "r.json"),
        "--baseline",
        str(tmp_path / "b.json"),
    )
    assert code == 0
    assert out.strip() == "gate passed: recall_at_5 1.0 >= baseline 0.9 (tolerance 0.0)"


def test_gate_main_fails_with_exit_code_one(tmp_path, monkeypatch, capsys):
    """
    Function name: test_gate_main_fails_with_exit_code_one
    Purpose: Check that main returns 1 and prints the failure message when the metric drops
        below the baseline, and that a tolerance can allow the drop.
    Parameters:
        tmp_path (Path): Temporary directory.
        monkeypatch (pytest.MonkeyPatch): Sets sys.argv.
        capsys (pytest.CaptureFixture): Captures the printed message.
    Returns:
        None: The test passes or fails through assertions.
    """
    _write_json(tmp_path / "r.json", {"recall_at_5": 0.8})
    _write_json(tmp_path / "b.json", {"recall_at_5": 0.9})
    files = ["--results", str(tmp_path / "r.json"), "--baseline", str(tmp_path / "b.json")]
    code, out = _run_gate(monkeypatch, capsys, *files)
    assert code == 1 and "gate FAILED: recall_at_5 0.8 < baseline 0.9" in out
    code, out = _run_gate(monkeypatch, capsys, *files, "--tolerance", "0.2")
    assert code == 0 and "gate passed" in out


def test_gate_main_missing_results_file_fails(tmp_path, monkeypatch, capsys):
    """
    Function name: test_gate_main_missing_results_file_fails
    Purpose: Check that a missing results file gives exit code 1 and a message naming the
        file, without reading the baseline.
    Parameters:
        tmp_path (Path): Temporary directory.
        monkeypatch (pytest.MonkeyPatch): Sets sys.argv.
        capsys (pytest.CaptureFixture): Captures the printed message.
    Returns:
        None: The test passes or fails through assertions.
    """
    missing = tmp_path / "none.json"
    code, out = _run_gate(
        monkeypatch,
        capsys,
        "--results",
        str(missing),
        "--baseline",
        str(tmp_path / "also-missing.json"),
    )
    assert code == 1
    assert f"gate FAILED: results file {missing} missing" in out


def test_gate_main_other_metric_and_missing_metric(tmp_path, monkeypatch, capsys):
    """
    Function name: test_gate_main_other_metric_and_missing_metric
    Purpose: Check that --metric selects another key, and that a metric absent from the
        results fails the gate.
    Parameters:
        tmp_path (Path): Temporary directory.
        monkeypatch (pytest.MonkeyPatch): Sets sys.argv.
        capsys (pytest.CaptureFixture): Captures the printed message.
    Returns:
        None: The test passes or fails through assertions.
    """
    _write_json(tmp_path / "r.json", {"mrr_at_10": 0.7})
    _write_json(tmp_path / "b.json", {"mrr_at_10": 0.7, "recall_at_5": 1.0})
    files = ["--results", str(tmp_path / "r.json"), "--baseline", str(tmp_path / "b.json")]
    code, out = _run_gate(monkeypatch, capsys, *files, "--metric", "mrr_at_10")
    assert code == 0 and "mrr_at_10" in out
    code, out = _run_gate(monkeypatch, capsys, *files)
    assert code == 1 and "results have no recall_at_5" in out


def test_gate_main_passes_when_baseline_lacks_metric(tmp_path, monkeypatch, capsys):
    """
    Function name: test_gate_main_passes_when_baseline_lacks_metric
    Purpose: Check that a baseline without the metric makes the gate pass.
    Parameters:
        tmp_path (Path): Temporary directory.
        monkeypatch (pytest.MonkeyPatch): Sets sys.argv.
        capsys (pytest.CaptureFixture): Captures the printed message.
    Returns:
        None: The test passes or fails through assertions.
    """
    _write_json(tmp_path / "r.json", {"recall_at_5": 0.1})
    _write_json(tmp_path / "b.json", {})
    code, out = _run_gate(
        monkeypatch,
        capsys,
        "--results",
        str(tmp_path / "r.json"),
        "--baseline",
        str(tmp_path / "b.json"),
    )
    assert code == 0 and "baseline has no recall_at_5" in out


# ---------------------------------------------------------------------------------------
# eval.report
# ---------------------------------------------------------------------------------------


@pytest.fixture
def report_paths(tmp_path, monkeypatch):
    """
    Function name: report_paths
    Purpose: Fixture that points eval.report at a temporary results folder and README.
    Parameters:
        tmp_path (Path): Temporary directory.
        monkeypatch (pytest.MonkeyPatch): Replaces RESULTS and README in eval.report.
    Returns:
        tuple[Path, Path]: The temporary results folder and the temporary README path.
    """
    results = tmp_path / "results"
    results.mkdir()
    readme = tmp_path / "README.md"
    monkeypatch.setattr(report_mod, "RESULTS", results)
    monkeypatch.setattr(report_mod, "README", readme)
    return results, readme


RETRIEVAL = {
    "questions": 3,
    "answerable": 2,
    "chunks": 4,
    "modes": {
        "dense": {"recall_at_5": 0.5, "mrr_at_10": 0.25, "median_latency_ms": 12.4},
        "full": {"recall_at_5": 1.0, "mrr_at_10": 0.9, "median_latency_ms": 2500.0},
        "other": {"recall_at_5": 0.0, "mrr_at_10": 0.0, "median_latency_ms": 0.2},
    },
    "not_found_gate": {
        "threshold": 1.5,
        "accuracy": 0.6667,
        "answerable_min_score": 1.0,
        "unanswerable_max_score": 0.5,
    },
}


def _answers_doc(model="m1", few_shot=True, temp=0.1):
    """
    Function name: _answers_doc
    Purpose: Build a small answer-quality results document for the report tests.
    Parameters:
        model (str): Answering model name.
        few_shot (bool): Whether few-shot examples were used.
        temp (float): Answer temperature.
    Returns:
        dict: The results document.
    """
    return {
        "llm_model": model,
        "judge_model": "judge-x",
        "few_shot": few_shot,
        "temperature": temp,
        "questions": 9,
        "summary": {
            "faithfulness": 0.9,
            "answer_relevancy": 0.8,
            "answered_rate": 1.0,
            "unanswerable_declined": 0.5,
            "citation_correct": 0.75,
            "median_llm_first_token_ms": 800.0,
            "median_total_ms": 4200.0,
        },
    }


@pytest.mark.parametrize(
    ("value", "digits", "expected"),
    [
        (None, 3, "-"),
        (0.5, 3, "0.500"),
        (1.0, 3, "1.000"),
        (0.66666, 2, "0.67"),
        (0.12345, 1, "0.1"),
        (7, 3, "7"),
        ("text", 3, "text"),
        (0, 3, "0"),
    ],
)
def test_report_fmt(value, digits, expected):
    """
    Function name: test_report_fmt
    Purpose: Check that floats get fixed decimals, None becomes a dash and other values are
        written with str().
    Parameters:
        value (object): Value to format.
        digits (int): Decimal places for floats.
        expected (str): Expected text.
    Returns:
        None: The test passes or fails through assertions.
    """
    assert report_mod._fmt(value, digits) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "-"),
        (0, "<1 ms"),
        (0.99, "<1 ms"),
        (1, "1 ms"),
        (999.4, "999 ms"),
        (1000, "1.0 s"),
        (13579.0, "13.6 s"),
    ],
)
def test_report_ms(value, expected):
    """
    Function name: test_report_ms
    Purpose: Check the millisecond formatting at the boundaries 1 ms and 1000 ms.
    Parameters:
        value (float | None): Duration in milliseconds.
        expected (str): Expected text.
    Returns:
        None: The test passes or fails through assertions.
    """
    assert report_mod._ms(value) == expected


def test_report_load_reads_existing_and_returns_none_for_missing(report_paths):
    """
    Function name: test_report_load_reads_existing_and_returns_none_for_missing
    Purpose: Check that _load parses an existing result file and gives None for a missing one.
    Parameters:
        report_paths (tuple[Path, Path]): Temporary results folder and README.
    Returns:
        None: The test passes or fails through assertions.
    """
    results, _ = report_paths
    _write_json(results / "retrieval.json", {"a": 1})
    assert report_mod._load("retrieval.json") == {"a": 1}
    assert report_mod._load("e2e.json") is None


def test_retrieval_section_rows_and_gate_sentence():
    """
    Function name: test_retrieval_section_rows_and_gate_sentence
    Purpose: Check the retrieval table: heading line, friendly mode names, unknown mode names
        kept, formatted numbers and the not-found gate sentence.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    lines = report_mod.retrieval_section(RETRIEVAL)
    assert lines[0] == "Retrieval over 2 answerable questions, 4 chunks (no LLM involved):"
    assert "| Dense only (FAISS, bge-small) | 0.500 | 0.250 | 12 ms |" in lines
    assert "| Hybrid + cross-encoder re-rank | 1.000 | 0.900 | 2.5 s |" in lines
    assert "| other | 0.000 | 0.000 | <1 ms |" in lines
    assert lines[-1].startswith("Not-found gate on the re-ranker score (threshold 1.5): 0.667")
    assert "of all 3 questions" in lines[-1]
    assert "lowest answerable score 1.0" in lines[-1]
    assert "highest unanswerable score 0.5" in lines[-1]


def test_retrieval_section_without_gate_and_when_missing():
    """
    Function name: test_retrieval_section_without_gate_and_when_missing
    Purpose: Check that the gate sentence is left out when the results have no gate, and that
        missing or empty results give the not-generated note.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    without = {k: v for k, v in RETRIEVAL.items() if k != "not_found_gate"}
    lines = report_mod.retrieval_section(without)
    assert not any("Not-found gate" in ln for ln in lines)
    for empty in (None, {}):
        out = report_mod.retrieval_section(empty)
        assert len(out) == 1 and "not generated yet" in out[0]


def test_answers_section_rows_sorted_by_file_and_judge_note(report_paths):
    """
    Function name: test_answers_section_rows_sorted_by_file_and_judge_note
    Purpose: Check that every answers*.json file gives one row in file name order with the
        few-shot label, temperature and formatted metrics, and that the judge note uses the
        first file.
    Parameters:
        report_paths (tuple[Path, Path]): Temporary results folder and README.
    Returns:
        None: The test passes or fails through assertions.
    """
    results, _ = report_paths
    _write_json(results / "answers_zero_shot.json", _answers_doc("m2", False, 0.0))
    _write_json(results / "answers.json", _answers_doc("m1", True, 0.1))
    _write_json(results / "retrieval.json", RETRIEVAL)
    lines = report_mod.answers_section()
    rows = [ln for ln in lines if ln.startswith("| m")]
    assert len(rows) == 2
    assert rows[0] == (
        "| m1, few-shot, temp 0.1 | 0.900 | 0.800 | 1.000 | 0.500 | 0.750 | 800 ms | 4.2 s |"
    )
    assert rows[1].startswith("| m2, zero-shot, temp 0.0 |")
    assert lines[-1] == (
        "Judge: judge-x through Ollama, over 9 questions. "
        "Faithfulness and relevancy are averaged over answered, answerable questions."
    )


def test_answers_section_without_files(report_paths):
    """
    Function name: test_answers_section_without_files
    Purpose: Check that an empty results folder gives the not-generated note.
    Parameters:
        report_paths (tuple[Path, Path]): Temporary results folder and README.
    Returns:
        None: The test passes or fails through assertions.
    """
    out = report_mod.answers_section()
    assert len(out) == 1 and "not generated yet" in out[0]


E2E = {
    "audio": {
        "wer": 0.0833,
        "server_timings": {
            "stt_ms": 4506.2,
            "rewrite_ms": 3877.3,
            "retrieval_ms": 411.4,
            "llm_first_token_ms": 11844.7,
            "llm_ms": 4047.1,
            "tts_ms": 632.1,
            "total_ms": 13516.7,
        },
    },
    "text": {"server_timings": {"retrieval_ms": 0.4, "llm_ms": 900.0, "total_ms": 1000.0}},
    "websocket": {
        "client_first_audio_ms": 13579.0,
        "client_total_ms": 14000.0,
        "sentences_streamed": 2,
    },
}


def test_latency_section_table_and_websocket_sentence():
    """
    Function name: test_latency_section_table_and_websocket_sentence
    Purpose: Check the latency table: audio and text columns, dashes for stages that exist
        in only one column, and the streaming sentence with the word error rate.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    lines = report_mod.latency_section(E2E)
    assert "| Speech-to-text (faster-whisper small, int8) | 4.5 s | - |" in lines
    assert "| Retrieval (hybrid + re-rank) | 411 ms | <1 ms |" in lines
    assert "| Query rewrite (LLM, only for casual speech) | 3.9 s | - |" in lines
    assert "| Total server time | 13.5 s | 1.0 s |" in lines
    assert lines[-1].startswith("Streaming (/ws/voice, same recording): first reply audio after")
    assert "13.6 s measured at the client, full reply after 14.0 s; 2 sentence(s)" in lines[-1]
    assert "Word error rate of the transcript on the Piper-generated question: 0.083." in lines[-1]


def test_latency_section_missing_results():
    """
    Function name: test_latency_section_missing_results
    Purpose: Check that missing end-to-end results give the not-generated note.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    out = report_mod.latency_section(None)
    assert len(out) == 1 and "not generated yet" in out[0]


def test_render_markdown_with_no_results_has_three_notes(report_paths):
    """
    Function name: test_render_markdown_with_no_results_has_three_notes
    Purpose: Check that with an empty results folder the Markdown has the three headings and
        three not-generated notes.
    Parameters:
        report_paths (tuple[Path, Path]): Temporary results folder and README.
    Returns:
        None: The test passes or fails through assertions.
    """
    md = report_mod.render_markdown()
    assert md.count("not generated yet") == 3
    assert [ln for ln in md.splitlines() if ln.startswith("### ")] == [
        "### Retrieval",
        "### Answers",
        "### Latency (one run on the development laptop, CPU only)",
    ]


def test_report_main_replaces_only_the_marked_block(report_paths, capsys):
    """
    Function name: test_report_main_replaces_only_the_marked_block
    Purpose: Check that main rewrites the text between the markers in the temporary README,
        keeps the text before and after, prints the Markdown, and gives the same file when
        run a second time.
    Parameters:
        report_paths (tuple[Path, Path]): Temporary results folder and README.
        capsys (pytest.CaptureFixture): Captures the printed Markdown.
    Returns:
        None: The test passes or fails through assertions.
    """
    results, readme = report_paths
    _write_json(results / "retrieval.json", RETRIEVAL)
    _write_json(results / "answers.json", _answers_doc())
    _write_json(results / "e2e.json", E2E)
    readme.write_text(
        f"# Title\n\nIntro.\n\n{report_mod.START}\nOLD TABLE\n{report_mod.END}\n\nOutro.\n",
        encoding="utf-8",
    )
    report_mod.main()
    printed = capsys.readouterr().out
    text = readme.read_text(encoding="utf-8")
    assert text.startswith("# Title\n\nIntro.\n\n" + report_mod.START + "\n")
    assert text.endswith(report_mod.END + "\n\nOutro.\n")
    assert "OLD TABLE" not in text
    assert "| Dense only (FAISS, bge-small)" in text
    assert printed.strip() == report_mod.render_markdown().strip()
    report_mod.main()
    assert readme.read_text(encoding="utf-8") == text


def test_report_main_keeps_backslashes_in_results(report_paths):
    """
    Function name: test_report_main_keeps_backslashes_in_results
    Purpose: Check that backslashes and group references in the rendered text are written
        literally rather than treated as regular expression replacement syntax.
    Parameters:
        report_paths (tuple[Path, Path]): Temporary results folder and README.
    Returns:
        None: The test passes or fails through assertions.
    """
    results, readme = report_paths
    _write_json(results / "answers.json", _answers_doc(model="C:\\tmp\\1 \\g<0>"))
    readme.write_text(f"{report_mod.START}\nx\n{report_mod.END}\n", encoding="utf-8")
    report_mod.main()
    assert "C:\\tmp\\1 \\g<0>, few-shot" in readme.read_text(encoding="utf-8")


def test_report_main_without_markers_exits_and_leaves_readme(report_paths):
    """
    Function name: test_report_main_without_markers_exits_and_leaves_readme
    Purpose: Check that a README without the markers makes main exit with a message and
        leaves the file unchanged.
    Parameters:
        report_paths (tuple[Path, Path]): Temporary results folder and README.
    Returns:
        None: The test passes or fails through assertions.
    """
    _, readme = report_paths
    readme.write_text("# No markers here\n", encoding="utf-8")
    with pytest.raises(SystemExit) as info:
        report_mod.main()
    assert "README has no <!-- RESULTS:START -->" in str(info.value)
    assert readme.read_text(encoding="utf-8") == "# No markers here\n"


# ---------------------------------------------------------------------------------------
# eval.retrieval_eval
# ---------------------------------------------------------------------------------------

FEE = Chunk("faq.md:0:0", "Monthly fee is 8 dollars.", "faq.md", None, "Fee")
WIRE = Chunk("faq.md:0:1", "Wires cost 45 dollars.", "faq.md", None, "Wires")
OTHER = Chunk("misc.md:0:0", "Unrelated text.", "misc.md", None, "Misc")

Q_FEE = Question("q1", "fee?", True, relevant=[{"source": "faq.md", "contains": "monthly fee"}])
Q_MISS = Question("q2", "missing?", True, relevant=[{"source": "faq.md", "contains": "nothing"}])
Q_NONE = Question("q3", "unanswerable?", False)


def _hit(chunk, rerank):
    """
    Function name: _hit
    Purpose: Build a hit with the given re-ranker score.
    Parameters:
        chunk (Chunk): The chunk of the hit.
        rerank (float): Re-ranker score stored under the rerank key.
    Returns:
        Hit: The hit.
    """
    return Hit(chunk, rerank, {"rerank": rerank})


class _Store:
    """
    Class name: _Store
    Purpose: Fake vector store with a name only.
    """

    name = "fakestore"


class FakeRetriever:
    """
    Class name: FakeRetriever
    Purpose: Retriever that returns scripted hits per question text and records each call.
    """

    def __init__(self, script):
        """
        Function name: FakeRetriever.__init__
        Purpose: Store the scripted hits and the chunk list.
        Parameters:
            script (dict[str, list[Hit]]): Hits to return for each question text.
        Returns:
            None: Initialises the instance.
        """
        self.script = script
        self.chunks = [FEE, WIRE, OTHER]
        self.store = _Store()
        self.calls: list[tuple[str, str, int]] = []

    def retrieve(self, query, mode="full", top_k=5):
        """
        Function name: FakeRetriever.retrieve
        Purpose: Record the call and return the scripted hits cut to top_k.
        Parameters:
            query (str): Question text.
            mode (str): Retrieval mode.
            top_k (int): Largest number of hits to return.
        Returns:
            list[Hit]: The scripted hits.
        """
        self.calls.append((query, mode, top_k))
        return self.script.get(query, [])[:top_k]


@pytest.fixture
def eval_env(settings, monkeypatch):
    """
    Function name: eval_env
    Purpose: Fixture that replaces get_settings, load_questions and load_retriever in
        eval.retrieval_eval with fakes: three questions (found at rank 2, missed, and
        unanswerable) and a scripted retriever.
    Parameters:
        settings (Settings): Test settings fixture.
        monkeypatch (pytest.MonkeyPatch): Replaces the module functions.
    Returns:
        FakeRetriever: The fake retriever.
    """
    retriever = FakeRetriever(
        {
            "fee?": [_hit(OTHER, 2.0), _hit(FEE, 1.0)],
            "missing?": [_hit(WIRE, 1.0)],
            "unanswerable?": [_hit(OTHER, 0.5)],
        }
    )
    monkeypatch.setattr(re_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(re_mod, "load_questions", lambda: [Q_FEE, Q_MISS, Q_NONE])
    monkeypatch.setattr(re_mod, "load_retriever", lambda s: retriever)
    return retriever


def test_evaluate_counts_and_metrics_for_one_mode(eval_env, settings):
    """
    Function name: test_evaluate_counts_and_metrics_for_one_mode
    Purpose: Check the header counts, the per-mode metrics (found at rank 2 and one miss) and
        that only answerable questions are queried, with the requested depth.
    Parameters:
        eval_env (FakeRetriever): Fake retriever fixture.
        settings (Settings): Test settings fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    out = re_mod.evaluate(["dense"], depth=7)
    assert out["questions"] == 3 and out["answerable"] == 2 and out["unanswerable"] == 1
    assert out["chunks"] == 3 and out["vector_store"] == "fakestore"
    assert out["embedding_model"] == settings.embedding.model
    assert out["rerank_model"] == settings.retrieval.rerank_model
    m = out["modes"]["dense"]
    assert m["recall_at_5"] == 0.5 and m["mrr_at_10"] == 0.25
    assert m["misses"] == ["q2"]
    assert m["per_question"] == [
        {"id": "q1", "first_relevant_rank": 2},
        {"id": "q2", "first_relevant_rank": None},
    ]
    assert m["median_latency_ms"] >= 0
    assert eval_env.calls == [("fee?", "dense", 7), ("missing?", "dense", 7)]
    assert "not_found_gate" not in out and "recall_at_5" not in out


def test_evaluate_full_mode_adds_gate_and_flat_metrics(eval_env, settings):
    """
    Function name: test_evaluate_full_mode_adds_gate_and_flat_metrics
    Purpose: Check that the full mode adds the not-found gate statistics, using the
        configured threshold, and copies Recall@5 and MRR to the top level for eval.gate.
    Parameters:
        eval_env (FakeRetriever): Fake retriever fixture.
        settings (Settings): Test settings fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    out = re_mod.evaluate(["dense", "full"])
    assert out["recall_at_5"] == out["modes"]["full"]["recall_at_5"] == 0.5
    assert out["mrr_at_10"] == 0.25
    g = out["not_found_gate"]
    assert g["threshold"] == settings.retrieval.not_found_threshold == 1.5
    # q1 top score 2.0 (answerable, above), q2 1.0 (answerable, below), q3 0.5 (below).
    assert g["accuracy"] == round(2 / 3, 4)
    assert g["answerable_min_score"] == 1.0 and g["unanswerable_max_score"] == 0.5
    assert g["scores"] == [
        {"id": "q1", "answerable": True, "top_rerank": 2.0},
        {"id": "q2", "answerable": True, "top_rerank": 1.0},
        {"id": "q3", "answerable": False, "top_rerank": 0.5},
    ]
    assert ("unanswerable?", "full", 1) in eval_env.calls


def test_evaluate_with_no_modes_gives_empty_modes(eval_env):
    """
    Function name: test_evaluate_with_no_modes_gives_empty_modes
    Purpose: Check that an empty mode list gives no mode results and no gate, without
        querying the retriever.
    Parameters:
        eval_env (FakeRetriever): Fake retriever fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    out = re_mod.evaluate([])
    assert out["modes"] == {} and "not_found_gate" not in out
    assert eval_env.calls == []


def test_evaluate_full_mode_when_retriever_returns_no_hits(eval_env):
    """
    Function name: test_evaluate_full_mode_when_retriever_returns_no_hits
    Purpose: Check that an empty retrieval result does not crash the evaluation and that
        the questions count as not found.
    Parameters:
        eval_env (FakeRetriever): Fake retriever fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    eval_env.script.clear()
    out = re_mod.evaluate(["full"])
    assert out["not_found_gate"]["accuracy"] == round(1 / 3, 4)


def test_evaluate_with_only_unanswerable_questions(eval_env, monkeypatch):
    """
    Function name: test_evaluate_with_only_unanswerable_questions
    Purpose: Check that a golden set without answerable questions finishes with zero
        metrics and no median latency.
    Parameters:
        eval_env (FakeRetriever): Fake retriever fixture.
        monkeypatch (pytest.MonkeyPatch): Replaces load_questions.
    Returns:
        None: The test passes or fails through assertions.
    """
    monkeypatch.setattr(re_mod, "load_questions", lambda: [Q_NONE])
    out = re_mod.evaluate(["dense"])
    assert out["modes"]["dense"]["recall_at_5"] == 0.0
    assert out["modes"]["dense"]["median_latency_ms"] is None


def test_retrieval_main_writes_json_and_prints_table(
    eval_env, settings, tmp_path, monkeypatch, capsys
):
    """
    Function name: test_retrieval_main_writes_json_and_prints_table
    Purpose: Check that main parses --modes (with spaces and empty items), creates the output
        folder, writes the results JSON and prints the table and the gate line.
    Parameters:
        eval_env (FakeRetriever): Fake retriever fixture.
        settings (Settings): Test settings fixture.
        tmp_path (Path): Temporary directory.
        monkeypatch (pytest.MonkeyPatch): Sets sys.argv and silences logging set-up.
        capsys (pytest.CaptureFixture): Captures the printed table.
    Returns:
        None: The test passes or fails through assertions.
    """
    monkeypatch.setattr(re_mod, "configure_logging", lambda level: None)
    out_file = tmp_path / "new" / "dir" / "retrieval.json"
    monkeypatch.setattr(
        sys, "argv", ["retrieval_eval", "--modes", " dense, full ,", "--out", str(out_file)]
    )
    re_mod.main()
    doc = json.loads(out_file.read_text(encoding="utf-8"))
    assert list(doc["modes"]) == ["dense", "full"]
    assert doc["recall_at_5"] == 0.5
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == ["mode", "Recall@5", "MRR@10", "p50", "ms"]
    assert lines[1].split()[:3] == ["dense", "0.500", "0.250"]
    assert lines[2].split()[:3] == ["full", "0.500", "0.250"]
    assert lines[3].startswith("not-found gate @ 1.5: accuracy 0.667 (answerable min 1.0")


def test_retrieval_main_without_full_mode_prints_no_gate_line(
    eval_env, tmp_path, monkeypatch, capsys
):
    """
    Function name: test_retrieval_main_without_full_mode_prints_no_gate_line
    Purpose: Check that main prints no gate line when the full mode is not evaluated.
    Parameters:
        eval_env (FakeRetriever): Fake retriever fixture.
        tmp_path (Path): Temporary directory.
        monkeypatch (pytest.MonkeyPatch): Sets sys.argv and silences logging set-up.
        capsys (pytest.CaptureFixture): Captures the printed table.
    Returns:
        None: The test passes or fails through assertions.
    """
    monkeypatch.setattr(re_mod, "configure_logging", lambda level: None)
    out_file = tmp_path / "r.json"
    monkeypatch.setattr(sys, "argv", ["retrieval_eval", "--modes", "bm25", "--out", str(out_file)])
    re_mod.main()
    assert "not-found gate" not in capsys.readouterr().out
    assert "not_found_gate" not in json.loads(out_file.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------------
# eval.answer_eval
# ---------------------------------------------------------------------------------------


class Judge:
    """
    Class name: Judge
    Purpose: Fake judge LLM that gives one scripted reply for the faithfulness prompt and
        another for the relevancy prompt, and records every call.
    """

    def __init__(self, faith="", rel=""):
        """
        Function name: Judge.__init__
        Purpose: Store the scripted replies and start an empty call log.
        Parameters:
            faith (str): Reply for the faithfulness prompt.
            rel (str): Reply for the relevancy prompt.
        Returns:
            None: Initialises the instance.
        """
        self.faith = faith
        self.rel = rel
        self.calls: list[dict] = []

    def complete(self, messages, *, temperature=None, max_tokens=None):
        """
        Function name: Judge.complete
        Purpose: Record the call and return the reply that matches the system prompt.
        Parameters:
            messages (list[dict]): Chat messages with "role" and "content" keys.
            temperature (float | None): Sampling temperature requested.
            max_tokens (int | None): Reply token limit requested.
        Returns:
            str: The scripted reply.
        """
        self.calls.append(
            {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        )
        return self.faith if messages[0]["content"] == ae.FAITHFULNESS_SYSTEM else self.rel


def test_faithfulness_scores_share_of_supported_statements():
    """
    Function name: test_faithfulness_scores_share_of_supported_statements
    Purpose: Check the score and the claim pairs for a two-sentence answer with one YES and
        one NO, and the judge prompt: system text, numbered statements, context, temperature
        0 and the token limit.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    judge = Judge(faith="1: YES\n2: NO")
    score, claims = ae.faithfulness(judge, "CTX TEXT", "The fee is 8 dollars. Wires cost 45.")
    assert score == 0.5
    assert claims == [[True, "The fee is 8 dollars."], [False, "Wires cost 45."]]
    call = judge.calls[0]
    assert call["temperature"] == 0.0 and call["max_tokens"] == 60
    assert call["messages"][0] == {"role": "system", "content": ae.FAITHFULNESS_SYSTEM}
    assert call["messages"][1]["content"] == (
        "CONTEXT:\nCTX TEXT\n\nSTATEMENTS:\n1. The fee is 8 dollars.\n2. Wires cost 45."
    )


def test_faithfulness_partial_verdicts_use_only_judged_statements():
    """
    Function name: test_faithfulness_partial_verdicts_use_only_judged_statements
    Purpose: Check that statements the judge did not vote on are left out of the score, and
        that numbers out of range are ignored.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    judge = Judge(faith="2: YES\n9: NO")
    score, claims = ae.faithfulness(judge, "c", "One is here. Two is here. Three is here.")
    assert score == 1.0
    assert claims == [[True, "Two is here."]]


@pytest.mark.parametrize("reply", ["", "I cannot say.", "yes\nno", "0: YES"])
def test_faithfulness_unparseable_judge_output_gives_none(reply):
    """
    Function name: test_faithfulness_unparseable_judge_output_gives_none
    Purpose: Check that judge output without numbered verdicts gives no score and no claims.
    Parameters:
        reply (str): Judge reply text.
    Returns:
        None: The test passes or fails through assertions.
    """
    assert ae.faithfulness(Judge(faith=reply), "c", "A sentence here.") == (None, [])


def test_faithfulness_empty_answer_skips_the_judge():
    """
    Function name: test_faithfulness_empty_answer_skips_the_judge
    Purpose: Check that an empty or blank answer gives no score and makes no judge call.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    judge = Judge(faith="1: YES")
    assert ae.faithfulness(judge, "c", "") == (None, [])
    assert ae.faithfulness(judge, "c", "   ") == (None, [])
    assert judge.calls == []


def test_answer_relevancy_scores_cosine_with_generated_questions(encoder):
    """
    Function name: test_answer_relevancy_scores_cosine_with_generated_questions
    Purpose: Check that list markers are stripped from the generated questions, only lines
        with a question mark count, at most three are used, and an identical question gives
        a score of 1.0. Also check the judge prompt settings.
    Parameters:
        encoder (FakeEncoder): Fake encoder fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    reply = (
        "Intro line\n1. What is the fee?\n- What is the fee?\n* What is the fee?\n"
        "4) What is the fee?"
    )
    judge = Judge(rel=reply)
    score, generated = ae.answer_relevancy(judge, encoder, "What is the fee?", "The fee is 8.")
    assert generated == ["What is the fee?"] * 3
    assert score == pytest.approx(1.0)
    call = judge.calls[0]
    assert call["temperature"] == 0.0 and call["max_tokens"] == 120
    assert call["messages"][0]["content"] == ae.RELEVANCY_SYSTEM
    assert call["messages"][1]["content"] == "The fee is 8."


def test_answer_relevancy_lower_for_unrelated_questions(encoder):
    """
    Function name: test_answer_relevancy_lower_for_unrelated_questions
    Purpose: Check that unrelated generated questions score lower than matching ones, and
        that a score is a float between 0 and 1.
    Parameters:
        encoder (FakeEncoder): Fake encoder fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    same, _ = ae.answer_relevancy(
        Judge(rel="Monthly fee amount?"), encoder, "Monthly fee amount?", "a"
    )
    other, _ = ae.answer_relevancy(
        Judge(rel="Zebra giraffe lion?"), encoder, "Monthly fee amount?", "a"
    )
    assert same == pytest.approx(1.0)
    assert 0.0 <= other < same


@pytest.mark.parametrize("reply", ["", "No questions here\nat all"])
def test_answer_relevancy_without_questions_gives_none(encoder, reply):
    """
    Function name: test_answer_relevancy_without_questions_gives_none
    Purpose: Check that a judge reply without a question mark gives no score.
    Parameters:
        encoder (FakeEncoder): Fake encoder fixture.
        reply (str): Judge reply text.
    Returns:
        None: The test passes or fails through assertions.
    """
    assert ae.answer_relevancy(Judge(rel=reply), encoder, "q?", "a") == (None, [])


def _row(answerable, found, ok, citations=(), faith=None, rel=None, cc=None, timings=None):
    """
    Function name: _row
    Purpose: Build one per-question result row for the summary tests.
    Parameters:
        answerable (bool): Whether the question is answerable.
        found (bool): Whether the pipeline found an answer.
        ok (bool): Whether the not-found decision was correct.
        citations (tuple[str, ...]): Citation labels.
        faith (float | None): Faithfulness score, left out when None.
        rel (float | None): Relevancy score, left out when None.
        cc (bool | None): Citation correctness, left out when None.
        timings (dict | None): Timings dictionary.
    Returns:
        dict: The result row.
    """
    row = {
        "answerable": answerable,
        "found": found,
        "not_found_correct": ok,
        "citations": list(citations),
        "timings": timings if timings is not None else {"total_ms": 100.0},
    }
    if faith is not None:
        row["faithfulness"] = faith
    if rel is not None:
        row["answer_relevancy"] = rel
    if cc is not None:
        row["citation_correct"] = cc
    return row


def test_summarize_mixed_rows():
    """
    Function name: test_summarize_mixed_rows
    Purpose: Check every summary metric for one answered question, one missed answerable
        question and one declined unanswerable question.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    rows = [
        _row(True, True, True, ["a"], 1.0, 0.8, True, {"llm_first_token_ms": 100, "total_ms": 200}),
        _row(True, False, False, timings={"total_ms": 400}),
        _row(False, False, True, timings={"total_ms": 300}),
    ]
    assert ae.summarize(rows) == {
        "faithfulness": 1.0,
        "faithfulness_scored": 1,
        "answer_relevancy": 0.8,
        "answer_relevancy_scored": 1,
        "answered_rate": 0.5,
        "not_found_accuracy": 0.6667,
        "unanswerable_declined": 1.0,
        "citation_rate": 1.0,
        "citation_correct": 1.0,
        "median_llm_first_token_ms": 100.0,
        "median_total_ms": 300.0,
    }


def test_summarize_skips_missing_scores_and_counts_wrong_citations():
    """
    Function name: test_summarize_skips_missing_scores_and_counts_wrong_citations
    Purpose: Check that rows without a faithfulness or relevancy score are not counted, that
        an answer without citations lowers the citation rate and that a missing
        citation_correct counts as wrong.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    rows = [
        _row(True, True, True, ["a"], 0.5, 0.4, True),
        _row(True, True, True, [], None, None, None),
        _row(True, True, True, ["b"], 1.0, None, False),
    ]
    s = ae.summarize(rows)
    assert s["faithfulness"] == 0.75 and s["faithfulness_scored"] == 2
    assert s["answer_relevancy"] == 0.4 and s["answer_relevancy_scored"] == 1
    assert s["citation_rate"] == 0.6667
    assert s["citation_correct"] == 0.3333
    assert s["answered_rate"] == 1.0


def test_summarize_empty_rows_gives_zeros_and_none():
    """
    Function name: test_summarize_empty_rows_gives_zeros_and_none
    Purpose: Check that an empty list gives zero rates and None for the median timings.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    s = ae.summarize([])
    assert s["answered_rate"] == 0.0 and s["faithfulness"] == 0.0
    assert s["faithfulness_scored"] == 0 and s["answer_relevancy"] == 0.0
    assert s["median_llm_first_token_ms"] is None and s["median_total_ms"] is None


def test_summarize_without_answerable_questions_has_zero_answered_rate():
    """
    Function name: test_summarize_without_answerable_questions_has_zero_answered_rate
    Purpose: Check that a set of unanswerable questions only does not divide by zero and
        reports the decline rate.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    s = ae.summarize([_row(False, False, True), _row(False, True, False)])
    assert s["answered_rate"] == 0.0
    assert s["unanswerable_declined"] == 0.5
    assert s["not_found_accuracy"] == 0.5


ANSWERED = Question("q1", "fee?", True, relevant=[{"source": "faq.md", "contains": "monthly fee"}])
UNKNOWN = Question("q2", "missing?", True, relevant=[{"source": "faq.md", "contains": "x"}])


class FakePipeline:
    """
    Class name: FakePipeline
    Purpose: Fake answer pipeline that returns scripted answer results per question.
    """

    def __init__(self):
        """
        Function name: FakePipeline.__init__
        Purpose: Build the scripted results, the chunk index and the call log.
        Parameters:
            None
        Returns:
            None: Initialises the instance.
        """
        self.few_shot = None
        self.context_k = 1
        self.calls: list[tuple[str, float | None]] = []
        self.retriever = type("R", (), {"by_id": {FEE.chunk_id: FEE, WIRE.chunk_id: WIRE}})()
        cite = Citation("faq.md", None, "Fee", FEE.chunk_id, "faq.md")
        passages = [
            {"chunk_id": FEE.chunk_id, "text": FEE.text},
            {"chunk_id": "x", "text": "ignored"},
        ]
        self.results = {
            "fee?": AnswerResult(
                "fee?",
                "fee query",
                "The fee is 8 dollars. [faq.md]",
                "The fee is 8 dollars.",
                True,
                [cite],
                passages,
                3.0,
                {"llm_first_token_ms": 50.0, "total_ms": 90.0},
            ),
            "missing?": AnswerResult(
                "missing?",
                "missing?",
                "Not found.",
                "Not found.",
                False,
                [],
                [],
                None,
                {"total_ms": 10.0},
            ),
        }

    def answer(self, question, temperature=None):
        """
        Function name: FakePipeline.answer
        Purpose: Record the call and return the scripted result.
        Parameters:
            question (str): Question text.
            temperature (float | None): Answer temperature requested.
        Returns:
            AnswerResult: The scripted result.
        """
        self.calls.append((question, temperature))
        return self.results[question]


@pytest.fixture
def run_env(settings, encoder, monkeypatch):
    """
    Function name: run_env
    Purpose: Fixture that replaces the settings, golden questions, pipeline, judge and encoder
        used by eval.answer_eval.run with fakes.
    Parameters:
        settings (Settings): Test settings fixture.
        encoder (FakeEncoder): Fake encoder fixture.
        monkeypatch (pytest.MonkeyPatch): Replaces the module functions.
    Returns:
        SimpleNamespace-like object: Holds pipeline, judge and the judge settings seen.
    """
    import rag.llm as llm_mod
    import rag.pipeline as pipeline_mod
    import retrieval.embeddings as emb_mod

    pipe = FakePipeline()
    judge = Judge(faith="1: YES", rel="fee?")
    seen = {}

    def fake_build_llm(cfg):
        """
        Function name: fake_build_llm
        Purpose: Record the LLM settings and return the fake judge.
        Parameters:
            cfg (LlmSettings): LLM settings passed by run().
        Returns:
            Judge: The fake judge.
        """
        seen["cfg"] = cfg
        return judge

    monkeypatch.setattr(ae, "get_settings", lambda: settings)
    monkeypatch.setattr(ae, "load_questions", lambda: [ANSWERED, UNKNOWN])
    monkeypatch.setattr(pipeline_mod, "build_pipeline", lambda s: pipe)
    monkeypatch.setattr(llm_mod, "build_llm", fake_build_llm)
    monkeypatch.setattr(emb_mod, "get_encoder", lambda s: encoder)
    return type("Env", (), {"pipe": pipe, "judge": judge, "seen": seen})()


def test_run_builds_one_row_per_question(run_env, settings):
    """
    Function name: test_run_builds_one_row_per_question
    Purpose: Check the rows for an answered question (citation, faithfulness and relevancy
        filled in) and a not-found question (only the basic fields), the settings returned,
        and that the pipeline gets the few-shot flag and temperature.
    Parameters:
        run_env (object): Fake pipeline, judge and settings fixture.
        settings (Settings): Test settings fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    used, rows = ae.run(None, False, 0.3, None)
    assert used is settings
    assert run_env.pipe.few_shot is False
    assert run_env.pipe.calls == [("fee?", 0.3), ("missing?", 0.3)]
    first, second = rows
    assert first["id"] == "q1" and first["query"] == "fee query" and first["found"] is True
    assert first["not_found_correct"] is True and first["citations"] == ["faq.md"]
    assert first["top_score"] == 3.0 and first["citation_correct"] is True
    assert first["faithfulness"] == 1.0 and first["claims"] == [[True, "The fee is 8 dollars."]]
    assert first["answer_relevancy"] == pytest.approx(1.0)
    assert first["generated_questions"] == ["fee?"]
    assert isinstance(first["eval_seconds"], float)
    assert second["found"] is False and second["not_found_correct"] is False
    for key in ("citation_correct", "faithfulness", "answer_relevancy"):
        assert key not in second
    # The judge saw only the first context_k passages, without the citation marker.
    faith_call = run_env.judge.calls[0]["messages"][1]["content"]
    assert faith_call.startswith("CONTEXT:\nMonthly fee is 8 dollars.\n\nSTATEMENTS:")
    assert "ignored" not in faith_call


def test_run_limit_and_judge_model(run_env, settings):
    """
    Function name: test_run_limit_and_judge_model
    Purpose: Check that a limit keeps only the first questions, that the judge model name
        replaces the configured model only for the judge, and that a limit of 0 means all.
    Parameters:
        run_env (object): Fake pipeline, judge and settings fixture.
        settings (Settings): Test settings fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    _, rows = ae.run(1, True, None, "judge-model")
    assert [r["id"] for r in rows] == ["q1"]
    assert run_env.pipe.few_shot is True
    assert run_env.seen["cfg"].model == "judge-model"
    assert settings.llm.model != "judge-model"
    assert run_env.pipe.calls == [("fee?", None)]
    _, rows = ae.run(0, True, None, None)
    assert len(rows) == 2
    assert run_env.seen["cfg"].model == settings.llm.model


def test_run_wrong_citation_is_marked_incorrect(run_env):
    """
    Function name: test_run_wrong_citation_is_marked_incorrect
    Purpose: Check that an answered question whose cited chunk is not relevant gets
        citation_correct False.
    Parameters:
        run_env (object): Fake pipeline, judge and settings fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
    wrong = Citation("faq.md", None, "Wires", WIRE.chunk_id, "faq.md")
    run_env.pipe.results["fee?"].citations = [wrong]
    run_env.pipe.results["fee?"].passages = [{"chunk_id": WIRE.chunk_id, "text": WIRE.text}]
    _, rows = ae.run(1, True, None, None)
    assert rows[0]["citation_correct"] is False


def _run_main(monkeypatch, tmp_path, settings, *args):
    """
    Function name: _run_main
    Purpose: Run eval.answer_eval.main with a fake run(), a temporary results folder and the
        given command-line arguments.
    Parameters:
        monkeypatch (pytest.MonkeyPatch): Replaces module attributes and sets sys.argv.
        tmp_path (Path): Temporary directory.
        settings (Settings): Settings returned by get_settings and run.
        args (str): Command-line arguments after the program name.
    Returns:
        list: The arguments that main passed to run().
    """
    passed = []
    rows = [_row(True, True, True, ["a"], 1.0, 0.8, True), _row(False, False, True)]

    def fake_run(*a):
        """
        Function name: fake_run
        Purpose: Record the arguments and return the settings with two result rows.
        Parameters:
            a (tuple): Arguments passed by main.
        Returns:
            tuple[Settings, list[dict]]: The settings and the rows.
        """
        passed.extend(a)
        return settings, rows

    monkeypatch.setattr(ae, "run", fake_run)
    monkeypatch.setattr(ae, "get_settings", lambda: settings)
    monkeypatch.setattr(ae, "configure_logging", lambda level: None)
    monkeypatch.setattr(ae, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(sys, "argv", ["answer_eval", *args])
    ae.main()
    return passed


def test_answer_main_writes_default_file_and_prints_summary(
    settings, tmp_path, monkeypatch, capsys
):
    """
    Function name: test_answer_main_writes_default_file_and_prints_summary
    Purpose: Check that main without options calls run with all questions and few-shot on,
        creates the results folder, writes answers.json with the settings, summary and rows,
        and prints the summary as JSON.
    Parameters:
        settings (Settings): Test settings fixture.
        tmp_path (Path): Temporary directory.
        monkeypatch (pytest.MonkeyPatch): Replaces module attributes and sets sys.argv.
        capsys (pytest.CaptureFixture): Captures the printed summary.
    Returns:
        None: The test passes or fails through assertions.
    """
    passed = _run_main(monkeypatch, tmp_path, settings)
    assert passed == [None, True, None, None]
    doc = json.loads((tmp_path / "results" / "answers.json").read_text(encoding="utf-8"))
    assert doc["llm_provider"] == "ollama" and doc["llm_model"] == settings.llm.model
    assert doc["judge_model"] == settings.llm.model and doc["few_shot"] is True
    assert doc["temperature"] == settings.llm.temperature
    assert doc["context_k"] == settings.retrieval.context_k
    assert doc["not_found_threshold"] == settings.retrieval.not_found_threshold
    assert doc["questions"] == 2 and len(doc["rows"]) == 2
    assert doc["summary"]["answered_rate"] == 1.0
    assert json.loads(capsys.readouterr().out) == doc["summary"]


def test_answer_main_options_tag_and_overrides(settings, tmp_path, monkeypatch):
    """
    Function name: test_answer_main_options_tag_and_overrides
    Purpose: Check that --limit, --no-few-shot, --temperature, --judge-model and --tag are
        passed on and recorded, and that the tag gives the file name answers_<tag>.json.
    Parameters:
        settings (Settings): Test settings fixture.
        tmp_path (Path): Temporary directory.
        monkeypatch (pytest.MonkeyPatch): Replaces module attributes and sets sys.argv.
    Returns:
        None: The test passes or fails through assertions.
    """
    passed = _run_main(
        monkeypatch,
        tmp_path,
        settings,
        "--limit", "5", "--no-few-shot", "--temperature", "0.0",
        "--judge-model", "judge-x", "--tag", "zero_shot",
    )  # fmt: skip
    assert passed == [5, False, 0.0, "judge-x"]
    assert not (tmp_path / "results" / "answers.json").exists()
    doc = json.loads((tmp_path / "results" / "answers_zero_shot.json").read_text(encoding="utf-8"))
    assert doc["few_shot"] is False and doc["temperature"] == 0.0
    assert doc["judge_model"] == "judge-x"


def test_answer_main_records_openai_model_for_openai_provider(settings, tmp_path, monkeypatch):
    """
    Function name: test_answer_main_records_openai_model_for_openai_provider
    Purpose: Check that the recorded answer model is the OpenAI model name when the provider
        is openai.
    Parameters:
        settings (Settings): Test settings fixture.
        tmp_path (Path): Temporary directory.
        monkeypatch (pytest.MonkeyPatch): Replaces module attributes and sets sys.argv.
    Returns:
        None: The test passes or fails through assertions.
    """
    llm = settings.llm.model_copy(update={"provider": "openai", "openai_model": "gpt-mini"})
    openai_settings = settings.model_copy(update={"llm": llm})
    _run_main(monkeypatch, tmp_path, openai_settings)
    doc = json.loads((tmp_path / "results" / "answers.json").read_text(encoding="utf-8"))
    assert doc["llm_provider"] == "openai" and doc["llm_model"] == "gpt-mini"
