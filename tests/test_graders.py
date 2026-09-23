"""Deterministic grader tests: classify, trec6, gsm8k, ifeval, humaneval."""

from __future__ import annotations

from fleetbench.graders import grade
from fleetbench.graders.humaneval_exec import extract_code


# --- classify -----------------------------------------------------------------
CLASSIFY_ARGS = {"labels": ["fruit", "furniture", "sports equipment"], "answer": "fruit"}


def test_classify_exact_match():
    r = grade("classify", CLASSIFY_ARGS, "fruit")
    assert r.passed is True and r.score == 1.0


def test_classify_in_sentence():
    r = grade("classify", CLASSIFY_ARGS, "This is clearly a fruit.")
    assert r.passed is True


def test_classify_wrong():
    r = grade("classify", CLASSIFY_ARGS, "furniture")
    assert r.passed is False and r.score == 0.0


# --- trec6 --------------------------------------------------------------------
TREC_ARGS = {"labels": ["ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM"], "answer": "LOC"}


def test_trec6_label_case_insensitive():
    assert grade("trec6_label", TREC_ARGS, "loc").passed is True
    assert grade("trec6_label", TREC_ARGS, "The answer is LOC").passed is True
    assert grade("trec6_label", TREC_ARGS, "HUM").passed is False


def test_trec6_negated_label_is_not_picked_as_final_answer():
    # The gold label (LOC) appears first but is negated; the real final
    # answer (ENTY) appears later. Must NOT score this as a LOC pass.
    r = grade("trec6_label", TREC_ARGS, "Not LOC, it is actually ENTY.")
    assert r.passed is False
    assert r.detail["picked"] == "ENTY"


def test_trec6_negated_wrong_label_lets_correct_final_answer_pass():
    # The wrong label (ENTY) appears first but is negated; the gold label
    # (LOC) is the real final answer.
    r = grade("trec6_label", TREC_ARGS, "Not ENTY, it is LOC")
    assert r.passed is True
    assert r.detail["picked"] == "LOC"


# --- gsm8k --------------------------------------------------------------------
def test_gsm8k_prefers_marker():
    args = {"answer": "18"}
    assert grade("gsm8k_numeric", args, "Some work here 12 + 6 ... #### 18").passed is True


def test_gsm8k_fallback_last_integer():
    args = {"answer": "24"}
    assert grade("gsm8k_numeric", args, "The result is 24").passed is True


def test_gsm8k_handles_commas_and_wrong():
    assert grade("gsm8k_numeric", {"answer": "1000"}, "#### 1,000").passed is True
    assert grade("gsm8k_numeric", {"answer": "5"}, "#### 6").passed is False


def test_gsm8k_reports_marker_presence():
    with_marker = grade("gsm8k_numeric", {"answer": "18"}, "work ... #### 18")
    assert with_marker.detail["marker_present"] is True
    fallback = grade("gsm8k_numeric", {"answer": "24"}, "the result is 24")
    assert fallback.detail["marker_present"] is False


def test_gsm8k_no_marker_does_not_grab_intermediate_value_as_answer():
    # Gold (42) appears as an intermediate value earlier in the response;
    # the real final answer is spelled out in words. With no '####' marker
    # and no digits on the final clause, this must NOT score as a pass.
    args = {"answer": 42}
    response = (
        "The answer is definitely not 42 ... 42 apples intermediate ... forty-nine"
    )
    result = grade("gsm8k_numeric", args, response)
    assert result.passed is not True
    assert result.score != 1.0


# --- ifeval -------------------------------------------------------------------
def test_ifeval_all_rules_pass():
    args = {"rules": [
        {"rule": "keywords:existence", "keywords": ["tide"]},
        {"rule": "punctuation:no_comma"},
    ]}
    r = grade("ifeval_rules", args, "The tide rises and falls near the shore")
    assert r.passed is True and r.score == 1.0


def test_ifeval_partial_credit():
    args = {"rules": [
        {"rule": "keywords:existence", "keywords": ["tide"]},
        {"rule": "punctuation:no_comma"},
    ]}
    r = grade("ifeval_rules", args, "The tide rises, and falls")  # has a comma
    assert r.passed is False
    assert r.score == 0.5


def test_ifeval_end_checker_and_word_count():
    assert grade("ifeval_rules", {"rules": [
        {"rule": "startend:end_checker", "end_phrase": "You can do it."}]},
        "Keep going. You can do it.").passed is True
    assert grade("ifeval_rules", {"rules": [
        {"rule": "length_constraints:number_words", "relation": "at least", "num_words": 5}]},
        "one two three four five six").passed is True


# --- humaneval ----------------------------------------------------------------
HE_ARGS = {
    "entry_point": "add",
    "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n",
    "prompt": "def add(a, b):\n",
}


def test_humaneval_pass():
    resp = "```python\ndef add(a, b):\n    return a + b\n```"
    r = grade("humaneval_exec", HE_ARGS, resp)
    assert r.passed is True


def test_humaneval_fail_wrong_impl():
    resp = "def add(a, b):\n    return a - b\n"
    r = grade("humaneval_exec", HE_ARGS, resp)
    assert r.passed is False


def test_extract_code_prefers_fence():
    assert extract_code("blah\n```python\nx = 1\n```\ntrailing") == "x = 1"
    assert extract_code("no fence here") == "no fence here"


def test_extract_code_strips_language_tag_fences():
    # A ```py (or ```python3) fence must not leak the language token into code.
    assert extract_code("```py\ndef add(a, b):\n    return a + b\n```") == (
        "def add(a, b):\n    return a + b"
    )
    assert "python3" not in extract_code("```python3\nx = 1\n```")


def test_humaneval_py_fence_passes(monkeypatch):
    args = {"entry_point": "add", "test": "def check(c):\n    assert c(2, 3) == 5\n", "prompt": ""}
    assert grade("humaneval_exec", args, "```py\ndef add(a, b):\n    return a + b\n```").passed is True


def test_humaneval_failure_reason_has_no_local_temp_path():
    # A failing solution's traceback must not leak the local temp dir path.
    args = {"entry_point": "add", "test": "def check(c):\n    assert c(2, 3) == 5\n", "prompt": ""}
    r = grade("humaneval_exec", args, "def add(a, b):\n    raise ValueError('nope')\n")
    assert r.passed is False
    reason = r.detail["reason"]
    assert "<tmpdir>" in reason  # temp dir redacted
    assert "/private/" not in reason and "/var/folders/" not in reason


def test_humaneval_subprocess_env_is_scrubbed(monkeypatch):
    # A candidate cannot read a secret from the environment: the subprocess env
    # is scrubbed, so this solution (which asserts the secret is absent) passes.
    monkeypatch.setenv("GROQ_API_KEY", "sekrit-value")
    code = (
        "import os\n"
        "def add(a, b):\n"
        "    assert 'GROQ_API_KEY' not in os.environ\n"
        "    return a + b\n"
    )
    args = {"entry_point": "add", "test": "def check(c):\n    assert c(2, 3) == 5\n", "prompt": ""}
    assert grade("humaneval_exec", args, code).passed is True
