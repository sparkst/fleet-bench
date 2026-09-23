"""Judge prompt building and score parsing."""

from __future__ import annotations

from fleetbench.graders.judge import (
    build_judge_prompt,
    parse_judge_response,
    recognized_dimension_count,
    score_from_dimensions,
)

RUBRIC = [
    {"dimension": "correctness", "desc": "is it right"},
    {"dimension": "clarity", "desc": "is it clear"},
]


def test_prompt_hides_model_and_lists_dimensions():
    prompt = build_judge_prompt("coding", "do x", RUBRIC, "some answer")
    assert "correctness" in prompt and "clarity" in prompt
    assert "do x" in prompt and "some answer" in prompt
    # No model id leaks into the judge prompt.
    assert "model" in prompt.lower()  # only the generic instruction text


def test_parse_plain_json():
    scores = parse_judge_response('{"correctness": 5, "clarity": 3}', RUBRIC)
    assert scores == {"correctness": 5, "clarity": 3}


def test_parse_json_in_prose_and_fence():
    text = "Here is my score:\n```json\n{\"correctness\": 4, \"clarity\": 2}\n```\n"
    scores = parse_judge_response(text, RUBRIC)
    assert scores == {"correctness": 4, "clarity": 2}


def test_parse_clamps_and_defaults_missing():
    scores = parse_judge_response('{"correctness": 9}', RUBRIC)
    assert scores["correctness"] == 5  # clamped to max
    assert scores["clarity"] == 0  # missing -> 0


def test_score_from_dimensions_normalized():
    assert score_from_dimensions({"a": 5, "b": 5}) == 1.0
    assert score_from_dimensions({"a": 0, "b": 5}) == 0.5
    assert score_from_dimensions({}) == 0.0


def test_recognized_dimension_count():
    assert recognized_dimension_count('{"correctness": 4, "clarity": 3}', RUBRIC) == 2
    assert recognized_dimension_count('{"correctness": 4}', RUBRIC) == 1
    # Prose / refusal / non-JSON: zero recognized -> caller marks n/a, not a 0 score.
    assert recognized_dimension_count("I cannot score this response.", RUBRIC) == 0
    assert recognized_dimension_count("", RUBRIC) == 0
