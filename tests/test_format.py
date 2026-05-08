"""
test_format.py
Tests that the LLM returns output in the expected format (JSON, text, list, etc.)
No external metric needed — this is deterministic structural validation.
"""

import json
import pytest
from utils.llm_client import get_completion


@pytest.mark.format
def test_json_format_adherence(test_cases, report):
    """When JSON is expected, the response must be valid parseable JSON."""

    cases = [tc for tc in test_cases if tc["category"] == "json_format"]

    for tc in cases:
        prompt = tc["prompt"] + "\n\nIMPORTANT: Respond ONLY with valid JSON. No explanation, no markdown."
        actual_output = get_completion(prompt)

        # Strip markdown code fences if model wraps JSON in them
        cleaned = actual_output.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

        passed = False
        parse_error = None
        parsed = None

        try:
            parsed = json.loads(cleaned)
            # Check that expected keys are present
            expected_keys = tc.get("expected_keywords", [])
            missing_keys = [k for k in expected_keys if k not in parsed]
            passed = len(missing_keys) == 0
            parse_error = f"Missing keys: {missing_keys}" if missing_keys else None
        except json.JSONDecodeError as e:
            parse_error = str(e)

        report.add_result(
            test_id=tc["id"],
            category="format",
            passed=passed,
            details={
                "prompt": prompt,
                "actual_output": actual_output,
                "parsed_json": parsed,
                "error": parse_error,
            },
        )

        assert passed, (
            f"[{tc['id']}] Format check failed!\n"
            f"Error: {parse_error}\n"
            f"Raw output: {actual_output}"
        )


@pytest.mark.format
def test_text_contains_expected_keywords(test_cases, report):
    """For text responses, validate that expected keywords appear in the output."""

    cases = [tc for tc in test_cases if tc["category"] == "factual"]

    for tc in cases:
        prompt = tc["prompt"]
        actual_output = get_completion(prompt)

        expected_keywords = tc.get("expected_keywords", [])
        missing = [kw for kw in expected_keywords if kw.lower() not in actual_output.lower()]
        passed = len(missing) == 0

        report.add_result(
            test_id=tc["id"],
            category="format",
            passed=passed,
            details={
                "prompt": prompt,
                "actual_output": actual_output,
                "expected_keywords": expected_keywords,
                "missing_keywords": missing,
            },
        )

        assert passed, (
            f"[{tc['id']}] Missing expected keywords: {missing}\n"
            f"Output: {actual_output}"
        )
