"""
test_relevance.py
Tests that the LLM response is relevant to the input prompt.
Uses lightweight local heuristics (no external API keys required).
"""

import pytest
from utils.llm_client import get_completion


@pytest.mark.relevance
def test_response_is_relevant(test_cases, report):
    """Model response must be meaningfully on-topic with the prompt."""

    cases = [tc for tc in test_cases if tc["category"] == "relevance"]

    for tc in cases:
        prompt = tc["prompt"]
        actual_output = get_completion(prompt)

        expected_keywords = tc.get("expected_keywords", [])
        missing = [kw for kw in expected_keywords if kw.lower() not in actual_output.lower()]

        # Heuristic: require expected keywords + keep it concise (prompt asks for 2 sentences)
        sentence_count = sum(1 for s in actual_output.replace("?", ".").replace("!", ".").split(".") if s.strip())
        passed = (len(missing) == 0) and (sentence_count <= 2)

        report.add_result(
            test_id=tc["id"],
            category="relevance",
            passed=passed,
            details={
                "prompt": prompt,
                "actual_output": actual_output,
                "expected_keywords": expected_keywords,
                "missing_keywords": missing,
                "sentence_count": sentence_count,
            },
        )

        assert passed, (
            f"[{tc['id']}] Low relevance detected!\n"
            f"Missing keywords: {missing}\n"
            f"Sentence count: {sentence_count}\n"
            f"Output: {actual_output}"
        )
