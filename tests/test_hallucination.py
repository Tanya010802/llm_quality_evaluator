"""
test_hallucination.py
Tests that the LLM does not fabricate information beyond what's in the context.
Uses lightweight local heuristics (no external API keys required).
"""

import pytest
from utils.llm_client import get_completion_with_context


@pytest.mark.hallucination
def test_no_hallucination_in_factual_response(test_cases, report):
    """Model must not contradict or fabricate beyond the given context."""

    # Pick only hallucination-category test cases
    cases = [tc for tc in test_cases if tc["category"] == "hallucination"]

    for tc in cases:
        prompt = tc["prompt"]
        context = tc["context"]

        # Get LLM response grounded in context
        actual_output = get_completion_with_context(prompt, context)

        expected_keywords = tc.get("expected_keywords", [])
        missing = [kw for kw in expected_keywords if kw.lower() not in actual_output.lower()]

        # Heuristic: ensure we mention only things supported by context for this case.
        # (This project’s test cases are small and explicitly keyworded.)
        context_lower = (context or "").lower()
        output_lower = actual_output.lower()

        # Flag obvious hallucinations of programming languages not in context.
        known_langs = {"python", "javascript", "java", "c++", "c#", "ruby", "go", "rust", "typescript", "php", "swift", "kotlin"}
        hallucinated_langs = sorted(
            lang for lang in known_langs
            if (lang in output_lower) and (lang not in context_lower)
        )

        passed = (len(missing) == 0) and (len(hallucinated_langs) == 0)

        report.add_result(
            test_id=tc["id"],
            category="hallucination",
            passed=passed,
            details={
                "prompt": prompt,
                "context": context,
                "actual_output": actual_output,
                "expected_keywords": expected_keywords,
                "missing_keywords": missing,
                "hallucinated_langs": hallucinated_langs,
            },
        )

        assert passed, (
            f"[{tc['id']}] Hallucination detected!\n"
            f"Missing keywords: {missing}\n"
            f"Hallucinated languages: {hallucinated_langs}\n"
            f"Output: {actual_output}"
        )
