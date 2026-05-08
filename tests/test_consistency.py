"""
test_consistency.py
Tests that the LLM produces stable, consistent outputs for the same prompt
across multiple runs. Useful for catching non-deterministic or flaky model behavior.
"""

import pytest
from utils.llm_client import get_completion


@pytest.mark.consistency
def test_output_consistency_across_runs(test_cases, report):
    """
    Run the same prompt 3 times and verify all responses contain
    the expected keywords. Flags models that produce wildly different answers.
    """

    cases = [tc for tc in test_cases if tc["category"] == "consistency"]
    NUM_RUNS = 3

    for tc in cases:
        prompt = tc["prompt"]
        expected_keywords = tc.get("expected_keywords", [])

        outputs = []
        run_results = []

        for run in range(NUM_RUNS):
            output = get_completion(prompt, temperature=0.3)  # slight temp for variance testing
            outputs.append(output)

            missing = [kw for kw in expected_keywords if kw.lower() not in output.lower()]
            run_results.append({
                "run": run + 1,
                "output": output,
                "missing_keywords": missing,
                "passed": len(missing) == 0,
            })

        all_passed = all(r["passed"] for r in run_results)

        report.add_result(
            test_id=tc["id"],
            category="consistency",
            passed=all_passed,
            details={
                "prompt": prompt,
                "expected_keywords": expected_keywords,
                "runs": run_results,
            },
        )

        failed_runs = [r for r in run_results if not r["passed"]]
        assert all_passed, (
            f"[{tc['id']}] Consistency check failed on {len(failed_runs)}/{NUM_RUNS} runs!\n"
            + "\n".join(
                f"  Run {r['run']}: Missing {r['missing_keywords']} | Output: {r['output'][:100]}..."
                for r in failed_runs
            )
        )
