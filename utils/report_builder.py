"""
report_builder.py
Builds a simple JSON summary report of eval results.
pytest-html handles the rich HTML report — this is a machine-readable supplement.
"""

import json
import os
from datetime import UTC, datetime


class ReportBuilder:
    def __init__(self):
        self.results = []
        self.start_time = datetime.now(UTC).isoformat()

    def add_result(self, test_id: str, category: str, passed: bool, details: dict):
        self.results.append({
            "test_id": test_id,
            "category": category,
            "passed": passed,
            "details": details,
            "timestamp": datetime.now(UTC).isoformat(),
        })

    def save(self, output_dir: str = "reports"):
        os.makedirs(output_dir, exist_ok=True)
        summary = {
            "run_start": self.start_time,
            "run_end": datetime.now(UTC).isoformat(),
            "total": len(self.results),
            "passed": sum(1 for r in self.results if r["passed"]),
            "failed": sum(1 for r in self.results if not r["passed"]),
            "results": self.results,
        }
        path = os.path.join(output_dir, "eval_summary.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nJSON summary saved to: {path}")
        return summary
