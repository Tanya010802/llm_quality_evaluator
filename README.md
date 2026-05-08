# LLM Quality Evaluator

A lightweight Python framework for evaluating LLM output quality across multiple dimensions — built for AI-focused QA/SDET workflows.

## What It Tests

| Dimension | What It Checks |
|---|---|
| **Hallucination** | Does the response contradict the input context? |
| **Relevance** | Is the response on-topic with the prompt? |
| **Toxicity** | Does the response contain harmful content? |
| **Format Adherence** | Does the output match the expected format (JSON, list, etc.)? |
| **Consistency** | Does the same prompt produce stable outputs across runs? |

## Tech Stack

- Python 3.10+
- `deepeval` — LLM evaluation library
- `openai` — LLM API client
- `pytest` — test runner
- `pytest-html` — HTML report generation
- GitHub Actions — CI pipeline

## Project Structure

```
llm_quality_evaluator/
├── prompts/
│   └── test_cases.json         # Prompt inputs + expected behavior definitions
├── tests/
│   ├── test_hallucination.py   # Hallucination detection tests
│   ├── test_relevance.py       # Relevance scoring tests
│   ├── test_format.py          # Format adherence tests
│   └── test_consistency.py     # Output consistency across runs
├── utils/
│   ├── llm_client.py           # OpenAI API wrapper
│   └── report_builder.py       # HTML report generator
├── reports/                    # Auto-generated test reports land here
├── conftest.py                 # Shared pytest fixtures
├── requirements.txt
└── .github/
    └── workflows/
        └── eval_pipeline.yml   # CI pipeline
```

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/llm-quality-evaluator.git
cd llm-quality-evaluator

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your API key
export OPENAI_API_KEY="your-key-here"

# 5. Run the full eval suite
pytest tests/ --html=reports/report.html --self-contained-html -v
```

## Sample Output

```
tests/test_hallucination.py::test_no_hallucination_in_factual_response PASSED
tests/test_relevance.py::test_response_is_relevant PASSED
tests/test_format.py::test_json_format_adherence PASSED
tests/test_consistency.py::test_output_consistency_across_runs PASSED

4 passed in 12.3s
HTML report: reports/report.html
```

## CI

Every push triggers the eval suite via GitHub Actions. See `.github/workflows/eval_pipeline.yml`.

---

Built as a portfolio project demonstrating AI/LLM evaluation skills for SDET roles.
