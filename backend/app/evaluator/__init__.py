"""
MiroFish Evaluator — AP Social Studies Question Quality Control Pipeline

Educational content evaluation framework with:
  1. LLM-based quality checks (question QC + explanation QC)
  2. IRT psychometric simulation (calibrated to AP score distributions)
  3. Unified reporting

Architecture:
  evaluator/
    __init__.py          ← this file
    pipeline.py          ← main orchestrator
    modules/
      question_qc.py     ← LLM-based question quality checks
      explanation_qc.py  ← LLM-based explanation quality checks
      psychometric.py    ← IRT simulation engine (MiroFish-unique)
    config/
      prompts.json       ← check prompts (adapted for AP Social Studies)
    utils.py             ← shared utilities
"""
