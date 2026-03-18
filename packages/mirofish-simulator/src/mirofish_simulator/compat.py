"""
Compatibility layer for InceptBench integration.

Provides aliases and helper functions expected by InceptBench:
- StructuralAnalyzer (alias for MisconceptionAnalyzer)
- parse_question (question parsing helper)
- analyze_sync methods
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Union

from .misconceptions import MisconceptionAnalyzer, MisconceptionAnalysisResult


def parse_question(content: Union[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Parse question content into standard format.

    Accepts:
    - JSON string
    - Dict with various question formats

    Returns:
    - Normalized dict with: text, options (dict), correct_answer
    - None if parsing fails
    """
    # Parse JSON string
    if isinstance(content, str):
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return None
    else:
        data = content

    if not isinstance(data, dict):
        return None

    # Extract question text
    text = (
        data.get("text")
        or data.get("question")
        or data.get("stem")
        or data.get("question_text")
    )
    if not text:
        return None

    # Extract options - handle multiple formats
    options = {}

    if "answer_options" in data:
        # InceptBench format: [{"key": "A", "text": "..."}, ...]
        for opt in data["answer_options"]:
            key = opt.get("key") or opt.get("option")
            txt = opt.get("text") or opt.get("value")
            if key and txt:
                options[str(key).upper()] = txt

    elif "options" in data:
        raw_opts = data["options"]
        if isinstance(raw_opts, dict):
            # Already a dict: {"A": "...", "B": "..."}
            options = {str(k).upper(): v for k, v in raw_opts.items()}
        elif isinstance(raw_opts, list):
            # List format: ["option1", "option2", ...]
            for i, opt in enumerate(raw_opts):
                key = chr(65 + i)  # A, B, C, D...
                if isinstance(opt, dict):
                    options[key] = opt.get("text") or opt.get("value") or str(opt)
                else:
                    options[key] = str(opt)

    elif "choices" in data:
        # Alternative key
        choices = data["choices"]
        if isinstance(choices, list):
            for i, opt in enumerate(choices):
                options[chr(65 + i)] = str(opt) if not isinstance(opt, dict) else opt.get("text", str(opt))

    if len(options) < 2:
        return None

    # Extract correct answer
    correct = (
        data.get("correct_answer")
        or data.get("answer")
        or data.get("correct")
        or data.get("correct_option")
    )

    if correct is None:
        return None

    # Normalize correct answer to letter
    if isinstance(correct, int):
        correct = chr(65 + correct)
    elif isinstance(correct, str):
        correct = correct.strip().upper()
        if len(correct) > 1 and correct[0].isalpha():
            correct = correct[0]

    return {
        "text": text,
        "options": options,
        "correct_answer": correct,
        "subject": data.get("subject"),
        "grade": data.get("grade"),
        "id": data.get("id") or data.get("question_id"),
    }


class StructuralAnalyzer:
    """
    Structural analysis of question quality.

    Alias for MisconceptionAnalyzer with InceptBench-compatible output format.
    Analyzes:
    - Distractor-to-misconception mapping
    - Plausibility of each option
    - Ambiguity issues
    - Redundant distractors
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ):
        self._analyzer = MisconceptionAnalyzer(
            api_key=api_key,
            base_url=base_url,
            model=model,
        )

    async def analyze(self, question: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze question structure.

        Returns dict with:
        - distractors: {option: {catches, plausible, why_attractive}}
        - ambiguity_issues: list of issues
        - correct_answer_issues: list of issues
        - redundant_pairs: list of [opt1, opt2] pairs
        - fixes: {option: suggested_fix}
        """
        # Normalize question format
        if "text" not in question:
            parsed = parse_question(question)
            if parsed:
                question = parsed

        result = await self._analyzer.analyze(question)

        # Convert to InceptBench-expected format
        distractors = {}
        for opt, mc in result.distractors.items():
            distractors[opt] = {
                "catches": mc.catches_misconception,
                "description": mc.description,
                "plausible": mc.confidence >= 0.5,
                "why_attractive": mc.description,
                "vulnerable_archetypes": mc.vulnerable_archetypes,
                "remediation": mc.remediation_topic,
            }

        # Add unmapped distractors as implausible
        for opt in result.unmapped_distractors:
            distractors[opt] = {
                "catches": "unknown",
                "description": "Could not map to known misconception",
                "plausible": False,
                "why_attractive": "unclear",
                "vulnerable_archetypes": [],
                "remediation": None,
            }

        # Detect redundant pairs (same misconception)
        redundant_pairs = []
        mc_to_opts: Dict[str, List[str]] = {}
        for opt, info in distractors.items():
            mc = info.get("catches", "unknown")
            if mc != "unknown":
                if mc not in mc_to_opts:
                    mc_to_opts[mc] = []
                mc_to_opts[mc].append(opt)

        for mc, opts in mc_to_opts.items():
            if len(opts) >= 2:
                redundant_pairs.append(opts[:2])

        return {
            "distractors": distractors,
            "ambiguity_issues": [],  # Could be enhanced
            "correct_answer_issues": [],  # Could be enhanced
            "redundant_pairs": redundant_pairs,
            "fixes": {},  # Could be enhanced with LLM suggestions
            "coverage_score": result.coverage_score,
            "confidence": result.analysis_confidence,
        }

    def analyze_sync(self, question: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous wrapper for analyze()."""
        return asyncio.get_event_loop().run_until_complete(self.analyze(question))


def add_sync_method_to_accessibility_analyzer():
    """
    Monkey-patch AccessibilityAnalyzer to add analyze_sync method.

    Called on module import.
    """
    from .accessibility import AccessibilityAnalyzer, AccessibilityResult

    def analyze_sync(
        self,
        content: Dict[str, Any],
        target_grade: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Synchronous accessibility analysis.

        Returns dict format for InceptBench compatibility.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, create new loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        self.analyze(content, target_grade, **kwargs)
                    )
                    result = future.result()
            else:
                result = loop.run_until_complete(
                    self.analyze(content, target_grade, **kwargs)
                )
        except RuntimeError:
            # No event loop, create one
            result = asyncio.run(self.analyze(content, target_grade, **kwargs))

        return result.to_dict()

    # Add method to class
    AccessibilityAnalyzer.analyze_sync = analyze_sync


# Apply monkey-patch on import
try:
    add_sync_method_to_accessibility_analyzer()
except Exception:
    pass  # Ignore if accessibility module not loaded yet
