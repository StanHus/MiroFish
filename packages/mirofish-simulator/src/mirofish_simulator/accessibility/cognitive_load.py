"""
Cognitive Load Analysis - Mental demands of answering a question.

Assesses:
- Steps required to answer
- Working memory demands
- Abstraction level
- Age-appropriate thresholds
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI


# ── Age-Appropriate Cognitive Thresholds ────────────────────────────────────
# Based on cognitive development research

COGNITIVE_THRESHOLDS = {
    # Grade: (max_steps, max_working_memory, can_handle_abstraction)
    1: (1, 2, False),
    2: (2, 2, False),
    3: (2, 3, False),
    4: (2, 3, True),   # Can start handling simple abstractions
    5: (3, 3, True),
    6: (3, 4, True),
    7: (4, 4, True),
    8: (4, 5, True),
    9: (5, 5, True),
    10: (5, 6, True),
    11: (6, 6, True),
    12: (6, 7, True),
}


@dataclass
class CognitiveStep:
    """A cognitive step required to answer."""

    action: str  # "recall", "compare", "apply", "analyze", "evaluate"
    description: str

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "description": self.description,
        }


@dataclass
class CognitiveLoadResult:
    """Result of cognitive load analysis."""

    steps: List[CognitiveStep]
    step_count: int
    working_memory_items: List[str]
    working_memory_count: int
    abstraction_level: str  # "concrete", "semi-abstract", "abstract"
    requires_multi_step_reasoning: bool
    requires_inference: bool

    # Grade comparison
    target_grade: int
    max_steps_for_grade: int
    max_memory_for_grade: int
    steps_verdict: str  # "appropriate", "challenging", "too_complex"
    memory_verdict: str
    overall_verdict: str

    def to_dict(self) -> dict:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "step_count": self.step_count,
            "working_memory_items": self.working_memory_items,
            "working_memory_count": self.working_memory_count,
            "abstraction_level": self.abstraction_level,
            "requires_multi_step_reasoning": self.requires_multi_step_reasoning,
            "requires_inference": self.requires_inference,
            "target_grade": self.target_grade,
            "thresholds": {
                "max_steps": self.max_steps_for_grade,
                "max_memory": self.max_memory_for_grade,
            },
            "verdicts": {
                "steps": self.steps_verdict,
                "memory": self.memory_verdict,
                "overall": self.overall_verdict,
            },
        }


class CognitiveLoadAnalyzer:
    """Analyzes cognitive demands of educational questions."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.model = model

        self.client: Optional[AsyncOpenAI] = None
        if self.api_key:
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    async def analyze(
        self,
        content: Dict[str, Any],
        target_grade: int,
    ) -> CognitiveLoadResult:
        """
        Analyze cognitive load of answering a question.

        Args:
            content: Question with text, options
            target_grade: Target grade level

        Returns:
            CognitiveLoadResult with cognitive demands
        """
        # Get thresholds for grade
        thresholds = COGNITIVE_THRESHOLDS.get(target_grade, (4, 4, True))
        max_steps, max_memory, can_abstract = thresholds

        if self.client:
            return await self._llm_analyze(content, target_grade, max_steps, max_memory, can_abstract)

        # Fallback: heuristic analysis
        return self._heuristic_analyze(content, target_grade, max_steps, max_memory, can_abstract)

    def _heuristic_analyze(
        self,
        content: Dict[str, Any],
        target_grade: int,
        max_steps: int,
        max_memory: int,
        can_abstract: bool,
    ) -> CognitiveLoadResult:
        """Basic heuristic analysis when LLM unavailable."""
        text = content.get("text") or content.get("question", "")
        options = content.get("options", [])

        # Estimate steps based on question complexity
        steps = [CognitiveStep("recall", "Remember relevant information")]

        # Check for comparison words
        if any(w in text.lower() for w in ["compare", "contrast", "different", "similar"]):
            steps.append(CognitiveStep("compare", "Compare multiple concepts"))

        # Check for application
        if any(w in text.lower() for w in ["apply", "use", "solve", "calculate"]):
            steps.append(CognitiveStep("apply", "Apply knowledge to solve"))

        # Check for analysis
        if any(w in text.lower() for w in ["analyze", "explain", "why", "how"]):
            steps.append(CognitiveStep("analyze", "Analyze relationship or cause"))

        # Check for evaluation
        if any(w in text.lower() for w in ["evaluate", "best", "most", "should"]):
            steps.append(CognitiveStep("evaluate", "Evaluate and select best option"))

        # Working memory: question + options
        memory_items = ["question prompt"]
        memory_items.extend([f"option {i+1}" for i in range(min(len(options), 4))])

        # Abstraction level
        abstract_words = ["concept", "principle", "theory", "relationship", "function", "purpose"]
        if any(w in text.lower() for w in abstract_words):
            abstraction = "abstract"
        elif len(text) > 100 or len(options) > 4:
            abstraction = "semi-abstract"
        else:
            abstraction = "concrete"

        # Verdicts
        step_count = len(steps)
        memory_count = len(memory_items)

        if step_count <= max_steps:
            steps_verdict = "appropriate"
        elif step_count <= max_steps + 1:
            steps_verdict = "challenging"
        else:
            steps_verdict = "too_complex"

        if memory_count <= max_memory:
            memory_verdict = "appropriate"
        elif memory_count <= max_memory + 2:
            memory_verdict = "challenging"
        else:
            memory_verdict = "too_complex"

        # Overall
        if steps_verdict == "too_complex" or memory_verdict == "too_complex":
            overall = "too_complex"
        elif steps_verdict == "challenging" or memory_verdict == "challenging":
            overall = "challenging"
        else:
            overall = "appropriate"

        return CognitiveLoadResult(
            steps=steps,
            step_count=step_count,
            working_memory_items=memory_items,
            working_memory_count=memory_count,
            abstraction_level=abstraction,
            requires_multi_step_reasoning=step_count > 2,
            requires_inference=any(s.action == "analyze" for s in steps),
            target_grade=target_grade,
            max_steps_for_grade=max_steps,
            max_memory_for_grade=max_memory,
            steps_verdict=steps_verdict,
            memory_verdict=memory_verdict,
            overall_verdict=overall,
        )

    async def _llm_analyze(
        self,
        content: Dict[str, Any],
        target_grade: int,
        max_steps: int,
        max_memory: int,
        can_abstract: bool,
    ) -> CognitiveLoadResult:
        """Use LLM for detailed cognitive analysis."""
        text = content.get("text") or content.get("question", "")
        options = content.get("options", [])
        options_text = "\n".join(f"- {o}" for o in options)

        prompt = f"""Analyze the cognitive demands of answering this question.

Question: {text}

Options:
{options_text}

Target grade: {target_grade}
Max steps for this grade: {max_steps}
Max working memory items: {max_memory}

Determine:
1. What cognitive steps are needed to answer? (recall, compare, apply, analyze, evaluate)
2. What items must be held in working memory simultaneously?
3. Is this concrete (observable things) or abstract (concepts, relationships)?

Return JSON:
{{
  "steps": [
    {{"action": "recall", "description": "what specific thing to recall"}},
    {{"action": "compare", "description": "what to compare"}}
  ],
  "working_memory_items": ["item 1", "item 2"],
  "abstraction_level": "concrete|semi-abstract|abstract",
  "requires_inference": true/false
}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are analyzing cognitive demands of educational questions. "
                            "Be specific about the mental steps required. "
                            "Respond only with valid JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=500,
            )

            result_text = response.choices[0].message.content.strip()

            # Handle markdown code blocks
            if "```" in result_text:
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]

            analysis = json.loads(result_text)

            steps = [
                CognitiveStep(s["action"], s["description"])
                for s in analysis.get("steps", [])
            ]
            memory_items = analysis.get("working_memory_items", [])
            abstraction = analysis.get("abstraction_level", "semi-abstract")
            requires_inference = analysis.get("requires_inference", False)

            step_count = len(steps)
            memory_count = len(memory_items)

            # Verdicts
            if step_count <= max_steps:
                steps_verdict = "appropriate"
            elif step_count <= max_steps + 1:
                steps_verdict = "challenging"
            else:
                steps_verdict = "too_complex"

            if memory_count <= max_memory:
                memory_verdict = "appropriate"
            elif memory_count <= max_memory + 2:
                memory_verdict = "challenging"
            else:
                memory_verdict = "too_complex"

            # Check abstraction
            if abstraction == "abstract" and not can_abstract:
                overall = "too_complex"
            elif steps_verdict == "too_complex" or memory_verdict == "too_complex":
                overall = "too_complex"
            elif steps_verdict == "challenging" or memory_verdict == "challenging":
                overall = "challenging"
            else:
                overall = "appropriate"

            return CognitiveLoadResult(
                steps=steps,
                step_count=step_count,
                working_memory_items=memory_items,
                working_memory_count=memory_count,
                abstraction_level=abstraction,
                requires_multi_step_reasoning=step_count > 2,
                requires_inference=requires_inference,
                target_grade=target_grade,
                max_steps_for_grade=max_steps,
                max_memory_for_grade=max_memory,
                steps_verdict=steps_verdict,
                memory_verdict=memory_verdict,
                overall_verdict=overall,
            )

        except Exception:
            return self._heuristic_analyze(content, target_grade, max_steps, max_memory, can_abstract)


async def analyze_cognitive_load(
    content: Dict[str, Any],
    target_grade: int,
    api_key: Optional[str] = None,
) -> CognitiveLoadResult:
    """Convenience function for cognitive load analysis."""
    analyzer = CognitiveLoadAnalyzer(api_key=api_key)
    return await analyzer.analyze(content, target_grade)
