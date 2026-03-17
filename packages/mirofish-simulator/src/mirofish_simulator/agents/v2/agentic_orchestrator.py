"""
Agentic Orchestrator - Fully agent-driven student simulation.

This orchestrator uses a pipeline of specialized agents that work together:

1. DistractorAgent - Analyzes what misconceptions lead to each wrong answer
2. StudentModelAgent - Models what this specific student believes/misconceives
3. SelectorAgent - Matches student misconceptions to distractor answers

The key insight: We're not asking an LLM to "not know" things (impossible).
We're doing a MATCHING operation: student misconceptions ↔ distractor misconceptions

This approach produces realistic wrong answers because we're leveraging what
LLMs ARE good at (analysis, matching) instead of what they CAN'T do (unknowing).
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .distractor_agent import DistractorAgent, DistractorAnalysis
from .student_model_agent import StudentModelAgent, StudentModel
from .selector_agent import SelectorAgent, SelectionResult


@dataclass
class AgenticSimulationResult:
    """Complete result from the agentic simulation pipeline."""

    # Core output
    selected: str  # "A", "B", "C", "D"
    selected_text: str
    is_correct: bool

    # Agent outputs (for transparency)
    distractor_analysis: DistractorAnalysis
    student_model: StudentModel
    selection_result: SelectionResult

    # Metadata
    grade: int
    archetype: str
    question_id: Optional[str]

    def to_dict(self) -> dict:
        return {
            "selected": self.selected,
            "selected_text": self.selected_text,
            "is_correct": self.is_correct,
            "grade": self.grade,
            "archetype": self.archetype,
            "question_id": self.question_id,
            "distractor_analysis": self.distractor_analysis.to_dict(),
            "student_model": self.student_model.to_dict(),
            "selection_result": self.selection_result.to_dict(),
        }

    def summary(self) -> str:
        """Human-readable summary of the simulation."""
        lines = [
            f"=== Grade {self.grade} {self.archetype} ===",
            f"Selected: {self.selected}) {self.selected_text}",
            f"Correct: {'✓' if self.is_correct else '✗'}",
            "",
            f"Reason: {self.selection_result.selection_reason}",
        ]

        if self.selection_result.misconception_matched:
            lines.append(f"Misconception: {self.selection_result.misconception_matched}")

        lines.append(f"Confidence: {self.selection_result.confidence}")

        return "\n".join(lines)


class AgenticOrchestrator:
    """
    Orchestrates the fully agentic student simulation.

    Pipeline:
    1. DistractorAgent analyzes the question to map misconceptions → answers
    2. StudentModelAgent models what a specific student believes
    3. SelectorAgent matches the student's misconceptions to an answer

    This is NOT about constraining knowledge. It's about:
    - What misconceptions EXIST for this question?
    - What misconceptions does THIS STUDENT have?
    - Which answer matches their misconceptions?
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ):
        import os
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.model = model

        # Initialize agents
        self.distractor_agent = DistractorAgent(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
        )
        self.student_model_agent = StudentModelAgent(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
        )
        self.selector_agent = SelectorAgent(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
        )

    async def simulate(
        self,
        question: Dict[str, Any],
        correct_answer: str,
        grade: int,
        archetype: str,
    ) -> AgenticSimulationResult:
        """
        Simulate how a student would answer a question.

        Args:
            question: Dict with "text"/"question" and "options"
            correct_answer: The correct option ("A", "B", "C", or "D")
            grade: Student grade (1-12)
            archetype: Student type (honors_overachiever, class_clown, etc.)

        Returns:
            AgenticSimulationResult with selected answer and full reasoning
        """
        # Step 1: Analyze distractors - what misconception leads to each wrong answer?
        distractor_analysis = await self.distractor_agent.analyze(
            question=question,
            correct_answer=correct_answer,
        )

        # Step 2: Model the student - what does this student believe/misconceive?
        student_model = await self.student_model_agent.model_student(
            question=question,
            grade=grade,
            archetype=archetype,
        )

        # Step 3: Select answer - match student misconceptions to distractors
        selection = await self.selector_agent.select(
            question=question,
            distractors=distractor_analysis,
            student=student_model,
        )

        return AgenticSimulationResult(
            selected=selection.selected,
            selected_text=selection.selected_text,
            is_correct=selection.is_correct,
            distractor_analysis=distractor_analysis,
            student_model=student_model,
            selection_result=selection,
            grade=grade,
            archetype=archetype,
            question_id=question.get("id"),
        )

    async def simulate_batch(
        self,
        question: Dict[str, Any],
        correct_answer: str,
        students: List[Dict[str, Any]],
    ) -> List[AgenticSimulationResult]:
        """
        Simulate multiple students answering the same question.

        The distractor analysis is done once and reused for efficiency.

        Args:
            question: The question
            correct_answer: Correct option
            students: List of {"grade": int, "archetype": str}

        Returns:
            List of simulation results
        """
        # Step 1: Analyze distractors ONCE (same for all students)
        distractor_analysis = await self.distractor_agent.analyze(
            question=question,
            correct_answer=correct_answer,
        )

        # Step 2 & 3: Model and select for each student (can parallelize)
        async def simulate_one(student: Dict[str, Any]) -> AgenticSimulationResult:
            grade = student.get("grade", 8)
            archetype = student.get("archetype", "average_student")

            student_model = await self.student_model_agent.model_student(
                question=question,
                grade=grade,
                archetype=archetype,
            )

            selection = await self.selector_agent.select(
                question=question,
                distractors=distractor_analysis,
                student=student_model,
            )

            return AgenticSimulationResult(
                selected=selection.selected,
                selected_text=selection.selected_text,
                is_correct=selection.is_correct,
                distractor_analysis=distractor_analysis,
                student_model=student_model,
                selection_result=selection,
                grade=grade,
                archetype=archetype,
                question_id=question.get("id"),
            )

        # Run all simulations concurrently
        results = await asyncio.gather(*[
            simulate_one(student) for student in students
        ])

        return list(results)
