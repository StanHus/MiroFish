"""
MiroFish Agent-Based Simulation (v2 Agentic Architecture).

The core insight: LLMs know everything, but students don't.
Instead of suppressing LLM knowledge (impossible), we use a multi-agent
pipeline that matches student misconceptions to wrong answers.

Usage:
    from mirofish_simulator import AgenticOrchestrator

    orchestrator = AgenticOrchestrator()
    result = await orchestrator.simulate(
        question={"text": "...", "options": [...]},
        correct_answer="B",
        grade=5,
        archetype="class_clown",
    )
"""

from .v2 import (
    AgenticOrchestrator,
    AgenticSimulationResult,
    DistractorAgent,
    DistractorAnalysis,
    DistractorMapping,
    StudentModelAgent,
    StudentModel,
    SelectorAgent,
    SelectionResult,
    StudentSimulator,
    SimulationResult,
    simulate_student,
    simulate_classroom,
    KnowledgeAgent,
    KnowledgeProfile,
    PerceptionAgent,
    PerceptionResult,
    AnswerAgent,
    AnswerResult,
    VerifierAgent,
    VerificationResult,
)

__all__ = [
    # Agentic orchestrator (recommended)
    "AgenticOrchestrator",
    "AgenticSimulationResult",
    # Agents
    "DistractorAgent",
    "DistractorAnalysis",
    "DistractorMapping",
    "StudentModelAgent",
    "StudentModel",
    "SelectorAgent",
    "SelectionResult",
    "KnowledgeAgent",
    "KnowledgeProfile",
    "PerceptionAgent",
    "PerceptionResult",
    "AnswerAgent",
    "AnswerResult",
    "VerifierAgent",
    "VerificationResult",
    # Convenience
    "StudentSimulator",
    "SimulationResult",
    "simulate_student",
    "simulate_classroom",
]
