"""
MiroFish Agent System v2 - Truly Agentic Student Simulation.

The problem with v1: LLMs "cheat" - they know answers despite constraints.
You cannot make an LLM "not know" something through prompting.

The solution: MISCONCEPTION MATCHING, not knowledge suppression.

NEW ARCHITECTURE (Agentic):
┌─────────────────────────────────────────────────────────────────────┐
│                    AGENTIC ORCHESTRATOR                              │
│                                                                      │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐            │
│  │  DISTRACTOR  │   │   STUDENT    │   │   SELECTOR   │            │
│  │    AGENT     │──▶│    MODEL     │──▶│    AGENT     │            │
│  │              │   │    AGENT     │   │              │            │
│  │ "What error  │   │              │   │ "Match       │            │
│  │  leads to    │   │ "What does   │   │  misconception│           │
│  │  each wrong  │   │  this student│   │  to answer"  │            │
│  │  answer?"    │   │  believe?"   │   │              │            │
│  └──────────────┘   └──────────────┘   └──────────────┘            │
└─────────────────────────────────────────────────────────────────────┘

Key insight: We're doing a MATCHING operation.
- DistractorAgent: What misconception → what answer?
- StudentModelAgent: What misconceptions does this student have?
- SelectorAgent: Match student's misconceptions → answer

This works because we're NOT asking the LLM to "not know" things.
We're asking it to MATCH - which LLMs ARE good at.

LEGACY ARCHITECTURE (still available):
- KnowledgeAgent + PerceptionAgent + AnswerAgent + VerifierAgent
- Works for some cases, but LLM often "cheats"
"""

# New agentic architecture (recommended)
from .agentic_orchestrator import AgenticOrchestrator, AgenticSimulationResult
from .distractor_agent import DistractorAgent, DistractorAnalysis, DistractorMapping
from .student_model_agent import StudentModelAgent, StudentModel
from .selector_agent import SelectorAgent, SelectionResult
from .experience_agent import (
    ExperienceAgent,
    ExperienceResult,
    ExperienceAssessment,
    assess_question_experience,
    DIVERSE_STUDENTS,
)

# Legacy architecture (still functional)
from .orchestrator import (
    StudentSimulator,
    SimulationResult,
    simulate_student,
    simulate_classroom,
)
from .knowledge_agent import KnowledgeAgent, KnowledgeProfile
from .perception_agent import PerceptionAgent, PerceptionResult
from .answer_agent import AnswerAgent, AnswerResult
from .verifier_agent import VerifierAgent, VerificationResult

__all__ = [
    # NEW: Agentic architecture (recommended)
    "AgenticOrchestrator",
    "AgenticSimulationResult",
    "DistractorAgent",
    "DistractorAnalysis",
    "DistractorMapping",
    "StudentModelAgent",
    "StudentModel",
    "SelectorAgent",
    "SelectionResult",
    "ExperienceAgent",
    "ExperienceResult",
    "ExperienceAssessment",
    "assess_question_experience",
    "DIVERSE_STUDENTS",
    # LEGACY: Original architecture
    "StudentSimulator",
    "SimulationResult",
    "simulate_student",
    "simulate_classroom",
    "KnowledgeAgent",
    "KnowledgeProfile",
    "PerceptionAgent",
    "PerceptionResult",
    "AnswerAgent",
    "AnswerResult",
    "VerifierAgent",
    "VerificationResult",
]
