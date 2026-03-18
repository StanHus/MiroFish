"""
MiroFish Simulator - Multi-agent student simulation for educational content.

The problem with traditional LLM simulation: LLMs "cheat" - they know answers even
when told they don't. You cannot make an LLM "not know" something through prompting.

The solution: MISCONCEPTION MATCHING, not knowledge suppression.

== BATCH-LEVEL SWARM INTELLIGENCE ==

The BatchHealthAnalyzer provides cost-efficient batch analysis:
- FREE heuristics detect patterns without any LLM calls
- ONE LLM call analyzes the entire batch holistically
- Output: which questions need attention + generator feedback

    from mirofish_simulator import BatchHealthAnalyzer

    analyzer = BatchHealthAnalyzer()
    report = await analyzer.analyze(
        questions=[q1, q2, q3, ...],
        curriculum_context={"standards": ["CCSS.MATH.3.OA.A.1", ...]}
    )

    print(report.questions_needing_attention)  # Only these need deep analysis
    print(report.generator_feedback)           # Actionable feedback for generator

Cost model:
    Current per-question:  N questions × 5 LLM calls = 5N calls
    BatchHealthAnalyzer:   Heuristics (free) + 1 call + selective deep dives

== PER-QUESTION SIMULATION ==

Architecture:
┌─────────────────────────────────────────────────────────────────────┐
│                    AgenticOrchestrator                               │
│                                                                      │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐            │
│  │  DISTRACTOR  │   │   STUDENT    │   │   SELECTOR   │            │
│  │    AGENT     │──▶│    MODEL     │──▶│    AGENT     │            │
│  │              │   │    AGENT     │   │              │            │
│  │ "What error  │   │ "What does   │   │ "Match       │            │
│  │  leads to    │   │  this student│   │  misconception│           │
│  │  each wrong  │   │  believe?"   │   │  to answer"  │            │
│  │  answer?"    │   │              │   │              │            │
│  └──────────────┘   └──────────────┘   └──────────────┘            │
└─────────────────────────────────────────────────────────────────────┘

Usage:
    from mirofish_simulator import AgenticOrchestrator

    orchestrator = AgenticOrchestrator()

    # Single student simulation
    result = await orchestrator.simulate(
        question={"text": "...", "options": [...]},
        correct_answer="B",
        grade=5,
        archetype="class_clown",
    )
    print(result.selected)                     # What they picked
    print(result.is_correct)                   # Right/wrong
    print(result.student_model.beliefs)        # What they believe
    print(result.student_model.misconceptions) # Their misconceptions
    print(result.selection_result.selection_reason)  # Why they picked this

    # Batch simulation (efficient - distractor analysis done once)
    results = await orchestrator.simulate_batch(
        question=question,
        correct_answer="B",
        students=[
            {"grade": 5, "archetype": "average_student"},
            {"grade": 8, "archetype": "honors_overachiever"},
            {"grade": 11, "archetype": "class_clown"},
        ]
    )

    # Accessibility analysis (deterministic, no LLM)
    from mirofish_simulator import AccessibilityAnalyzer
    analyzer = AccessibilityAnalyzer()
    result = await analyzer.analyze(content, target_grade=5)
    print(f"Reading Level: Grade {result.reading_level.flesch_kincaid_grade}")
"""

# ── Multi-Agent Simulation ────────────────────────────────────────────────────
from .agents.v2 import (
    AgenticOrchestrator,
    AgenticSimulationResult,
    DistractorAgent,
    DistractorAnalysis,
    DistractorMapping,
    StudentModelAgent,
    StudentModel,
    SelectorAgent,
    SelectionResult,
    ExperienceAgent,
    ExperienceResult,
    ExperienceAssessment,
    assess_question_experience,
    DIVERSE_STUDENTS,
    StudentSimulator,
    SimulationResult as V2SimulationResult,
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

# ── Accessibility Analysis ──────────────────────────────────────────────────
from .accessibility import (
    AccessibilityAnalyzer,
    AccessibilityResult,
    AccessibilityVerdict,
    ReadingLevelAnalyzer,
    ReadingLevelResult,
    VocabularyAnalyzer,
    VocabularyIssue,
    PriorKnowledgeAnalyzer,
    PriorKnowledgeResult,
    CognitiveLoadAnalyzer,
    CognitiveLoadResult,
    RecommendationEngine,
    Recommendation,
    flesch_kincaid_grade,
    analyze_accessibility,
)

# ── Core Simulation ──────────────────────────────────────────────────────────
from .simulator import (
    Simulator,
    SimulationResult,
    StudentResponse,
    ArchetypePerformance,
    DistractorAnalysis,
    simulate_content,
    ARCHETYPES,
    DEFAULT_POPULATION,
)

from .profiles import StudentProfile, ARCHETYPE_TRAITS

from .misconceptions import (
    MisconceptionAnalyzer,
    MisconceptionAnalysisResult,
    DistractorMisconception,
    analyze_misconceptions,
)

from .taxonomies import (
    get_taxonomy,
    get_misconception,
    SUBJECT_TAXONOMIES,
)

from .comparative import (
    ComparativeAnalyzer,
    ComparativeAnalysisResult,
    analyze_quiz,
)

from .batch_health import (
    BatchHealthAnalyzer,
    BatchHealthReport,
    PatternFindings,
    analyze_batch,
)

from .adversarial import AdversarialSwarm
from .memory import AgentMemory

# Compatibility layer for InceptBench
from .compat import (
    StructuralAnalyzer,
    parse_question,
)

from .cognition import (
    CognitiveModel,
    CognitiveLens,
    RetentionModel,
    RetentionContext,
    PerceptionModel,
    PerceivedQuestion,
    create_cognitive_lens,
)

__version__ = "0.10.1"  # Added StructuralAnalyzer, parse_question for InceptBench compat

__all__ = [
    # Agentic Simulation (v2 - RECOMMENDED)
    "AgenticOrchestrator",
    "AgenticSimulationResult",
    "DistractorAgent",
    "DistractorAnalysis",
    "DistractorMapping",
    "StudentModelAgent",
    "StudentModel",
    "SelectorAgent",
    "SelectionResult",
    # Experience Assessment
    "ExperienceAgent",
    "ExperienceResult",
    "ExperienceAssessment",
    "assess_question_experience",
    "DIVERSE_STUDENTS",
    # Multi-Agent Pipeline
    "StudentSimulator",
    "V2SimulationResult",
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
    # Accessibility analysis
    "AccessibilityAnalyzer",
    "AccessibilityResult",
    "AccessibilityVerdict",
    "ReadingLevelAnalyzer",
    "ReadingLevelResult",
    "VocabularyAnalyzer",
    "VocabularyIssue",
    "PriorKnowledgeAnalyzer",
    "PriorKnowledgeResult",
    "CognitiveLoadAnalyzer",
    "CognitiveLoadResult",
    "RecommendationEngine",
    "Recommendation",
    "flesch_kincaid_grade",
    "analyze_accessibility",
    # Core simulation
    "Simulator",
    "SimulationResult",
    "StudentResponse",
    "StudentProfile",
    "ArchetypePerformance",
    "DistractorAnalysis",
    "simulate_content",
    "ARCHETYPES",
    "ARCHETYPE_TRAITS",
    "DEFAULT_POPULATION",
    # Cognitive modeling
    "CognitiveModel",
    "CognitiveLens",
    "RetentionModel",
    "RetentionContext",
    "PerceptionModel",
    "PerceivedQuestion",
    "create_cognitive_lens",
    # Misconceptions
    "MisconceptionAnalyzer",
    "MisconceptionAnalysisResult",
    "DistractorMisconception",
    "analyze_misconceptions",
    # Taxonomies
    "get_taxonomy",
    "get_misconception",
    "SUBJECT_TAXONOMIES",
    # Comparative analysis
    "ComparativeAnalyzer",
    "ComparativeAnalysisResult",
    "analyze_quiz",
    # Batch health (swarm intelligence)
    "BatchHealthAnalyzer",
    "BatchHealthReport",
    "PatternFindings",
    "analyze_batch",
    # Adversarial testing
    "AdversarialSwarm",
    # Agent memory
    "AgentMemory",
    # Compatibility (InceptBench)
    "StructuralAnalyzer",
    "parse_question",
]
