# MiroFish Simulator

**Agentic student simulation** using misconception matching - produces realistic wrong answers without LLM "cheating".

## The Problem

Traditional LLM-based student simulation doesn't work: **LLMs know everything**. When you ask GPT-4 to "act like a 5th grader who doesn't know about electoral votes", it still picks 270 because it can't actually "not know" things.

## The Solution: Misconception Matching

Instead of trying to suppress LLM knowledge (impossible), we use a **multi-agent pipeline** that matches student misconceptions to wrong answers:

```
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
```

### How It Works

1. **DistractorAgent** - Analyzes each wrong answer to identify what misconception leads to it
   - "Option A (218) catches students who confuse electoral votes with House majority"
   - "Option D (435) catches students who confuse with total Congress members"

2. **StudentModelAgent** - Models what a specific student believes and misconceives
   - Grade 5 class_clown: Low familiarity, vague beliefs, common misconceptions
   - Grade 11 honors: High familiarity, specific knowledge, few misconceptions

3. **SelectorAgent** - Matches the student's misconceptions to the appropriate answer
   - If student has misconception matching a distractor → pick that distractor
   - If student has correct belief with high familiarity → pick correct answer

## Quick Start

```python
import asyncio
from mirofish_simulator import AgenticOrchestrator

orchestrator = AgenticOrchestrator()

question = {
    "text": "How many electoral votes are needed to win the presidency?",
    "options": ["218", "270", "300", "435"],
}

async def main():
    # Single student
    result = await orchestrator.simulate(
        question=question,
        correct_answer="B",
        grade=5,
        archetype="class_clown",
    )

    print(f"Selected: {result.selected}")           # "C" (wrong!)
    print(f"Correct: {result.is_correct}")          # False
    print(f"Familiarity: {result.student_model.topic_familiarity:.0%}")  # 40%
    print(f"Misconception: {result.selection_result.misconception_matched}")

asyncio.run(main())
```

### Batch Simulation

```python
# Efficient - distractor analysis done once, reused for all students
results = await orchestrator.simulate_batch(
    question=question,
    correct_answer="B",
    students=[
        {"grade": 5, "archetype": "class_clown"},
        {"grade": 8, "archetype": "average_student"},
        {"grade": 11, "archetype": "honors_overachiever"},
    ]
)

for r in results:
    status = "✓" if r.is_correct else "✗"
    print(f"Grade {r.grade} {r.archetype}: {r.selected}) {status}")
```

## Realistic Results

The system produces realistic differentiation:

| Student | Electoral Votes (hard) | Branches of Gov (easy) |
|---------|----------------------|----------------------|
| Grade 5 class_clown | ✗ | ✓ |
| Grade 8 average | ✗ | ✓ |
| Grade 11 honors | ✓ | ✓ |

- **Easy questions**: All students get them right (basic civics)
- **Hard factual questions**: Only students with specific knowledge get them right
- **Archetypes matter**: class_clown (low familiarity) gets more wrong than honors

## Agent Details

### DistractorAgent

Maps each wrong answer to the misconception that leads to it.

```python
from mirofish_simulator import DistractorAgent

agent = DistractorAgent()
analysis = await agent.analyze(question, correct_answer="B")

for mapping in analysis.mappings:
    if not mapping.is_correct:
        print(f"{mapping.option}) {mapping.option_text}")
        print(f"   Misconception: {mapping.leads_from_misconception}")
        print(f"   Grade appeal: {mapping.grade_level_appeal}")
```

### StudentModelAgent

Models what a student believes (correct and incorrect).

```python
from mirofish_simulator import StudentModelAgent

agent = StudentModelAgent()
student = await agent.model_student(question, grade=5, archetype="class_clown")

print(f"Beliefs: {student.beliefs}")
print(f"Misconceptions: {student.misconceptions}")
print(f"Topic familiarity: {student.topic_familiarity:.0%}")
print(f"Guesses when unsure: {student.guesses_when_unsure}")
```

### SelectorAgent

Matches student misconceptions to answers.

```python
from mirofish_simulator import SelectorAgent

agent = SelectorAgent()
selection = await agent.select(question, distractor_analysis, student_model)

print(f"Selected: {selection.selected}")
print(f"Reason: {selection.selection_reason}")
print(f"Misconception matched: {selection.misconception_matched}")
```

## Archetypes

| Archetype | Familiarity | Behavior |
|-----------|-------------|----------|
| `honors_overachiever` | High (80%+) | Specific knowledge, confident |
| `average_student` | Medium (60-70%) | Taught content, some gaps |
| `class_clown` | Low (40%) | Minimal attention, guesses |
| `esl_student` | Medium | Core concepts solid, vocabulary issues |
| `disengaged_but_smart` | Variable | Has ability, inconsistent |
| `quiet_thinker` | Medium | Second-guesses self |
| `debate_club_kid` | High in interests | Good at arguments |

## Installation

```bash
pip install mirofish-simulator
```

Or from source:
```bash
cd packages/mirofish-simulator
pip install -e .
```

## Environment

```bash
export OPENAI_API_KEY="sk-..."
```

## API Reference

### AgenticOrchestrator

```python
orchestrator = AgenticOrchestrator(
    api_key: str = None,      # Uses OPENAI_API_KEY env var if not provided
    base_url: str = None,     # Custom API base URL
    model: str = "gpt-4o-mini",
)

# Single simulation
result = await orchestrator.simulate(
    question: dict,           # {"text": "...", "options": [...]}
    correct_answer: str,      # "A", "B", "C", or "D"
    grade: int,               # 1-12
    archetype: str,           # See archetypes above
)

# Batch simulation (efficient)
results = await orchestrator.simulate_batch(
    question: dict,
    correct_answer: str,
    students: list,           # [{"grade": 5, "archetype": "..."}, ...]
)
```

### AgenticSimulationResult

```python
result.selected              # "A", "B", "C", "D"
result.selected_text         # The full answer text
result.is_correct            # True/False
result.grade                 # Student grade
result.archetype             # Student archetype

# Agent outputs
result.distractor_analysis   # DistractorAnalysis
result.student_model         # StudentModel
result.selection_result      # SelectionResult

# Methods
result.to_dict()             # Full dict representation
result.summary()             # Human-readable summary
```

## Accessibility Analysis (Static)

For deterministic content analysis without LLM:

```python
from mirofish_simulator import AccessibilityAnalyzer

analyzer = AccessibilityAnalyzer()
result = await analyzer.analyze(content, target_grade=5)

print(f"Reading Level: Grade {result.reading_level.flesch_kincaid_grade}")
print(f"Vocabulary Issues: {len(result.vocabulary.issues)}")
```

## Version History

### v0.8.0 (Current)
- **Agentic misconception-matching architecture**
- DistractorAgent, StudentModelAgent, SelectorAgent
- Factual vs conceptual question handling
- Realistic wrong answers without LLM cheating

### v0.7.0
- Multi-agent with verification (deprecated approach)

### v0.6.0
- Single agent simulation

## License

MIT
