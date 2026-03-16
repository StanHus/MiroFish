# MiroFish + InceptBench Integration Plan

## Executive Summary

**MiroFish** is a multi-agent simulation engine that creates digital worlds with thousands of AI agents (including student personas). **InceptBench** is an educational content evaluation framework that scores questions, quizzes, and readings on quality metrics.

**The Integration**: Use MiroFish's student simulation capabilities to enhance InceptBench's evaluation with realistic student behavior modeling.

---

## Current State Analysis

### InceptBench Has:
- LLM-based content classification (question, quiz, reading, etc.)
- 8-11 quality metrics per content type
- Hierarchical evaluation (quiz → questions)
- Curriculum-aware evaluation via vector search
- REST API + CLI + Python SDK

### MiroFish Has:
- `StudentProfileGenerator` - creates diverse student agents with:
  - 8 archetypes (honors_overachiever, debate_club_kid, quiet_thinker, etc.)
  - Demographics, MBTI, learning styles, GPA ranges, engagement levels
  - LLM-enriched personas
- `EvaluationPipeline` with:
  - Question QC (LLM-based)
  - Explanation QC (LLM-based)
  - **Psychometric Simulation (IRT-based item analysis)**
- `SimulationRunner` - multi-agent social simulation with:
  - Agent actions and interactions
  - Round-by-round tracking
  - Zep memory integration
  - Report generation

---

## Integration Opportunities

### 1. Student Response Simulation (Primary)
**Goal**: Simulate how real students would interact with educational content.

**How it works**:
1. InceptBench sends content to MiroFish
2. MiroFish generates diverse student population (e.g., 30 students across archetypes)
3. Each student agent "attempts" the content based on their profile
4. Returns simulated responses, engagement metrics, confusion points

**Value**:
- Predict difficulty more accurately than static LLM estimation
- Identify content that only works for certain student types
- Find accessibility issues

### 2. Psychometric Enhancement
**Goal**: Use MiroFish's IRT simulation for InceptBench's difficulty scoring.

**Current**: InceptBench estimates difficulty via LLM
**Enhanced**: MiroFish simulates N students → computes IRT parameters → returns to InceptBench

**Metrics to add**:
- Item difficulty (IRT `b` parameter)
- Item discrimination (IRT `a` parameter)
- Distractor effectiveness
- DIF (Differential Item Functioning) across demographics

### 3. Engagement Prediction
**Goal**: Predict how engaging content will be before deployment.

**Simulation approach**:
1. Generate classroom of 25-40 student agents
2. Present the content
3. Simulate discussion/interaction
4. Measure: participation rates, question quality, off-topic drift

### 4. Content A/B Testing
**Goal**: Compare content variants in simulation.

**Example**:
- Version A: "What year did X happen?"
- Version B: "Why did X lead to Y?"

Run both through student simulation → compare learning outcomes.

---

## Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       InceptBench                           │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │   CLI/API   │  │  Evaluators  │  │ Curriculum Search │  │
│  └──────┬──────┘  └──────┬───────┘  └─────────┬─────────┘  │
│         │                │                    │             │
│         └────────────────┼────────────────────┘             │
│                          │                                  │
│                 ┌────────▼────────┐                         │
│                 │ MiroFish Bridge │ ◄── NEW                 │
│                 └────────┬────────┘                         │
└──────────────────────────┼──────────────────────────────────┘
                           │
                           │ HTTP/gRPC
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                        MiroFish                             │
│  ┌──────────────────┐  ┌─────────────────────────────────┐ │
│  │ Student Profile  │  │     Simulation Runner           │ │
│  │   Generator      │  │  (OASIS multi-agent engine)     │ │
│  └────────┬─────────┘  └──────────────┬──────────────────┘ │
│           │                           │                     │
│  ┌────────▼───────────────────────────▼──────────────────┐ │
│  │              Evaluation Pipeline                       │ │
│  │   • Question QC  • Explanation QC  • Psychometrics    │ │
│  └───────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: API Bridge (Week 1-2)

**MiroFish side** — new endpoint:
```python
# backend/app/api/inceptbench_routes.py

@bp.route('/api/inceptbench/simulate', methods=['POST'])
def simulate_student_response():
    """
    Simulate student population responding to educational content.

    Input:
      - content: the educational content (question, quiz, reading)
      - population_config: student population parameters
      - simulation_rounds: how many interaction rounds

    Output:
      - simulated_responses: per-student responses
      - aggregate_metrics: difficulty, discrimination, engagement
      - demographic_breakdown: performance by student archetype
    """
```

**InceptBench side** — new evaluator:
```python
# inceptbench_new/evaluators/simulation_evaluator.py

class SimulationEvaluator:
    """Uses MiroFish to simulate student interactions with content."""

    async def evaluate(self, content: dict) -> SimulationResult:
        # 1. Send content to MiroFish
        # 2. Get simulated responses
        # 3. Convert to InceptBench metrics
```

### Phase 2: Psychometric Integration (Week 3)

Move MiroFish's psychometric module to be callable from InceptBench:
- IRT difficulty estimation
- Discrimination analysis
- Distractor effectiveness

### Phase 3: Full Classroom Simulation (Week 4+)

Full OASIS-powered classroom simulation:
- Discussion threads
- Student-to-student interaction
- Teacher intervention modeling
- Long-term learning outcomes

---

## API Contract (Draft)

### Request: Simulate Content
```json
POST /api/inceptbench/simulate
{
  "content": {
    "type": "question",
    "text": "What is the primary function of the Electoral College?",
    "options": ["A) Direct democracy", "B) Elect president", "C) Pass laws", "D) Appoint judges"],
    "correct_answer": "B",
    "grade": "11",
    "subject": "AP Government"
  },
  "population": {
    "size": 30,
    "archetypes": ["honors_overachiever", "debate_club_kid", "quiet_thinker", "disengaged_but_smart"],
    "demographic_distribution": "balanced"
  },
  "simulation": {
    "rounds": 1,
    "include_discussion": false
  }
}
```

### Response: Simulation Results
```json
{
  "aggregate": {
    "accuracy": 0.73,
    "difficulty_irt": 0.42,
    "discrimination_irt": 1.8,
    "avg_time_seconds": 45,
    "engagement_score": 0.81
  },
  "by_archetype": {
    "honors_overachiever": {"accuracy": 0.95, "avg_time": 30},
    "debate_club_kid": {"accuracy": 0.80, "avg_time": 40},
    "quiet_thinker": {"accuracy": 0.70, "avg_time": 55},
    "disengaged_but_smart": {"accuracy": 0.50, "avg_time": 25}
  },
  "distractor_analysis": {
    "A": {"selection_rate": 0.10, "attracted_archetypes": ["disengaged_but_smart"]},
    "C": {"selection_rate": 0.12, "attracted_archetypes": ["esl_student"]},
    "D": {"selection_rate": 0.05, "attracted_archetypes": []}
  },
  "concerns": [
    "High difficulty variance across archetypes (σ=0.18) — may indicate accessibility issue",
    "Distractor A attracts disengaged students — check for misleading wording"
  ]
}
```

---

## Files to Create/Modify

### MiroFish (this repo)
```
backend/
├── app/
│   ├── api/
│   │   └── inceptbench_routes.py    # NEW - API bridge
│   ├── services/
│   │   └── inceptbench_simulator.py # NEW - simulation orchestrator
│   └── evaluator/
│       └── modules/
│           └── student_response.py  # NEW - student response model
└── configs/
    └── inceptbench_defaults.json    # NEW - default population configs
```

### InceptBench (separate repo)
```
inceptbench_new/
├── evaluators/
│   └── simulation_evaluator.py      # NEW - MiroFish integration
├── services/
│   └── mirofish_client.py           # NEW - API client
└── cli/
    └── commands.py                   # MODIFY - add --simulate flag
```

---

## Success Metrics

1. **Accuracy correlation**: Simulated difficulty should correlate (r > 0.7) with real student performance data
2. **Coverage**: Identify accessibility issues in 90%+ of content before deployment
3. **Speed**: Full simulation (30 students) completes in < 10 seconds
4. **Integration**: Single CLI command: `inceptbench evaluate content.json --simulate`

---

## Next Steps

1. [ ] Create `/api/inceptbench/simulate` endpoint in MiroFish
2. [ ] Build `InceptBenchSimulator` service class
3. [ ] Create MiroFish client in InceptBench
4. [ ] Add `--simulate` flag to InceptBench CLI
5. [ ] Test with sample AP Government questions
6. [ ] Validate against real student performance data
