#!/usr/bin/env python3
"""
AP Item Validator — Psychometric Simulation Engine

Takes AP-style MCQ questions, runs them against calibrated student profiles,
and produces standardized item analysis metrics:

  - Difficulty Index (p-value): % of students who answered correctly
  - Discrimination Index: point-biserial correlation between item score and total score
  - Distractor Efficiency: how well each wrong answer attracts low-ability students
  - Score Distribution: mapped to real AP score bands (5/4/3/2/1)
  - Reliability: KR-20 (Kuder-Richardson) for the full test

Student profiles are calibrated to produce score distributions matching
real College Board AP US Government data (2024-2025):
  5: ~24%  |  4: ~25%  |  3: ~23%  |  2: ~18%  |  1: ~10%

Usage:
    python ap_item_validator.py --questions questions.json --output report.json
    python ap_item_validator.py --questions questions.json --students 200 --runs 5
"""

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

# ── path + env setup ─────────────────────────────────────────────────────────
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
_backend_dir = os.path.abspath(os.path.join(_scripts_dir, ".."))
_project_root = os.path.abspath(os.path.join(_backend_dir, ".."))

from dotenv import load_dotenv
_env_file = os.path.join(_project_root, ".env")
if os.path.exists(_env_file):
    load_dotenv(_env_file)

from openai import OpenAI


# ══════════════════════════════════════════════════════════════════════════════
# REAL AP US GOVERNMENT SCORE DISTRIBUTION (2024-2025 average)
# Source: College Board official score distributions
# ══════════════════════════════════════════════════════════════════════════════
AP_GOV_SCORE_DISTRIBUTION = {
    5: 0.240,  # 24.0%
    4: 0.249,  # 24.9%
    3: 0.235,  # 23.5%
    2: 0.183,  # 18.3%
    1: 0.094,  # 9.4%
}
AP_GOV_MEAN_SCORE = 3.36


# ══════════════════════════════════════════════════════════════════════════════
# STUDENT ABILITY MODEL
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SimulatedStudent:
    """
    A calibrated student with a latent ability parameter (theta).

    theta follows a distribution that reproduces the real AP score distribution
    when mapped through a standard IRT model.

    theta ~  N(0, 1) but weighted to match AP Gov score bands:
      theta > 1.2   → AP 5  (24%)
      0.3 < theta    → AP 4  (25%)
      -0.4 < theta   → AP 3  (23%)
      -1.1 < theta   → AP 2  (18%)
      theta < -1.1   → AP 1  (10%)
    """
    student_id: int
    theta: float  # latent ability (-3 to +3 scale)
    ap_band: int  # expected AP score (1-5)

    # Behavioral modifiers
    guessing_tendency: float = 0.0     # 0-1, how likely to guess randomly
    carelessness: float = 0.0          # 0-1, chance of getting easy items wrong
    test_anxiety: float = 0.0          # 0-1, penalty on borderline items
    reading_speed: float = 1.0         # multiplier for complex question penalty

    # Archetype label (for reporting)
    archetype: str = "average_student"

    def probability_correct(self, difficulty: float, discrimination: float = 1.0) -> float:
        """
        2PL IRT model: probability of correct answer.

        P(correct) = guessing + (1 - guessing) * logistic(a * (theta - b))

        Where:
          a = discrimination parameter
          b = difficulty parameter
          c = guessing parameter (1/num_options for MCQ)
        """
        c = 0.25  # 4-option MCQ baseline guessing
        c_effective = c + self.guessing_tendency * 0.10  # extra guessing for some students

        # Apply test anxiety as a theta penalty on borderline items
        effective_theta = self.theta
        if self.test_anxiety > 0:
            # Anxiety hurts most on items near ability level
            proximity = abs(self.theta - difficulty)
            if proximity < 0.5:
                effective_theta -= self.test_anxiety * 0.3

        z = discrimination * (effective_theta - difficulty)
        p_irt = 1.0 / (1.0 + math.exp(-z))

        # 3PL: account for guessing
        p = c_effective + (1.0 - c_effective) * p_irt

        # Carelessness: even high-ability students sometimes miss easy items
        if self.carelessness > 0 and difficulty < self.theta - 1.0:
            p *= (1.0 - self.carelessness * 0.15)

        return max(0.0, min(1.0, p))

    def select_answer(
        self,
        correct_idx: int,
        num_options: int,
        difficulty: float,
        discrimination: float = 1.0,
        distractor_attractiveness: Optional[List[float]] = None,
    ) -> int:
        """
        Simulate selecting an answer. Returns the chosen option index (0-based).

        If the student gets it wrong, they don't pick randomly — they're
        attracted to plausible distractors based on distractor_attractiveness weights.
        """
        p_correct = self.probability_correct(difficulty, discrimination)

        if random.random() < p_correct:
            return correct_idx

        # Wrong answer — pick a distractor
        if distractor_attractiveness is not None:
            # Remove correct answer from weights
            weights = list(distractor_attractiveness)
            weights[correct_idx] = 0.0
            total = sum(weights)
            if total > 0:
                weights = [w / total for w in weights]
                return random.choices(range(num_options), weights=weights)[0]

        # Uniform random among wrong answers
        wrong_options = [i for i in range(num_options) if i != correct_idx]
        return random.choice(wrong_options)


def generate_calibrated_population(
    n_students: int = 200,
    seed: Optional[int] = None,
) -> List[SimulatedStudent]:
    """
    Generate a student population calibrated to real AP Gov score distributions.

    Uses stratified sampling to ensure the ability distribution reproduces
    the official College Board score bands.
    """
    if seed is not None:
        random.seed(seed)

    students = []

    # Archetype definitions with behavioral modifiers
    archetypes = {
        "high_achiever": {
            "theta_range": (1.2, 2.5),
            "guessing": (0.0, 0.05),
            "carelessness": (0.0, 0.05),
            "anxiety": (0.05, 0.15),
            "reading_speed": (1.1, 1.3),
            "weight": AP_GOV_SCORE_DISTRIBUTION[5],  # 24%
            "ap_band": 5,
        },
        "strong_student": {
            "theta_range": (0.3, 1.2),
            "guessing": (0.02, 0.10),
            "carelessness": (0.02, 0.10),
            "anxiety": (0.05, 0.20),
            "reading_speed": (1.0, 1.2),
            "weight": AP_GOV_SCORE_DISTRIBUTION[4],  # 25%
            "ap_band": 4,
        },
        "average_student": {
            "theta_range": (-0.4, 0.3),
            "guessing": (0.05, 0.20),
            "carelessness": (0.05, 0.15),
            "anxiety": (0.10, 0.30),
            "reading_speed": (0.8, 1.1),
            "weight": AP_GOV_SCORE_DISTRIBUTION[3],  # 23%
            "ap_band": 3,
        },
        "struggling_student": {
            "theta_range": (-1.1, -0.4),
            "guessing": (0.10, 0.30),
            "carelessness": (0.10, 0.25),
            "anxiety": (0.15, 0.40),
            "reading_speed": (0.7, 0.9),
            "weight": AP_GOV_SCORE_DISTRIBUTION[2],  # 18%
            "ap_band": 2,
        },
        "at_risk_student": {
            "theta_range": (-2.5, -1.1),
            "guessing": (0.20, 0.50),
            "carelessness": (0.15, 0.35),
            "anxiety": (0.20, 0.50),
            "reading_speed": (0.5, 0.8),
            "weight": AP_GOV_SCORE_DISTRIBUTION[1],  # 10%
            "ap_band": 1,
        },
    }

    sid = 0
    for arch_name, arch in archetypes.items():
        n_in_band = round(n_students * arch["weight"])
        for _ in range(n_in_band):
            theta = random.uniform(*arch["theta_range"])
            students.append(SimulatedStudent(
                student_id=sid,
                theta=theta,
                ap_band=arch["ap_band"],
                guessing_tendency=random.uniform(*arch["guessing"]),
                carelessness=random.uniform(*arch["carelessness"]),
                test_anxiety=random.uniform(*arch["anxiety"]),
                reading_speed=random.uniform(*arch["reading_speed"]),
                archetype=arch_name,
            ))
            sid += 1

    # Fill remainder if rounding left us short
    while len(students) < n_students:
        theta = random.gauss(0, 1)
        band = theta_to_ap_band(theta)
        students.append(SimulatedStudent(
            student_id=sid, theta=theta, ap_band=band,
            archetype="average_student",
        ))
        sid += 1

    random.shuffle(students)
    return students[:n_students]


def theta_to_ap_band(theta: float) -> int:
    if theta > 1.2:
        return 5
    elif theta > 0.3:
        return 4
    elif theta > -0.4:
        return 3
    elif theta > -1.1:
        return 2
    else:
        return 1


# ══════════════════════════════════════════════════════════════════════════════
# QUESTION MODEL
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class APQuestion:
    """An AP-style MCQ question with metadata."""
    question_id: str
    stem: str                         # The question text
    options: List[str]                # Answer options (A, B, C, D)
    correct_index: int                # 0-based index of correct answer
    difficulty_estimate: float        # Expected difficulty (-3 to +3, higher = harder)
    discrimination_estimate: float    # Expected discrimination (0.5 to 2.0)
    topic: str = ""                   # e.g., "14th Amendment", "Federalism"
    skill: str = ""                   # e.g., "analysis", "recall", "application"
    distractor_attractiveness: Optional[List[float]] = None  # per-option weights

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "APQuestion":
        # Map difficulty labels to IRT parameters
        diff_map = {
            "easy": -1.0, "medium_easy": -0.5, "medium": 0.0,
            "medium_hard": 0.5, "hard": 1.0, "very_hard": 1.5,
        }
        diff = d.get("difficulty_estimate", 0.0)
        if isinstance(diff, str):
            diff = diff_map.get(diff.lower(), 0.0)

        disc = d.get("discrimination_estimate", 1.0)

        # Distractor attractiveness
        da = d.get("distractor_attractiveness", None)
        if da is None:
            # Default: equal attractiveness for all options
            n = len(d.get("options", ["A", "B", "C", "D"]))
            da = [1.0] * n

        return cls(
            question_id=d.get("question_id", d.get("id", "Q0")),
            stem=d.get("stem", d.get("question", "")),
            options=d.get("options", []),
            correct_index=d.get("correct_index", d.get("answer_index", 0)),
            difficulty_estimate=diff,
            discrimination_estimate=disc,
            topic=d.get("topic", ""),
            skill=d.get("skill", ""),
            distractor_attractiveness=da,
        )


# ══════════════════════════════════════════════════════════════════════════════
# LLM QUESTION DIFFICULTY ESTIMATOR
# ══════════════════════════════════════════════════════════════════════════════

class DifficultyEstimator:
    """Use LLM to estimate IRT parameters for questions that don't have them."""

    def __init__(self):
        self.api_key = os.environ.get("LLM_API_KEY", "")
        self.base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
        self.model_name = os.environ.get("LLM_MODEL_NAME", "gpt-5-mini")
        self.client = None
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def estimate(self, question: APQuestion) -> Tuple[float, float, List[float]]:
        """
        Returns (difficulty, discrimination, distractor_attractiveness).
        Uses LLM if available, otherwise uses heuristics.
        """
        if not self.client:
            return self._heuristic_estimate(question)

        prompt = f"""Analyze this AP US Government multiple choice question and estimate its psychometric properties.

Question: {question.stem}
Options:
{chr(10).join(f'  {chr(65+i)}. {opt}' for i, opt in enumerate(question.options))}
Correct Answer: {chr(65 + question.correct_index)}

Return a JSON object with:
1. "difficulty": float from -2.0 (very easy) to 2.0 (very hard). 0.0 = average AP student gets it right 50% of the time.
2. "discrimination": float from 0.3 to 2.0. How well does this item separate high-ability from low-ability students? 1.0 = average, >1.5 = excellent.
3. "distractor_attractiveness": array of 4 floats (one per option). How attractive is each option to a student who doesn't know the answer? Correct answer should be 0.0. Higher = more attractive distractor.
4. "reasoning": brief explanation of your estimates.

Consider:
- Bloom's taxonomy level (recall vs analysis vs evaluation)
- Common student misconceptions that might make distractors attractive
- Vocabulary complexity and reading demand
- Whether the question requires application of concepts or just memorization

Return ONLY valid JSON."""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an AP exam psychometrician. Estimate IRT parameters for test items. Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            content = response.choices[0].message.content.strip()
            # Strip markdown code fences if present
            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            result = json.loads(content)
            diff = float(result.get("difficulty", 0.0))
            disc = float(result.get("discrimination", 1.0))
            da = result.get("distractor_attractiveness", [1.0] * len(question.options))
            return (
                max(-3.0, min(3.0, diff)),
                max(0.3, min(2.5, disc)),
                [float(x) for x in da],
            )
        except Exception as e:
            print(f"    ⚠ LLM estimation failed for {question.question_id}: {e}")
            return self._heuristic_estimate(question)

    def _heuristic_estimate(self, question: APQuestion) -> Tuple[float, float, List[float]]:
        """Fallback: use question length and vocabulary as rough heuristics."""
        stem_len = len(question.stem.split())
        if stem_len > 60:
            diff = 0.5  # longer stems tend to be harder
        elif stem_len > 30:
            diff = 0.0
        else:
            diff = -0.3

        disc = 1.0  # assume average discrimination
        da = [1.0] * len(question.options)
        da[question.correct_index] = 0.0
        return diff, disc, da


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ItemResult:
    """Results for a single question across all students."""
    question_id: str
    topic: str
    skill: str

    # Core psychometric indices
    difficulty_index: float = 0.0         # p-value (% correct)
    discrimination_index: float = 0.0      # point-biserial correlation
    difficulty_irt: float = 0.0            # IRT difficulty parameter used

    # Distractor analysis
    option_frequencies: List[float] = field(default_factory=list)   # % choosing each option
    option_by_band: Dict[int, List[float]] = field(default_factory=dict)  # band → [% per option]

    # Quality classification
    quality: str = "acceptable"  # excellent, good, acceptable, poor, discard

    # Raw data
    n_students: int = 0
    responses: List[int] = field(default_factory=list)  # chosen option per student

    def classify_quality(self):
        """Classify item quality based on standard psychometric criteria."""
        p = self.difficulty_index
        d = self.discrimination_index

        if 0.30 <= p <= 0.70 and d >= 0.30:
            self.quality = "excellent"
        elif 0.20 <= p <= 0.80 and d >= 0.20:
            self.quality = "good"
        elif 0.15 <= p <= 0.85 and d >= 0.10:
            self.quality = "acceptable"
        elif d < 0.10 or d < 0:
            self.quality = "discard"
        else:
            self.quality = "poor"


@dataclass
class TestResult:
    """Aggregate results for a full test."""
    n_questions: int = 0
    n_students: int = 0

    # Score distribution mapped to AP bands
    score_distribution: Dict[int, float] = field(default_factory=dict)  # band → %
    mean_score: float = 0.0
    std_dev: float = 0.0

    # Reliability
    kr20: float = 0.0  # Kuder-Richardson 20

    # Item summary
    items: List[ItemResult] = field(default_factory=list)
    quality_distribution: Dict[str, int] = field(default_factory=dict)

    # Per-student scores (for computing correlations)
    student_total_scores: List[float] = field(default_factory=list)


def run_simulation(
    questions: List[APQuestion],
    n_students: int = 200,
    n_runs: int = 3,
    seed: Optional[int] = 42,
    progress_callback: Optional[Callable] = None,
) -> TestResult:
    """
    Run the psychometric simulation.

    Args:
        questions: list of AP questions to validate
        n_students: number of simulated students per run
        n_runs: number of independent runs to average (reduces noise)
        seed: random seed for reproducibility
        progress_callback: (step, total, message)

    Returns:
        TestResult with full item analysis
    """
    estimator = DifficultyEstimator()
    n_questions = len(questions)
    total_steps = n_runs * n_questions + n_questions  # estimation + runs
    step = 0

    # Step 1: Estimate IRT parameters for each question
    for q in questions:
        step += 1
        if progress_callback:
            progress_callback(step, total_steps, f"Estimating parameters: {q.question_id}")

        if q.difficulty_estimate == 0.0 and q.discrimination_estimate == 1.0:
            diff, disc, da = estimator.estimate(q)
            q.difficulty_estimate = diff
            q.discrimination_estimate = disc
            q.distractor_attractiveness = da

    # Step 2: Run simulations
    # Accumulate responses across runs
    all_responses = {q.question_id: [] for q in questions}  # qid → [run1_responses, run2_responses, ...]
    all_total_scores = []

    for run_idx in range(n_runs):
        run_seed = (seed + run_idx * 1000) if seed is not None else None
        students = generate_calibrated_population(n_students, seed=run_seed)

        # Each student answers each question
        student_scores = []
        for student in students:
            score = 0
            for q in questions:
                step += 1
                chosen = student.select_answer(
                    correct_idx=q.correct_index,
                    num_options=len(q.options),
                    difficulty=q.difficulty_estimate,
                    discrimination=q.discrimination_estimate,
                    distractor_attractiveness=q.distractor_attractiveness,
                )
                all_responses[q.question_id].append((student, chosen))
                if chosen == q.correct_index:
                    score += 1

            student_scores.append((student, score))

        all_total_scores.extend(student_scores)

        if progress_callback:
            progress_callback(
                n_questions + (run_idx + 1) * n_questions,
                total_steps,
                f"Run {run_idx + 1}/{n_runs} complete",
            )

    # Step 3: Compute item statistics
    total_students = n_students * n_runs
    total_scores_array = [s for _, s in all_total_scores]
    mean_total = sum(total_scores_array) / len(total_scores_array)
    var_total = sum((s - mean_total) ** 2 for s in total_scores_array) / len(total_scores_array)
    std_total = math.sqrt(var_total) if var_total > 0 else 1.0

    item_results = []
    item_variances = []

    for q in questions:
        responses = all_responses[q.question_id]  # [(student, chosen), ...]

        # Difficulty index (p-value)
        n_correct = sum(1 for _, c in responses if c == q.correct_index)
        p_value = n_correct / len(responses) if responses else 0.0

        # Option frequencies
        n_options = len(q.options)
        option_counts = [0] * n_options
        for _, chosen in responses:
            if 0 <= chosen < n_options:
                option_counts[chosen] += 1
        option_freq = [c / len(responses) for c in option_counts] if responses else [0] * n_options

        # Option frequencies by AP band
        band_option_counts = {b: [0] * n_options for b in range(1, 6)}
        band_totals = {b: 0 for b in range(1, 6)}
        for student, chosen in responses:
            b = student.ap_band
            if 0 <= chosen < n_options:
                band_option_counts[b][chosen] += 1
            band_totals[b] += 1
        option_by_band = {}
        for b in range(1, 6):
            if band_totals[b] > 0:
                option_by_band[b] = [c / band_totals[b] for c in band_option_counts[b]]
            else:
                option_by_band[b] = [0.0] * n_options

        # Discrimination: point-biserial correlation
        item_scores = [1 if c == q.correct_index else 0 for _, c in responses]
        student_totals = [None] * len(responses)

        # Build a lookup of student total scores
        student_score_map = {}
        for student, total_score in all_total_scores:
            student_score_map[student.student_id] = total_score

        corr_scores = []
        for (student, chosen), item_score in zip(responses, item_scores):
            ts = student_score_map.get(student.student_id, mean_total)
            corr_scores.append((item_score, ts))

        # Point-biserial correlation
        rpb = compute_point_biserial(corr_scores)

        # Item variance for KR-20
        item_var = p_value * (1 - p_value)
        item_variances.append(item_var)

        ir = ItemResult(
            question_id=q.question_id,
            topic=q.topic,
            skill=q.skill,
            difficulty_index=round(p_value, 3),
            discrimination_index=round(rpb, 3),
            difficulty_irt=round(q.difficulty_estimate, 2),
            option_frequencies=[round(f, 3) for f in option_freq],
            option_by_band={b: [round(f, 3) for f in freqs] for b, freqs in option_by_band.items()},
            n_students=len(responses),
        )
        ir.classify_quality()
        item_results.append(ir)

    # Step 4: Test-level statistics
    # KR-20 reliability
    sum_item_var = sum(item_variances)
    kr20 = (n_questions / (n_questions - 1)) * (1 - sum_item_var / var_total) if var_total > 0 and n_questions > 1 else 0.0

    # Score distribution by AP band
    band_counts = {b: 0 for b in range(1, 6)}
    for student, _ in all_total_scores:
        band_counts[student.ap_band] += 1
    score_dist = {b: round(c / len(all_total_scores), 3) for b, c in band_counts.items()}

    # Quality distribution
    quality_dist = {}
    for ir in item_results:
        quality_dist[ir.quality] = quality_dist.get(ir.quality, 0) + 1

    result = TestResult(
        n_questions=n_questions,
        n_students=total_students,
        score_distribution=score_dist,
        mean_score=round(mean_total, 2),
        std_dev=round(std_total, 2),
        kr20=round(kr20, 3),
        items=item_results,
        quality_distribution=quality_dist,
        student_total_scores=total_scores_array,
    )

    return result


def compute_point_biserial(pairs: List[Tuple[int, float]]) -> float:
    """Compute point-biserial correlation between binary item scores and continuous total scores."""
    if not pairs:
        return 0.0

    ones = [total for item, total in pairs if item == 1]
    zeros = [total for item, total in pairs if item == 0]

    if not ones or not zeros:
        return 0.0

    n = len(pairs)
    n1 = len(ones)
    n0 = len(zeros)
    mean1 = sum(ones) / n1
    mean0 = sum(zeros) / n0

    all_scores = [total for _, total in pairs]
    mean_all = sum(all_scores) / n
    var_all = sum((s - mean_all) ** 2 for s in all_scores) / n
    sd_all = math.sqrt(var_all) if var_all > 0 else 1.0

    p = n1 / n
    q = n0 / n

    rpb = ((mean1 - mean0) / sd_all) * math.sqrt(p * q)
    return rpb


# ══════════════════════════════════════════════════════════════════════════════
# REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def format_report(result: TestResult, questions: List[APQuestion]) -> Dict[str, Any]:
    """Format results into a comprehensive JSON report."""
    q_map = {q.question_id: q for q in questions}

    items_report = []
    for ir in result.items:
        q = q_map.get(ir.question_id)
        option_labels = [chr(65 + i) for i in range(len(ir.option_frequencies))]

        # Distractor analysis
        distractor_analysis = []
        for i, (label, freq) in enumerate(zip(option_labels, ir.option_frequencies)):
            is_correct = (i == q.correct_index) if q else False
            # Check if distractor attracts low-ability more than high-ability
            low_freq = ir.option_by_band.get(1, [0] * len(option_labels))[i] if ir.option_by_band else 0
            high_freq = ir.option_by_band.get(5, [0] * len(option_labels))[i] if ir.option_by_band else 0

            status = "correct_answer" if is_correct else (
                "effective" if low_freq > high_freq and freq >= 0.05 else
                "weak" if freq < 0.05 else
                "problematic"
            )
            distractor_analysis.append({
                "option": label,
                "frequency": freq,
                "is_correct": is_correct,
                "low_ability_freq": round(low_freq, 3),
                "high_ability_freq": round(high_freq, 3),
                "status": status,
            })

        items_report.append({
            "question_id": ir.question_id,
            "topic": ir.topic,
            "skill": ir.skill,
            "stem_preview": q.stem[:100] + "..." if q and len(q.stem) > 100 else (q.stem if q else ""),
            "difficulty_index": ir.difficulty_index,
            "discrimination_index": ir.discrimination_index,
            "quality": ir.quality,
            "option_analysis": distractor_analysis,
            "recommendation": get_recommendation(ir),
        })

    return {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "n_students": result.n_students,
            "n_questions": result.n_questions,
            "engine": "AP Item Validator v1.0 (IRT 2PL/3PL)",
        },
        "test_level": {
            "kr20_reliability": result.kr20,
            "mean_raw_score": result.mean_score,
            "std_dev": result.std_dev,
            "mean_percent_correct": round(result.mean_score / result.n_questions * 100, 1) if result.n_questions > 0 else 0,
            "score_distribution": {
                f"AP_{b}": f"{pct*100:.1f}%" for b, pct in sorted(result.score_distribution.items(), reverse=True)
            },
            "target_distribution": {
                f"AP_{b}": f"{pct*100:.1f}%" for b, pct in sorted(AP_GOV_SCORE_DISTRIBUTION.items(), reverse=True)
            },
            "quality_summary": result.quality_distribution,
        },
        "items": items_report,
    }


def get_recommendation(ir: ItemResult) -> str:
    """Generate actionable recommendation for each item."""
    if ir.quality == "excellent":
        return "✅ Keep as-is. Strong item with good difficulty and discrimination."
    elif ir.quality == "good":
        return "✅ Keep. Minor improvements possible to distractors."
    elif ir.quality == "acceptable":
        if ir.difficulty_index < 0.20:
            return "⚠️ Too hard. Consider simplifying the stem or making distractors more distinct."
        elif ir.difficulty_index > 0.80:
            return "⚠️ Too easy. Increase cognitive demand or improve distractor plausibility."
        elif ir.discrimination_index < 0.20:
            return "⚠️ Low discrimination. Review distractors — some may be confusing to high-ability students."
        return "⚠️ Acceptable but review distractors for effectiveness."
    elif ir.quality == "poor":
        if ir.discrimination_index < 0:
            return "❌ Negative discrimination — high-ability students get this WRONG more often. Check for ambiguity in the stem or correct answer."
        return "❌ Poor item. Needs significant revision to stem and/or distractors."
    else:
        return "❌ Discard. This item does not differentiate between ability levels."


def print_report(report: Dict[str, Any]):
    """Pretty-print the report to console."""
    print("\n" + "=" * 70)
    print("  📊  AP ITEM VALIDATION REPORT")
    print("=" * 70)

    tl = report["test_level"]
    print(f"\n  Students simulated:  {report['meta']['n_students']}")
    print(f"  Questions analyzed:  {report['meta']['n_questions']}")
    print(f"  KR-20 Reliability:   {tl['kr20_reliability']}")
    print(f"  Mean % Correct:      {tl['mean_percent_correct']}%")
    print(f"  Std Dev (raw):       {tl['std_dev']}")

    print(f"\n  Score Distribution (simulated vs. target):")
    for band in [5, 4, 3, 2, 1]:
        sim = tl["score_distribution"].get(f"AP_{band}", "0.0%")
        target = tl["target_distribution"].get(f"AP_{band}", "0.0%")
        print(f"    AP {band}:  {sim:>6s}  (target: {target})")

    print(f"\n  Quality Summary:")
    for q, count in sorted(tl["quality_summary"].items()):
        emoji = {"excellent": "🟢", "good": "🟡", "acceptable": "🟠", "poor": "🔴", "discard": "⛔"}.get(q, "❓")
        print(f"    {emoji} {q:12s}: {count} items")

    print(f"\n{'─' * 70}")
    print(f"  ITEM DETAILS")
    print(f"{'─' * 70}")

    for item in report["items"]:
        q_emoji = {"excellent": "🟢", "good": "🟡", "acceptable": "🟠", "poor": "🔴", "discard": "⛔"}.get(item["quality"], "❓")
        print(f"\n  {q_emoji} {item['question_id']} [{item['topic']}]")
        print(f"     {item['stem_preview']}")
        print(f"     Difficulty: {item['difficulty_index']:.3f}  |  Discrimination: {item['discrimination_index']:.3f}  |  Quality: {item['quality']}")

        for opt in item["option_analysis"]:
            marker = "✓" if opt["is_correct"] else " "
            status_icon = {"effective": "✅", "weak": "⚠️", "problematic": "❌", "correct_answer": "🎯"}.get(opt["status"], "")
            print(f"       {marker} {opt['option']}: {opt['frequency']*100:5.1f}%  (low:{opt['low_ability_freq']*100:4.1f}% / high:{opt['high_ability_freq']*100:4.1f}%)  {status_icon} {opt['status']}")

        print(f"     → {item['recommendation']}")

    print(f"\n{'=' * 70}\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="AP Item Validator — Psychometric Simulation")
    parser.add_argument("--questions", type=str, required=True, help="Questions JSON file")
    parser.add_argument("--output", type=str, default=None, help="Output report JSON path")
    parser.add_argument("--students", type=int, default=200, help="Students per simulation run")
    parser.add_argument("--runs", type=int, default=3, help="Number of simulation runs to average")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quiet", action="store_true", help="Suppress console output")
    args = parser.parse_args()

    # Load questions
    with open(args.questions, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Support both {"questions": [...]} and bare [...]
    if isinstance(raw, list):
        questions_data = raw
    else:
        questions_data = raw.get("questions", raw.get("items", []))

    questions = [APQuestion.from_dict(q) for q in questions_data]
    print(f"\n  📝 Loaded {len(questions)} questions")

    def progress(step, total, msg):
        if not args.quiet:
            print(f"  [{step}/{total}] {msg}")

    # Run simulation
    result = run_simulation(
        questions=questions,
        n_students=args.students,
        n_runs=args.runs,
        seed=args.seed,
        progress_callback=progress,
    )

    # Generate report
    report = format_report(result, questions)

    if not args.quiet:
        print_report(report)

    # Save report
    output_path = args.output or "item_validation_report.json"
    # Remove non-serializable data
    report_clean = json.loads(json.dumps(report, default=str))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_clean, f, ensure_ascii=False, indent=2)
    print(f"  💾 Report saved to {output_path}")


if __name__ == "__main__":
    main()
