"""
Psychometric Simulation Module — IRT-based Item Analysis
IRT-based psychometric simulation for item analysis.

Runs calibrated student populations through questions using Item Response Theory
(2PL/3PL model) and produces standardized psychometric metrics.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════════════════
# AP SCORE CALIBRATION DATA
# ═══════════════════════════════════════════════════════════════════════════════

AP_SCORE_DISTRIBUTIONS = {
    "ap_us_government": {
        5: 0.240, 4: 0.249, 3: 0.235, 2: 0.183, 1: 0.094,
        "mean": 3.36, "label": "AP US Government and Politics",
    },
    "ap_us_history": {
        5: 0.122, 4: 0.177, 3: 0.236, 2: 0.221, 1: 0.244,
        "mean": 2.71, "label": "AP United States History",
    },
    "ap_world_history": {
        5: 0.112, 4: 0.189, 3: 0.266, 2: 0.221, 1: 0.212,
        "mean": 2.77, "label": "AP World History: Modern",
    },
    "ap_human_geography": {
        5: 0.158, 4: 0.192, 3: 0.213, 2: 0.181, 1: 0.256,
        "mean": 2.82, "label": "AP Human Geography",
    },
}

# Theta cutoffs for each AP band
THETA_CUTOFFS = {
    5: 1.2,    # theta > 1.2
    4: 0.3,    # 0.3 < theta <= 1.2
    3: -0.4,   # -0.4 < theta <= 0.3
    2: -1.1,   # -1.1 < theta <= -0.4
    1: -3.0,   # theta <= -1.1
}


def theta_to_band(theta: float) -> int:
    if theta > 1.2: return 5
    if theta > 0.3: return 4
    if theta > -0.4: return 3
    if theta > -1.1: return 2
    return 1


# ═══════════════════════════════════════════════════════════════════════════════
# STUDENT MODEL
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SimStudent:
    student_id: int
    theta: float
    ap_band: int
    guessing: float = 0.0
    carelessness: float = 0.0
    anxiety: float = 0.0

    def p_correct(self, difficulty: float, discrimination: float = 1.0) -> float:
        """3PL IRT probability of correct response."""
        c = 0.25 + self.guessing * 0.10
        eff_theta = self.theta
        if self.anxiety > 0 and abs(self.theta - difficulty) < 0.5:
            eff_theta -= self.anxiety * 0.3
        z = discrimination * (eff_theta - difficulty)
        p_irt = 1.0 / (1.0 + math.exp(-z))
        p = c + (1.0 - c) * p_irt
        if self.carelessness > 0 and difficulty < self.theta - 1.0:
            p *= (1.0 - self.carelessness * 0.15)
        return max(0.0, min(1.0, p))

    def answer(
        self, correct_idx: int, n_options: int,
        difficulty: float, discrimination: float = 1.0,
        distractor_weights: Optional[List[float]] = None,
    ) -> int:
        if random.random() < self.p_correct(difficulty, discrimination):
            return correct_idx
        if distractor_weights:
            w = list(distractor_weights)
            w[correct_idx] = 0.0
            total = sum(w)
            if total > 0:
                return random.choices(range(n_options), weights=[x / total for x in w])[0]
        wrong = [i for i in range(n_options) if i != correct_idx]
        return random.choice(wrong)


def generate_population(
    n: int, subject: str = "ap_us_government", seed: Optional[int] = None
) -> List[SimStudent]:
    """Generate population calibrated to real AP score distributions."""
    if seed is not None:
        random.seed(seed)

    dist = AP_SCORE_DISTRIBUTIONS.get(subject, AP_SCORE_DISTRIBUTIONS["ap_us_government"])

    band_configs = {
        5: {"theta": (1.2, 2.5), "guess": (0, 0.05), "care": (0, 0.05), "anx": (0.05, 0.15)},
        4: {"theta": (0.3, 1.2), "guess": (0.02, 0.10), "care": (0.02, 0.10), "anx": (0.05, 0.20)},
        3: {"theta": (-0.4, 0.3), "guess": (0.05, 0.20), "care": (0.05, 0.15), "anx": (0.10, 0.30)},
        2: {"theta": (-1.1, -0.4), "guess": (0.10, 0.30), "care": (0.10, 0.25), "anx": (0.15, 0.40)},
        1: {"theta": (-2.5, -1.1), "guess": (0.20, 0.50), "care": (0.15, 0.35), "anx": (0.20, 0.50)},
    }

    students = []
    sid = 0
    for band in [5, 4, 3, 2, 1]:
        count = round(n * dist.get(band, 0.2))
        cfg = band_configs[band]
        for _ in range(count):
            students.append(SimStudent(
                student_id=sid,
                theta=random.uniform(*cfg["theta"]),
                ap_band=band,
                guessing=random.uniform(*cfg["guess"]),
                carelessness=random.uniform(*cfg["care"]),
                anxiety=random.uniform(*cfg["anx"]),
            ))
            sid += 1

    while len(students) < n:
        theta = random.gauss(0, 1)
        students.append(SimStudent(sid, theta, theta_to_band(theta)))
        sid += 1

    random.shuffle(students)
    return students[:n]


# ═══════════════════════════════════════════════════════════════════════════════
# ITEM ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ItemStats:
    question_id: str
    difficulty_index: float = 0.0        # p-value
    discrimination_index: float = 0.0     # point-biserial r
    option_frequencies: List[float] = field(default_factory=list)
    option_by_band: Dict[int, List[float]] = field(default_factory=dict)
    quality: str = "acceptable"
    n_responses: int = 0

    def classify(self):
        p, d = self.difficulty_index, self.discrimination_index
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
class TestStats:
    n_questions: int = 0
    n_students: int = 0
    kr20: float = 0.0
    mean_score: float = 0.0
    std_dev: float = 0.0
    score_distribution: Dict[int, float] = field(default_factory=dict)
    items: List[ItemStats] = field(default_factory=list)
    quality_summary: Dict[str, int] = field(default_factory=dict)


def point_biserial(pairs: List[Tuple[int, float]]) -> float:
    """Point-biserial r between binary item scores and continuous total scores."""
    if not pairs:
        return 0.0
    ones = [t for s, t in pairs if s == 1]
    zeros = [t for s, t in pairs if s == 0]
    if not ones or not zeros:
        return 0.0
    n = len(pairs)
    m1, m0 = sum(ones) / len(ones), sum(zeros) / len(zeros)
    all_t = [t for _, t in pairs]
    mu = sum(all_t) / n
    var = sum((t - mu) ** 2 for t in all_t) / n
    sd = math.sqrt(var) if var > 0 else 1.0
    p = len(ones) / n
    return ((m1 - m0) / sd) * math.sqrt(p * (1 - p))


def run_psychometric_simulation(
    questions: List[Dict[str, Any]],
    n_students: int = 200,
    n_runs: int = 3,
    subject: str = "ap_us_government",
    seed: int = 42,
    progress_callback: Optional[Callable] = None,
) -> TestStats:
    """
    Main simulation entry point.

    Args:
        questions: list of dicts with keys:
            question_id, correct_index, n_options, difficulty, discrimination,
            distractor_weights (optional)
        n_students: per run
        n_runs: independent runs to average
        subject: AP subject for calibration
        seed: reproducibility
        progress_callback: (step, total, msg)

    Returns:
        TestStats with full item analysis
    """
    nq = len(questions)
    all_responses = {q["question_id"]: [] for q in questions}
    all_totals = []

    for run in range(n_runs):
        pop = generate_population(n_students, subject, seed=seed + run * 1000)
        for student in pop:
            score = 0
            for q in questions:
                chosen = student.answer(
                    correct_idx=q.get("correct_index", 0),
                    n_options=q.get("n_options", 4),
                    difficulty=q.get("difficulty", 0.0),
                    discrimination=q.get("discrimination", 1.0),
                    distractor_weights=q.get("distractor_weights"),
                )
                all_responses[q["question_id"]].append((student, chosen))
                if chosen == q.get("correct_index", 0):
                    score += 1
            all_totals.append((student, score))

        if progress_callback:
            progress_callback(run + 1, n_runs, f"Run {run + 1}/{n_runs}")

    total_n = len(all_totals)
    scores = [s for _, s in all_totals]
    mean = sum(scores) / total_n
    var = sum((s - mean) ** 2 for s in scores) / total_n
    std = math.sqrt(var) if var > 0 else 1.0

    score_map = {}
    for st, _ in all_totals:
        score_map[st.student_id] = score_map.get(st.student_id, [])
    # Rebuild per-student score map
    stu_scores = {}
    for st, sc in all_totals:
        stu_scores[st.student_id] = sc

    items = []
    item_vars = []

    for q in questions:
        qid = q["question_id"]
        responses = all_responses[qid]
        ci = q.get("correct_index", 0)
        no = q.get("n_options", 4)

        n_correct = sum(1 for _, c in responses if c == ci)
        p = n_correct / len(responses) if responses else 0

        # Option frequencies
        counts = [0] * no
        for _, c in responses:
            if 0 <= c < no:
                counts[c] += 1
        freq = [c / len(responses) for c in counts] if responses else [0] * no

        # By band
        band_counts = {b: [0] * no for b in range(1, 6)}
        band_totals = {b: 0 for b in range(1, 6)}
        for st, c in responses:
            band_counts[st.ap_band][c] += 1
            band_totals[st.ap_band] += 1
        by_band = {}
        for b in range(1, 6):
            if band_totals[b] > 0:
                by_band[b] = [round(c / band_totals[b], 3) for c in band_counts[b]]

        # Point-biserial
        pairs = [(1 if c == ci else 0, stu_scores.get(st.student_id, mean)) for st, c in responses]
        rpb = point_biserial(pairs)

        item_vars.append(p * (1 - p))

        ist = ItemStats(
            question_id=qid,
            difficulty_index=round(p, 3),
            discrimination_index=round(rpb, 3),
            option_frequencies=[round(f, 3) for f in freq],
            option_by_band=by_band,
            n_responses=len(responses),
        )
        ist.classify()
        items.append(ist)

    # KR-20
    kr20 = (nq / (nq - 1)) * (1 - sum(item_vars) / var) if var > 0 and nq > 1 else 0

    # Score distribution
    band_counts = {b: 0 for b in range(1, 6)}
    for st, _ in all_totals:
        band_counts[st.ap_band] += 1
    score_dist = {b: round(c / total_n, 3) for b, c in band_counts.items()}

    quality_summary = {}
    for it in items:
        quality_summary[it.quality] = quality_summary.get(it.quality, 0) + 1

    return TestStats(
        n_questions=nq,
        n_students=total_n,
        kr20=round(kr20, 3),
        mean_score=round(mean, 2),
        std_dev=round(std, 2),
        score_distribution=score_dist,
        items=items,
        quality_summary=quality_summary,
    )
