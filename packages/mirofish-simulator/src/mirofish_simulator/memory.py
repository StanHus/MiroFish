"""
Agent Memory System.

Logs all agentic decisions, enables learning from feedback, and calibrates over time.

Every agentic tool (adversarial, structural, etc.) logs its decisions here.
Humans can provide feedback (valid/false_positive) to calibrate thresholds.

Storage: JSON files in .agent_memory/ directory
"""

import fcntl
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

# Default storage location
DEFAULT_MEMORY_DIR = Path.cwd() / ".agent_memory"
LOG_RETENTION_DAYS = 30


@dataclass
class AgentDecision:
    """A single decision made by an agentic tool."""
    timestamp: str
    tool: str  # "adversarial", "structural", etc.
    question_hash: str
    question_preview: str
    decision: str  # "FLAG", "PASS", etc.
    confidence: float  # 0-1
    details: Dict[str, Any]
    feedback: Optional[Literal["valid", "false_positive", "needs_review"]] = None
    feedback_notes: Optional[str] = None


class AgentMemory:
    """
    Manages agent decision logging and calibration.

    Usage:
        memory = AgentMemory()  # Uses .agent_memory/ in cwd
        memory = AgentMemory("/path/to/memory")  # Custom location

        # Log a decision
        decision_id = memory.log_decision(
            tool="adversarial",
            question={"text": "...", "options": {...}},
            decision="FLAG",
            confidence=0.8,
            details={"verdict": "AMBIGUOUS", ...}
        )

        # Get calibration
        cal = memory.get_calibration("adversarial")
        threshold_adjustment = cal.get("threshold_adjustment", 0)

        # Add feedback
        memory.add_feedback(decision_id, "valid")
    """

    def __init__(self, memory_dir: Optional[Path] = None):
        self.memory_dir = Path(memory_dir) if memory_dir else DEFAULT_MEMORY_DIR
        self.logs_dir = self.memory_dir / "logs"
        self.calibration_file = self.memory_dir / "calibration.json"
        self.feedback_file = self.memory_dir / "feedback.json"
        self._ensure_dirs()

    def _ensure_dirs(self):
        """Create memory directories and clean up old logs."""
        self.memory_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)

        # Create .gitignore
        gitignore = self.memory_dir / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text("logs/\n")

        self._cleanup_old_logs()

    def _cleanup_old_logs(self):
        """Remove log files older than LOG_RETENTION_DAYS."""
        if not self.logs_dir.exists():
            return

        cutoff = datetime.now().timestamp() - (LOG_RETENTION_DAYS * 86400)
        for log_file in self.logs_dir.glob("*.jsonl"):
            if log_file.stat().st_mtime < cutoff:
                log_file.unlink()

    def _hash_question(self, question: Dict[str, Any]) -> str:
        """Create stable hash of question for deduplication."""
        content = json.dumps({
            "text": question.get("text", ""),
            "options": question.get("options", {}),
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _get_log_file(self, tool: str) -> Path:
        """Get log file path for a tool (one file per day per tool)."""
        date = datetime.now().strftime("%Y-%m-%d")
        return self.logs_dir / f"{tool}_{date}.jsonl"

    def log_decision(
        self,
        tool: str,
        question: Dict[str, Any],
        decision: str,
        confidence: float,
        details: Dict[str, Any],
    ) -> str:
        """
        Log an agentic decision.

        Returns: decision_id for later feedback
        """
        question_hash = self._hash_question(question)
        question_preview = question.get("text", "")[:100]

        record = AgentDecision(
            timestamp=datetime.now().isoformat(),
            tool=tool,
            question_hash=question_hash,
            question_preview=question_preview,
            decision=decision,
            confidence=confidence,
            details=details,
        )

        # Append to daily log file with file locking
        log_file = self._get_log_file(tool)
        with open(log_file, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(asdict(record)) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return f"{tool}:{question_hash}:{record.timestamp}"

    def add_feedback(
        self,
        decision_id: str,
        feedback: Literal["valid", "false_positive", "needs_review"],
        notes: Optional[str] = None,
    ):
        """
        Add human feedback to a decision.

        - "valid" = the flag was correct
        - "false_positive" = shouldn't have flagged
        - "needs_review" = unclear
        """
        feedback_data = {}
        if self.feedback_file.exists():
            feedback_data = json.loads(self.feedback_file.read_text())

        feedback_data[decision_id] = {
            "feedback": feedback,
            "notes": notes,
            "timestamp": datetime.now().isoformat(),
        }

        self.feedback_file.write_text(json.dumps(feedback_data, indent=2))
        self._recalibrate()

    def _recalibrate(self):
        """Recalibrate thresholds based on feedback."""
        if not self.feedback_file.exists():
            return

        feedback_data = json.loads(self.feedback_file.read_text())

        # Count feedback by tool
        stats: Dict[str, Dict[str, int]] = {}
        for decision_id, fb in feedback_data.items():
            tool = decision_id.split(":")[0]
            if tool not in stats:
                stats[tool] = {"valid": 0, "false_positive": 0, "needs_review": 0, "total": 0}

            stats[tool][fb["feedback"]] += 1
            stats[tool]["total"] += 1

        # Calculate calibration adjustments
        calibration = {"updated": datetime.now().isoformat(), "tools": {}}

        for tool, counts in stats.items():
            if counts["total"] < 5:
                continue

            fp_rate = counts["false_positive"] / counts["total"]
            valid_rate = counts["valid"] / counts["total"]

            calibration["tools"][tool] = {
                "false_positive_rate": round(fp_rate, 3),
                "valid_rate": round(valid_rate, 3),
                "total_feedback": counts["total"],
                "recommendation": self._get_recommendation(fp_rate),
            }

        self.calibration_file.write_text(json.dumps(calibration, indent=2))

    def _get_recommendation(self, fp_rate: float) -> str:
        """Get recommendation based on false positive rate."""
        if fp_rate > 0.5:
            return "RAISE_THRESHOLD - Too many false positives"
        elif fp_rate > 0.3:
            return "SLIGHTLY_RAISE - Some false positives"
        elif fp_rate < 0.1:
            return "LOWER_THRESHOLD - Very few false positives, could catch more"
        else:
            return "GOOD - Threshold well calibrated"

    def get_calibration(self, tool: str) -> Dict[str, Any]:
        """
        Get current calibration for a tool.

        Returns threshold adjustments based on feedback.
        """
        if not self.calibration_file.exists():
            return {"calibrated": False, "threshold_adjustment": 0}

        calibration = json.loads(self.calibration_file.read_text())
        tool_cal = calibration.get("tools", {}).get(tool, {})

        if not tool_cal:
            return {"calibrated": False, "threshold_adjustment": 0}

        fp_rate = tool_cal.get("false_positive_rate", 0)

        # Adjust threshold based on false positive rate
        if fp_rate > 0.5:
            adjustment = 1
        elif fp_rate > 0.3:
            adjustment = 0.5
        elif fp_rate < 0.1 and tool_cal.get("total_feedback", 0) >= 10:
            adjustment = -0.5
        else:
            adjustment = 0

        return {
            "calibrated": True,
            "threshold_adjustment": adjustment,
            "false_positive_rate": fp_rate,
            "recommendation": tool_cal.get("recommendation", ""),
            "total_feedback": tool_cal.get("total_feedback", 0),
        }

    def get_recent_decisions(
        self,
        tool: Optional[str] = None,
        limit: int = 20,
        only_flags: bool = False,
    ) -> List[Dict[str, Any]]:
        """Get recent decisions for review."""
        decisions = []

        log_files = sorted(self.logs_dir.glob("*.jsonl"), reverse=True)

        for log_file in log_files:
            if tool and not log_file.name.startswith(tool):
                continue

            with open(log_file) as f:
                for line in f:
                    record = json.loads(line)
                    if only_flags and record["decision"] != "FLAG":
                        continue
                    decisions.append(record)

                    if len(decisions) >= limit:
                        break

            if len(decisions) >= limit:
                break

        return decisions

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of what the system has learned."""
        total_decisions = 0
        decisions_by_tool: Dict[str, int] = {}

        for log_file in self.logs_dir.glob("*.jsonl"):
            tool = log_file.name.split("_")[0]
            count = sum(1 for _ in open(log_file))
            total_decisions += count
            decisions_by_tool[tool] = decisions_by_tool.get(tool, 0) + count

        calibration = {}
        if self.calibration_file.exists():
            calibration = json.loads(self.calibration_file.read_text())

        feedback_counts = {"valid": 0, "false_positive": 0, "needs_review": 0}
        if self.feedback_file.exists():
            feedback_data = json.loads(self.feedback_file.read_text())
            for fb in feedback_data.values():
                feedback_counts[fb["feedback"]] += 1

        total_feedback = sum(feedback_counts.values())

        return {
            "total_decisions_logged": total_decisions,
            "decisions_by_tool": decisions_by_tool,
            "total_feedback_received": total_feedback,
            "feedback_breakdown": feedback_counts,
            "calibration": calibration.get("tools", {}),
            "learning_status": self._get_learning_status(total_decisions, total_feedback),
        }

    def _get_learning_status(self, total: int, feedback: int) -> str:
        """Describe current learning status."""
        if total == 0:
            return "NO_DATA - No decisions logged yet"
        elif feedback == 0:
            return "NEEDS_FEEDBACK - Decisions logged but no feedback provided"
        elif feedback < 10:
            return "LEARNING - Some feedback, need more for reliable calibration"
        elif feedback < 50:
            return "CALIBRATING - Good amount of feedback, calibration improving"
        else:
            return "CALIBRATED - Strong feedback history, thresholds well-tuned"
