"""
InceptBench Integration API Routes

Provides endpoints for InceptBench to simulate student responses to educational content.

Endpoints:
    POST /api/inceptbench/simulate - Simulate student population responding to content
    POST /api/inceptbench/batch    - Batch simulation for multiple content items
    GET  /api/inceptbench/health   - Health check for InceptBench integration
"""

import asyncio
import traceback
from flask import request, jsonify

from . import inceptbench_bp
from ..services.inceptbench_simulator import InceptBenchSimulator, DEFAULT_POPULATION
from ..utils.logger import get_logger

logger = get_logger('mirofish.api.inceptbench')

# Singleton simulator instance
_simulator: InceptBenchSimulator = None


def get_simulator() -> InceptBenchSimulator:
    """Get or create the simulator instance."""
    global _simulator
    if _simulator is None:
        _simulator = InceptBenchSimulator()
    return _simulator


# ── Health Check ────────────────────────────────────────────────────────────

@inceptbench_bp.route('/health', methods=['GET'])
def health():
    """
    Health check for InceptBench integration.

    Returns:
        JSON with status and capabilities
    """
    return jsonify({
        "status": "ok",
        "service": "MiroFish InceptBench Integration",
        "version": "1.0.0",
        "capabilities": [
            "student_simulation",
            "psychometric_analysis",
            "distractor_analysis",
            "archetype_breakdown",
        ],
        "default_population": DEFAULT_POPULATION,
    })


# ── Single Content Simulation ───────────────────────────────────────────────

@inceptbench_bp.route('/simulate', methods=['POST'])
def simulate_content():
    """
    Simulate student population responding to educational content.

    Request Body:
        {
            "content": {
                "type": "question",
                "text": "What is 2+2?",
                "options": ["3", "4", "5", "6"],
                "correct_answer": "B",
                "grade": "3",
                "subject": "mathematics"
            },
            "population": {
                "size": 30,
                "archetypes": ["honors_overachiever", "quiet_thinker", ...],
                "distribution": {"honors_overachiever": 0.2, ...}
            },
            "include_responses": false
        }

    Returns:
        {
            "success": true,
            "data": {
                "aggregate": {...},
                "by_archetype": {...},
                "distractor_analysis": {...},
                "concerns": [...],
                "recommendations": [...]
            }
        }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "error": "Request body is required"
            }), 400

        content = data.get("content")
        if not content:
            return jsonify({
                "success": False,
                "error": "content field is required"
            }), 400

        # Validate content has minimum required fields
        if not content.get("text") and not content.get("question"):
            return jsonify({
                "success": False,
                "error": "content must have 'text' or 'question' field"
            }), 400

        # Normalize content
        if "question" in content and "text" not in content:
            content["text"] = content["question"]

        population_config = data.get("population", {})
        include_responses = data.get("include_responses", False)

        logger.info(
            f"Simulating content: type={content.get('type', 'question')}, "
            f"population_size={population_config.get('size', 30)}"
        )

        # Run simulation
        simulator = get_simulator()
        result = asyncio.run(
            simulator.simulate_content(
                content=content,
                population_config=population_config,
            )
        )

        # Convert to dict
        result_dict = result.to_dict()

        # Optionally include raw responses
        if include_responses:
            result_dict["responses"] = [
                {
                    "student_id": r.student_id,
                    "archetype": r.archetype,
                    "selected_answer": r.selected_answer,
                    "is_correct": r.is_correct,
                    "confidence": r.confidence,
                    "time_seconds": r.time_seconds,
                    "reasoning": r.reasoning,
                }
                for r in result.responses
            ]

        return jsonify({
            "success": True,
            "data": result_dict
        })

    except Exception as e:
        logger.error(f"Simulation failed: {e}\n{traceback.format_exc()}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ── Batch Simulation ────────────────────────────────────────────────────────

@inceptbench_bp.route('/batch', methods=['POST'])
def batch_simulate():
    """
    Batch simulation for multiple content items.

    Request Body:
        {
            "contents": [
                {"type": "question", "text": "...", "options": [...], ...},
                {"type": "question", "text": "...", "options": [...], ...}
            ],
            "population": {
                "size": 30,
                ...
            }
        }

    Returns:
        {
            "success": true,
            "data": {
                "results": [
                    {"content_index": 0, "result": {...}},
                    {"content_index": 1, "result": {...}}
                ],
                "summary": {
                    "total_items": 2,
                    "avg_accuracy": 0.75,
                    "avg_difficulty": 0.3,
                    "total_concerns": 3
                }
            }
        }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "error": "Request body is required"
            }), 400

        contents = data.get("contents", [])
        if not contents:
            return jsonify({
                "success": False,
                "error": "contents array is required"
            }), 400

        if len(contents) > 50:
            return jsonify({
                "success": False,
                "error": "Maximum 50 content items per batch"
            }), 400

        population_config = data.get("population", {})

        logger.info(f"Batch simulating {len(contents)} content items")

        simulator = get_simulator()
        results = []
        total_accuracy = 0
        total_difficulty = 0
        total_concerns = 0

        for i, content in enumerate(contents):
            try:
                # Normalize content
                if "question" in content and "text" not in content:
                    content["text"] = content["question"]

                result = asyncio.run(
                    simulator.simulate_content(
                        content=content,
                        population_config=population_config,
                    )
                )

                result_dict = result.to_dict()
                results.append({
                    "content_index": i,
                    "success": True,
                    "result": result_dict
                })

                total_accuracy += result.accuracy
                total_difficulty += result.difficulty_irt
                total_concerns += len(result.concerns)

            except Exception as e:
                logger.warning(f"Failed to simulate content {i}: {e}")
                results.append({
                    "content_index": i,
                    "success": False,
                    "error": str(e)
                })

        n_success = sum(1 for r in results if r.get("success"))

        return jsonify({
            "success": True,
            "data": {
                "results": results,
                "summary": {
                    "total_items": len(contents),
                    "successful": n_success,
                    "failed": len(contents) - n_success,
                    "avg_accuracy": total_accuracy / n_success if n_success > 0 else 0,
                    "avg_difficulty": total_difficulty / n_success if n_success > 0 else 0,
                    "total_concerns": total_concerns,
                }
            }
        })

    except Exception as e:
        logger.error(f"Batch simulation failed: {e}\n{traceback.format_exc()}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ── Population Archetypes Info ──────────────────────────────────────────────

@inceptbench_bp.route('/archetypes', methods=['GET'])
def list_archetypes():
    """
    List available student archetypes and their characteristics.

    Returns:
        JSON with archetype details
    """
    archetypes = {
        "honors_overachiever": {
            "description": "High-achieving student focused on academic excellence",
            "typical_accuracy": 0.92,
            "engagement": "high",
            "characteristics": ["detail-oriented", "thorough", "perfectionist"],
        },
        "debate_club_kid": {
            "description": "Intellectually curious, loves to argue and discuss",
            "typical_accuracy": 0.78,
            "engagement": "high",
            "characteristics": ["argumentative", "quick-thinking", "confident"],
        },
        "quiet_thinker": {
            "description": "Thoughtful introvert who processes deeply",
            "typical_accuracy": 0.70,
            "engagement": "medium",
            "characteristics": ["reflective", "careful", "reserved"],
        },
        "socially_engaged_activist": {
            "description": "Passionate about social issues and justice",
            "typical_accuracy": 0.68,
            "engagement": "high",
            "characteristics": ["passionate", "opinionated", "empathetic"],
        },
        "disengaged_but_smart": {
            "description": "Capable but unmotivated student",
            "typical_accuracy": 0.55,
            "engagement": "low",
            "characteristics": ["intelligent", "bored", "minimal-effort"],
        },
        "esl_student": {
            "description": "English language learner with strong effort",
            "typical_accuracy": 0.45,
            "engagement": "medium",
            "characteristics": ["hardworking", "language-challenged", "persistent"],
        },
        "class_clown": {
            "description": "Prioritizes humor over academics",
            "typical_accuracy": 0.35,
            "engagement": "low",
            "characteristics": ["humorous", "distractible", "social"],
        },
        "politically_conservative": {
            "description": "Student with traditional/conservative viewpoints",
            "typical_accuracy": 0.65,
            "engagement": "medium",
            "characteristics": ["traditional", "structured", "rule-following"],
        },
    }

    return jsonify({
        "success": True,
        "data": {
            "archetypes": archetypes,
            "default_distribution": DEFAULT_POPULATION["distribution"],
        }
    })
