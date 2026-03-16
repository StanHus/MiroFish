"""
Shared utilities for the evaluator pipeline.
Shared evaluation utilities.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def load_prompts(prompts_file: Optional[str] = None) -> Dict[str, Any]:
    if prompts_file is None:
        prompts_file = Path(__file__).parent / "config" / "prompts.json"
    with open(prompts_file, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_xml_response(response_text: str) -> Tuple[int, str]:
    """Parse <quality_check><score>0|1</score><reasoning>...</reasoning></quality_check>"""
    try:
        if "<quality_check>" in response_text:
            import xml.etree.ElementTree as ET
            match = re.search(r"<quality_check>(.*?)</quality_check>", response_text, re.DOTALL)
            if match:
                try:
                    root = ET.fromstring(f"<quality_check>{match.group(1)}</quality_check>")
                    score_el = root.find("score")
                    reason_el = root.find("reasoning")
                    if score_el is not None and score_el.text:
                        score = 1 if int(score_el.text.strip()) > 0 else 0
                        reasoning = reason_el.text.strip() if reason_el is not None and reason_el.text else ""
                        return score, reasoning
                except ET.ParseError:
                    pass
            # Partial
            sm = re.search(r"<score>(\d+)</score>", response_text)
            rm = re.search(r"<reasoning>(.*?)(?:</reasoning>|$)", response_text, re.DOTALL)
            if sm:
                return (1 if int(sm.group(1)) > 0 else 0), (rm.group(1).strip() if rm else "")

        # Fallback
        if "[1]" in response_text:
            return 1, "Contains [1]"
        if "[0]" in response_text:
            return 0, "Contains [0]"
        low = response_text.lower()
        if any(w in low for w in ["correct", "good", "appropriate", "passes"]):
            return 1, "Positive keywords"
        return 0, "No clear positive indicators"
    except Exception as e:
        logger.warning(f"Parse error: {e}")
        return 0, f"Parse error: {e}"


def parse_json_response(response_text: str) -> Optional[Dict[str, Any]]:
    try:
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            return json.loads(response_text[start:end].strip())
        if "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            content = response_text[start:end].strip()
            if content.startswith("{") or content.startswith("["):
                return json.loads(content)
        return json.loads(response_text)
    except json.JSONDecodeError:
        logger.warning(f"JSON parse failed: {response_text[:200]}")
        return None


def fill_prompt_variables(template: str, variables: Dict[str, Any]) -> str:
    filled = template
    for var, value in variables.items():
        filled = filled.replace(f"{{{var}}}", str(value))
        filled = filled.replace(f"[{var.upper()}]", str(value))
    return filled


def calculate_pass_rate(results: List[Dict]) -> Dict[str, Any]:
    if not results:
        return {"total": 0, "passed": 0, "failed": 0, "pass_rate": 0.0, "average_score": 0.0}
    total = len(results)
    passed = sum(1 for r in results if r.get("overall_score", 0) >= 0.8)
    avg = sum(r.get("overall_score", 0) for r in results) / total
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": passed / total,
        "average_score": avg,
    }
