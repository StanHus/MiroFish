"""
Misconception taxonomies for different subjects.

Each taxonomy defines common student misconceptions that can be mapped
to distractor choices in educational assessments.
"""

from .ap_government import AP_GOVERNMENT_MISCONCEPTIONS, get_ap_gov_misconception
from .mathematics import MATHEMATICS_MISCONCEPTIONS, get_math_misconception

# Subject to taxonomy mapping
SUBJECT_TAXONOMIES = {
    "ap_government": AP_GOVERNMENT_MISCONCEPTIONS,
    "ap government": AP_GOVERNMENT_MISCONCEPTIONS,
    "ap gov": AP_GOVERNMENT_MISCONCEPTIONS,
    "civics": AP_GOVERNMENT_MISCONCEPTIONS,
    "government": AP_GOVERNMENT_MISCONCEPTIONS,
    "mathematics": MATHEMATICS_MISCONCEPTIONS,
    "math": MATHEMATICS_MISCONCEPTIONS,
    "algebra": MATHEMATICS_MISCONCEPTIONS,
    "calculus": MATHEMATICS_MISCONCEPTIONS,
    "geometry": MATHEMATICS_MISCONCEPTIONS,
}


def get_taxonomy(subject: str) -> dict:
    """Get the misconception taxonomy for a subject."""
    subject_lower = subject.lower().strip()
    return SUBJECT_TAXONOMIES.get(subject_lower, {})


def get_misconception(subject: str, misconception_id: str) -> dict:
    """Get a specific misconception by subject and ID."""
    subject_lower = subject.lower().strip()

    if subject_lower in ["ap_government", "ap government", "ap gov", "civics", "government"]:
        return get_ap_gov_misconception(misconception_id)
    elif subject_lower in ["mathematics", "math", "algebra", "calculus", "geometry"]:
        return get_math_misconception(misconception_id)

    return {}


__all__ = [
    "AP_GOVERNMENT_MISCONCEPTIONS",
    "MATHEMATICS_MISCONCEPTIONS",
    "SUBJECT_TAXONOMIES",
    "get_taxonomy",
    "get_misconception",
    "get_ap_gov_misconception",
    "get_math_misconception",
]
