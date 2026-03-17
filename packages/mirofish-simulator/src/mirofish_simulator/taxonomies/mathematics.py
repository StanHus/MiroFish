"""
Mathematics misconception taxonomy.

Defines common student misconceptions in mathematics,
organized by category with remediation standards.
"""

from typing import Dict, Optional


# Misconception taxonomy organized by category
MATHEMATICS_MISCONCEPTIONS: Dict[str, Dict] = {
    # ── Arithmetic Errors ──────────────────────────────────────────────────────
    "sign_error_multiplication": {
        "id": "sign_error_multiplication",
        "category": "arithmetic_error",
        "description": "Incorrect handling of signs when multiplying negative numbers",
        "common_triggers": [
            "negative times negative",
            "multiply negatives",
            "-3 * -2",
        ],
        "vulnerable_archetypes": ["esl_student", "class_clown", "disengaged_but_smart"],
        "remediation_standards": ["CCSS.MATH.CONTENT.7.NS.A.2"],
        "remediation_topic": "Multiplication of integers with different signs",
    },
    "sign_error_subtraction": {
        "id": "sign_error_subtraction",
        "category": "arithmetic_error",
        "description": "Errors when subtracting negative numbers (double negatives)",
        "common_triggers": [
            "minus negative",
            "subtract negative",
            "5 - (-3)",
        ],
        "vulnerable_archetypes": ["esl_student", "class_clown"],
        "remediation_standards": ["CCSS.MATH.CONTENT.7.NS.A.1"],
        "remediation_topic": "Subtracting integers",
    },
    "order_of_operations": {
        "id": "order_of_operations",
        "category": "arithmetic_error",
        "description": "Incorrect application of PEMDAS/BODMAS",
        "common_triggers": [
            "order of operations",
            "PEMDAS",
            "2 + 3 * 4",
        ],
        "vulnerable_archetypes": ["class_clown", "disengaged_but_smart"],
        "remediation_standards": ["CCSS.MATH.CONTENT.5.OA.A.1"],
        "remediation_topic": "Order of operations conventions",
    },
    "fraction_addition_error": {
        "id": "fraction_addition_error",
        "category": "arithmetic_error",
        "description": "Adding fractions by adding numerators and denominators separately",
        "common_triggers": [
            "add fractions",
            "1/2 + 1/3",
            "common denominator",
        ],
        "vulnerable_archetypes": ["esl_student", "class_clown", "quiet_thinker"],
        "remediation_standards": ["CCSS.MATH.CONTENT.5.NF.A.1"],
        "remediation_topic": "Adding fractions with unlike denominators",
    },
    "decimal_place_value": {
        "id": "decimal_place_value",
        "category": "arithmetic_error",
        "description": "Misunderstanding decimal place values (0.5 vs 0.05)",
        "common_triggers": [
            "decimal",
            "0.5",
            "tenths hundredths",
        ],
        "vulnerable_archetypes": ["esl_student", "class_clown"],
        "remediation_standards": ["CCSS.MATH.CONTENT.5.NBT.A.3"],
        "remediation_topic": "Decimal place value understanding",
    },

    # ── Algebraic Misconceptions ───────────────────────────────────────────────
    "distribution_error": {
        "id": "distribution_error",
        "category": "algebraic_error",
        "description": "Incorrect distribution of multiplication over addition/subtraction",
        "common_triggers": [
            "distribute",
            "2(x + 3)",
            "expand",
        ],
        "vulnerable_archetypes": ["disengaged_but_smart", "class_clown"],
        "remediation_standards": ["CCSS.MATH.CONTENT.6.EE.A.3"],
        "remediation_topic": "Distributive property",
    },
    "exponent_multiplication": {
        "id": "exponent_multiplication",
        "category": "algebraic_error",
        "description": "Multiplying exponents instead of adding when multiplying like bases",
        "common_triggers": [
            "x^2 * x^3",
            "exponent rules",
            "multiply powers",
        ],
        "vulnerable_archetypes": ["disengaged_but_smart", "quiet_thinker"],
        "remediation_standards": ["CCSS.MATH.CONTENT.8.EE.A.1"],
        "remediation_topic": "Properties of integer exponents",
    },
    "exponent_addition": {
        "id": "exponent_addition",
        "category": "algebraic_error",
        "description": "Adding exponents when adding terms (x^2 + x^2 = x^4)",
        "common_triggers": [
            "x^2 + x^2",
            "add like terms",
            "combine exponents",
        ],
        "vulnerable_archetypes": ["class_clown", "disengaged_but_smart"],
        "remediation_standards": ["CCSS.MATH.CONTENT.6.EE.A.4"],
        "remediation_topic": "Combining like terms vs. exponent rules",
    },
    "variable_as_label": {
        "id": "variable_as_label",
        "category": "algebraic_error",
        "description": "Treating variables as labels rather than quantities (3a + 2b = 5ab)",
        "common_triggers": [
            "combine terms",
            "3a + 2b",
            "like terms",
        ],
        "vulnerable_archetypes": ["esl_student", "class_clown"],
        "remediation_standards": ["CCSS.MATH.CONTENT.6.EE.A.2"],
        "remediation_topic": "Understanding variables as quantities",
    },
    "equation_balance": {
        "id": "equation_balance",
        "category": "algebraic_error",
        "description": "Not maintaining balance when solving equations",
        "common_triggers": [
            "solve for x",
            "both sides",
            "isolate variable",
        ],
        "vulnerable_archetypes": ["disengaged_but_smart", "class_clown"],
        "remediation_standards": ["CCSS.MATH.CONTENT.6.EE.B.7"],
        "remediation_topic": "Solving equations by maintaining equality",
    },
    "negative_coefficient_error": {
        "id": "negative_coefficient_error",
        "category": "algebraic_error",
        "description": "Errors when dividing by negative coefficients",
        "common_triggers": [
            "-2x = 6",
            "divide by negative",
            "negative coefficient",
        ],
        "vulnerable_archetypes": ["quiet_thinker", "disengaged_but_smart"],
        "remediation_standards": ["CCSS.MATH.CONTENT.7.EE.B.4"],
        "remediation_topic": "Solving equations with negative coefficients",
    },

    # ── Function Misconceptions ────────────────────────────────────────────────
    "slope_intercept_confusion": {
        "id": "slope_intercept_confusion",
        "category": "function_error",
        "description": "Confusing slope with y-intercept in y = mx + b",
        "common_triggers": [
            "slope intercept",
            "y = mx + b",
            "what is m",
        ],
        "vulnerable_archetypes": ["esl_student", "disengaged_but_smart"],
        "remediation_standards": ["CCSS.MATH.CONTENT.8.F.B.4"],
        "remediation_topic": "Slope-intercept form understanding",
    },
    "function_notation": {
        "id": "function_notation",
        "category": "function_error",
        "description": "Interpreting f(x) as f times x rather than function notation",
        "common_triggers": [
            "f(x)",
            "function notation",
            "f(3)",
        ],
        "vulnerable_archetypes": ["esl_student", "class_clown", "disengaged_but_smart"],
        "remediation_standards": ["CCSS.MATH.CONTENT.HSF.IF.A.1"],
        "remediation_topic": "Function notation and evaluation",
    },
    "inverse_operation_confusion": {
        "id": "inverse_operation_confusion",
        "category": "function_error",
        "description": "Confusing inverse functions with reciprocals",
        "common_triggers": [
            "inverse function",
            "f^-1",
            "undo function",
        ],
        "vulnerable_archetypes": ["quiet_thinker", "disengaged_but_smart"],
        "remediation_standards": ["CCSS.MATH.CONTENT.HSF.BF.B.4"],
        "remediation_topic": "Inverse functions vs. multiplicative inverses",
    },

    # ── Geometry Misconceptions ────────────────────────────────────────────────
    "area_perimeter_confusion": {
        "id": "area_perimeter_confusion",
        "category": "geometry_error",
        "description": "Confusing area with perimeter formulas",
        "common_triggers": [
            "area of rectangle",
            "perimeter",
            "length times width",
        ],
        "vulnerable_archetypes": ["esl_student", "class_clown"],
        "remediation_standards": ["CCSS.MATH.CONTENT.3.MD.D.8"],
        "remediation_topic": "Distinguishing area from perimeter",
    },
    "angle_sum_triangle": {
        "id": "angle_sum_triangle",
        "category": "geometry_error",
        "description": "Incorrectly applying or misremembering triangle angle sum (180 degrees)",
        "common_triggers": [
            "triangle angles",
            "sum of angles",
            "180 degrees",
        ],
        "vulnerable_archetypes": ["disengaged_but_smart", "class_clown"],
        "remediation_standards": ["CCSS.MATH.CONTENT.8.G.A.5"],
        "remediation_topic": "Triangle angle sum theorem",
    },
    "similar_vs_congruent": {
        "id": "similar_vs_congruent",
        "category": "geometry_error",
        "description": "Confusing similar shapes with congruent shapes",
        "common_triggers": [
            "similar triangles",
            "congruent",
            "same shape",
        ],
        "vulnerable_archetypes": ["esl_student", "quiet_thinker"],
        "remediation_standards": ["CCSS.MATH.CONTENT.8.G.A.4"],
        "remediation_topic": "Similarity vs. congruence",
    },
    "pythagorean_theorem_misuse": {
        "id": "pythagorean_theorem_misuse",
        "category": "geometry_error",
        "description": "Applying Pythagorean theorem to non-right triangles or incorrectly",
        "common_triggers": [
            "a^2 + b^2",
            "pythagorean",
            "hypotenuse",
        ],
        "vulnerable_archetypes": ["disengaged_but_smart", "quiet_thinker"],
        "remediation_standards": ["CCSS.MATH.CONTENT.8.G.B.7"],
        "remediation_topic": "Pythagorean theorem application",
    },

    # ── Proportional Reasoning Errors ──────────────────────────────────────────
    "additive_vs_multiplicative": {
        "id": "additive_vs_multiplicative",
        "category": "proportional_error",
        "description": "Using additive reasoning instead of multiplicative for proportions",
        "common_triggers": [
            "proportion",
            "ratio",
            "scale",
        ],
        "vulnerable_archetypes": ["esl_student", "class_clown"],
        "remediation_standards": ["CCSS.MATH.CONTENT.6.RP.A.3"],
        "remediation_topic": "Proportional relationships",
    },
    "percentage_of_vs_percentage_increase": {
        "id": "percentage_of_vs_percentage_increase",
        "category": "proportional_error",
        "description": "Confusing percentage of a number with percentage increase/decrease",
        "common_triggers": [
            "percent increase",
            "50% of",
            "percent change",
        ],
        "vulnerable_archetypes": ["esl_student", "disengaged_but_smart", "class_clown"],
        "remediation_standards": ["CCSS.MATH.CONTENT.7.RP.A.3"],
        "remediation_topic": "Percent problems and percent change",
    },
    "unit_rate_confusion": {
        "id": "unit_rate_confusion",
        "category": "proportional_error",
        "description": "Errors in calculating or interpreting unit rates",
        "common_triggers": [
            "unit rate",
            "per hour",
            "miles per gallon",
        ],
        "vulnerable_archetypes": ["esl_student", "class_clown"],
        "remediation_standards": ["CCSS.MATH.CONTENT.6.RP.A.2"],
        "remediation_topic": "Unit rates and ratios",
    },

    # ── Statistical Misconceptions ─────────────────────────────────────────────
    "mean_median_mode_confusion": {
        "id": "mean_median_mode_confusion",
        "category": "statistics_error",
        "description": "Confusing mean, median, and mode calculations or uses",
        "common_triggers": [
            "average",
            "median",
            "mode",
        ],
        "vulnerable_archetypes": ["esl_student", "class_clown", "quiet_thinker"],
        "remediation_standards": ["CCSS.MATH.CONTENT.6.SP.B.5"],
        "remediation_topic": "Measures of center",
    },
    "probability_misconception": {
        "id": "probability_misconception",
        "category": "statistics_error",
        "description": "Gambler's fallacy or misunderstanding of independent events",
        "common_triggers": [
            "probability",
            "coin flip",
            "independent events",
        ],
        "vulnerable_archetypes": ["socially_engaged_activist", "quiet_thinker"],
        "remediation_standards": ["CCSS.MATH.CONTENT.7.SP.C.5"],
        "remediation_topic": "Probability of independent events",
    },
}


def get_math_misconception(misconception_id: str) -> Optional[Dict]:
    """Get a specific mathematics misconception by ID."""
    return MATHEMATICS_MISCONCEPTIONS.get(misconception_id)


def get_misconceptions_by_category(category: str) -> Dict[str, Dict]:
    """Get all misconceptions in a specific category."""
    return {
        k: v for k, v in MATHEMATICS_MISCONCEPTIONS.items()
        if v.get("category") == category
    }


def get_misconceptions_for_archetype(archetype: str) -> Dict[str, Dict]:
    """Get misconceptions that commonly affect a specific archetype."""
    return {
        k: v for k, v in MATHEMATICS_MISCONCEPTIONS.items()
        if archetype in v.get("vulnerable_archetypes", [])
    }


# Category descriptions for reporting
CATEGORY_DESCRIPTIONS = {
    "arithmetic_error": "Errors in basic arithmetic operations",
    "algebraic_error": "Misconceptions in algebraic manipulation",
    "function_error": "Misunderstandings about functions and their notation",
    "geometry_error": "Errors in geometric reasoning",
    "proportional_error": "Misconceptions in proportional reasoning",
    "statistics_error": "Misunderstandings in statistics and probability",
}
