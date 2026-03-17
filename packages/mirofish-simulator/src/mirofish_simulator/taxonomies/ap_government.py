"""
AP Government and Civics misconception taxonomy.

Defines common student misconceptions in US Government and Politics,
organized by category with remediation standards.
"""

from typing import Dict, Optional


# Misconception taxonomy organized by category
AP_GOVERNMENT_MISCONCEPTIONS: Dict[str, Dict] = {
    # ── Institutional Confusion ────────────────────────────────────────────────
    "confuses_electoral_college_with_congress": {
        "id": "confuses_electoral_college_with_congress",
        "category": "institutional_confusion",
        "description": "Conflates the Electoral College with Congressional functions",
        "common_triggers": [
            "electoral college",
            "electing congress",
            "vote for president",
        ],
        "vulnerable_archetypes": ["esl_student", "disengaged_but_smart", "class_clown"],
        "remediation_standards": ["CCSS.ELA-LITERACY.RH.11-12.7"],
        "remediation_topic": "Separation of electoral processes for different branches",
    },
    "confuses_branches_of_government": {
        "id": "confuses_branches_of_government",
        "category": "institutional_confusion",
        "description": "Mixes up powers and responsibilities of executive, legislative, and judicial branches",
        "common_triggers": [
            "who makes laws",
            "presidential powers",
            "supreme court role",
        ],
        "vulnerable_archetypes": ["esl_student", "class_clown", "quiet_thinker"],
        "remediation_standards": ["CCSS.ELA-LITERACY.RH.11-12.1"],
        "remediation_topic": "Separation of powers and checks and balances",
    },
    "confuses_federal_and_state": {
        "id": "confuses_federal_and_state",
        "category": "institutional_confusion",
        "description": "Conflates federal government powers with state government powers",
        "common_triggers": [
            "federalism",
            "state rights",
            "national vs local",
        ],
        "vulnerable_archetypes": ["esl_student", "socially_engaged_activist"],
        "remediation_standards": ["CCSS.ELA-LITERACY.RH.11-12.7"],
        "remediation_topic": "Federalism and the division of powers",
    },

    # ── Constitutional Misunderstandings ───────────────────────────────────────
    "misunderstands_amendment_process": {
        "id": "misunderstands_amendment_process",
        "category": "constitutional_misunderstanding",
        "description": "Incorrectly describes how the Constitution is amended",
        "common_triggers": [
            "amending constitution",
            "ratification",
            "two-thirds",
        ],
        "vulnerable_archetypes": ["disengaged_but_smart", "class_clown"],
        "remediation_standards": ["CCSS.ELA-LITERACY.RH.11-12.2"],
        "remediation_topic": "Article V amendment procedures",
    },
    "misunderstands_bill_of_rights_scope": {
        "id": "misunderstands_bill_of_rights_scope",
        "category": "constitutional_misunderstanding",
        "description": "Believes Bill of Rights applies to private entities or misunderstands its limits",
        "common_triggers": [
            "first amendment",
            "free speech",
            "private company",
        ],
        "vulnerable_archetypes": ["socially_engaged_activist", "debate_club_kid"],
        "remediation_standards": ["CCSS.ELA-LITERACY.RH.11-12.6"],
        "remediation_topic": "State action doctrine and constitutional limits",
    },
    "confuses_rights_with_privileges": {
        "id": "confuses_rights_with_privileges",
        "category": "constitutional_misunderstanding",
        "description": "Conflates constitutionally protected rights with privileges or benefits",
        "common_triggers": [
            "right to vote",
            "driving license",
            "healthcare",
        ],
        "vulnerable_archetypes": ["quiet_thinker", "politically_conservative"],
        "remediation_standards": ["CCSS.ELA-LITERACY.RH.11-12.9"],
        "remediation_topic": "Distinction between rights, privileges, and entitlements",
    },

    # ── Electoral Process Errors ───────────────────────────────────────────────
    "misunderstands_primary_vs_general": {
        "id": "misunderstands_primary_vs_general",
        "category": "electoral_process_error",
        "description": "Confuses primary elections with general elections",
        "common_triggers": [
            "primary election",
            "general election",
            "nomination",
        ],
        "vulnerable_archetypes": ["esl_student", "disengaged_but_smart"],
        "remediation_standards": ["CCSS.ELA-LITERACY.RH.11-12.3"],
        "remediation_topic": "Two-stage electoral process in US",
    },
    "misunderstands_winner_take_all": {
        "id": "misunderstands_winner_take_all",
        "category": "electoral_process_error",
        "description": "Incorrectly applies winner-take-all concept or believes it's universal",
        "common_triggers": [
            "electoral votes",
            "popular vote",
            "swing states",
        ],
        "vulnerable_archetypes": ["quiet_thinker", "socially_engaged_activist"],
        "remediation_standards": ["CCSS.ELA-LITERACY.RH.11-12.7"],
        "remediation_topic": "Electoral College mechanics by state",
    },
    "believes_popular_vote_decides": {
        "id": "believes_popular_vote_decides",
        "category": "electoral_process_error",
        "description": "Believes the national popular vote directly determines presidential elections",
        "common_triggers": [
            "popular vote",
            "electoral college",
            "who won",
        ],
        "vulnerable_archetypes": ["socially_engaged_activist", "esl_student", "class_clown"],
        "remediation_standards": ["CCSS.ELA-LITERACY.RH.11-12.7"],
        "remediation_topic": "Electoral College vs popular vote",
    },

    # ── Supreme Court Misconceptions ───────────────────────────────────────────
    "misunderstands_judicial_review": {
        "id": "misunderstands_judicial_review",
        "category": "judicial_misconception",
        "description": "Incorrectly describes the power of judicial review or its origin",
        "common_triggers": [
            "marbury v madison",
            "unconstitutional",
            "supreme court power",
        ],
        "vulnerable_archetypes": ["disengaged_but_smart", "class_clown"],
        "remediation_standards": ["CCSS.ELA-LITERACY.RH.11-12.3"],
        "remediation_topic": "Marbury v. Madison and judicial review",
    },
    "believes_court_can_make_laws": {
        "id": "believes_court_can_make_laws",
        "category": "judicial_misconception",
        "description": "Believes the Supreme Court can create legislation rather than interpret law",
        "common_triggers": [
            "supreme court ruling",
            "court decision",
            "making law",
        ],
        "vulnerable_archetypes": ["esl_student", "quiet_thinker"],
        "remediation_standards": ["CCSS.ELA-LITERACY.RH.11-12.1"],
        "remediation_topic": "Judicial interpretation vs. legislation",
    },

    # ── Political Party Confusion ──────────────────────────────────────────────
    "confuses_party_platforms": {
        "id": "confuses_party_platforms",
        "category": "party_confusion",
        "description": "Attributes positions to the wrong political party",
        "common_triggers": [
            "republican position",
            "democrat position",
            "party platform",
        ],
        "vulnerable_archetypes": ["esl_student", "disengaged_but_smart"],
        "remediation_standards": ["CCSS.ELA-LITERACY.RH.11-12.6"],
        "remediation_topic": "Major party platforms and positions",
    },
    "oversimplifies_political_spectrum": {
        "id": "oversimplifies_political_spectrum",
        "category": "party_confusion",
        "description": "Treats political ideology as strictly binary left-right",
        "common_triggers": [
            "liberal",
            "conservative",
            "moderate",
        ],
        "vulnerable_archetypes": ["socially_engaged_activist", "politically_conservative"],
        "remediation_standards": ["CCSS.ELA-LITERACY.RH.11-12.6"],
        "remediation_topic": "Political ideology dimensions and nuance",
    },

    # ── Civic Engagement Misconceptions ────────────────────────────────────────
    "overestimates_individual_impact": {
        "id": "overestimates_individual_impact",
        "category": "civic_engagement_error",
        "description": "Overestimates the direct impact of individual actions on policy",
        "common_triggers": [
            "my vote matters",
            "writing congressman",
            "petition",
        ],
        "vulnerable_archetypes": ["socially_engaged_activist", "honors_overachiever"],
        "remediation_standards": ["CCSS.ELA-LITERACY.RH.11-12.8"],
        "remediation_topic": "Collective action and civic participation",
    },
    "misunderstands_lobbying": {
        "id": "misunderstands_lobbying",
        "category": "civic_engagement_error",
        "description": "Confuses lobbying with bribery or misunderstands its legal role",
        "common_triggers": [
            "lobbyist",
            "special interest",
            "campaign contribution",
        ],
        "vulnerable_archetypes": ["socially_engaged_activist", "politically_conservative"],
        "remediation_standards": ["CCSS.ELA-LITERACY.RH.11-12.8"],
        "remediation_topic": "Lobbying regulations and First Amendment petitioning",
    },
}


def get_ap_gov_misconception(misconception_id: str) -> Optional[Dict]:
    """Get a specific AP Government misconception by ID."""
    return AP_GOVERNMENT_MISCONCEPTIONS.get(misconception_id)


def get_misconceptions_by_category(category: str) -> Dict[str, Dict]:
    """Get all misconceptions in a specific category."""
    return {
        k: v for k, v in AP_GOVERNMENT_MISCONCEPTIONS.items()
        if v.get("category") == category
    }


def get_misconceptions_for_archetype(archetype: str) -> Dict[str, Dict]:
    """Get misconceptions that commonly affect a specific archetype."""
    return {
        k: v for k, v in AP_GOVERNMENT_MISCONCEPTIONS.items()
        if archetype in v.get("vulnerable_archetypes", [])
    }


# Category descriptions for reporting
CATEGORY_DESCRIPTIONS = {
    "institutional_confusion": "Confusion about government institutions and their roles",
    "constitutional_misunderstanding": "Misunderstandings about Constitutional provisions",
    "electoral_process_error": "Errors in understanding electoral procedures",
    "judicial_misconception": "Misconceptions about the judicial system",
    "party_confusion": "Confusion about political parties and ideology",
    "civic_engagement_error": "Misunderstandings about civic participation",
}
