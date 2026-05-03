# code/classifier.py
"""
Deterministic escalation via danger keyword detection only.
All semantic routing is delegated to the retrieval confidence
score and LLM relevance check in main.py and agent.py.

Two jobs:
  1. Infer domain when company == "None"
  2. Hard-escalate on danger keywords — safety critical cases only
"""

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Domain inference keywords
# ---------------------------------------------------------------------------
DOMAIN_KEYWORDS = {
    "claude": [
        "claude", "anthropic", "claude.ai", "pro plan", "max plan",
        "team plan", "enterprise plan", "claude code", "claude desktop",
        "artifact", "project", "conversation", "prompt", "context window",
        "bedrock", "aws", "lti", "connector", "mcp",
    ],
    "hackerrank": [
        "hackerrank", "hacker rank", "assessment", "test", "candidate",
        "recruiter", "interview", "coding challenge", "submission",
        "plagiarism", "proctoring", "scorecard", "resume builder",
        "certificate", "mock interview", "skillup", "chakra",
    ],
    "visa": [
        "visa", "card", "payment", "merchant", "transaction", "refund",
        "chargeback", "atm", "traveller", "travelers cheque",
        "dispute", "fraud", "stolen card", "lost card", "cash advance",
        "checkout", "contactless", "chip", "pin",
    ],
}

# ---------------------------------------------------------------------------
# Danger keyword list
# Simple substring match — no regex gymnastics.
# If ANY of these appear → always escalate immediately.
# These represent genuine high-risk signals where word presence
# alone is sufficient to warrant human intervention.
# ---------------------------------------------------------------------------
DANGER_KEYWORDS = [
    # Financial fraud
    "fraud",
    "fraudulent",
    "unauthorized charge",
    "unauthorized transaction",
    "unauthorized access",
    "chargeback",
    "scam",
    "phishing",

    # Identity
    "identity theft",
    "identity stolen",
    "identity compromised",
    "someone stole my",
    "someone stole our",

    # Assessment integrity — request_type: invalid
    "increase my score",
    "increase the score",
    "change my score",
    "change my result",
    "change the result",
    "tell the company to",
    "tell the recruiter to",
    "move me to the next",

    # Account access
    "not the workspace owner",
    "not the workspace admin",
    "not the owner",
    "not the admin",
    "not an admin",
    "not an owner",
    "i am not the owner",
    "i am not the admin",
    "even though i am not",

    # Security
    "security vulnerability",
    "security bug",
    "security flaw",
    "bug bounty",
    "zero day",
    "zero-day",
    "cve-",

    # Billing actions
    "cancel subscription",
    "cancel our subscription",
    "pause subscription",
    "pause our subscription",
    "suspend subscription",
    "order id",
    "payment id",
    "transaction id",

    # Rescheduling
    "rescheduling",
    "reschedule",
    "postpone",
    "alternative date",
    "alternative time",

    # Compliance
    "infosec",
    "fill in the form",
    "fill out the form",
    "filling in the form",
    "compliance form",
    "security questionnaire",
    "security form",

    # Data
    "data breach",
    "data leak",
    "personal data stolen",
    "personal data exposed",

    # Explicit financial demands
    "please refund me",
    "refund me immediately",
    "refund us immediately",
    "refund me today",
    "refund me asap",
    "refund me now",
    "ban the seller",
    "ban the merchant",

    # Platform-wide outages
    "none of the submissions",
    "submissions across any",
    "all requests to claude",
    "all requests are failing",
    "all requests failing",
    "stopped working completely",

    # extra
    "give me the refund",
    "give me a refund",
    "want a refund",
    "need a refund",
]

# ---------------------------------------------------------------------------
# Integrity violation keywords — these get request_type: invalid
# ---------------------------------------------------------------------------
INTEGRITY_KEYWORDS = [
    "increase my score",
    "increase the score",
    "change my score",
    "change my result",
    "change the result",
    "tell the company to",
    "tell the recruiter to",
    "move me to the next",
]

# Minimum word count — tickets shorter than this are too vague
VAGUE_WORD_THRESHOLD = 5


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def infer_domain(issue: str, subject: str) -> Optional[str]:
    """
    Infer domain from ticket text when company == 'None'.
    Returns 'claude', 'hackerrank', 'visa', or None.
    """
    combined = f"{subject} {issue}".lower()
    scores   = {domain: 0 for domain in DOMAIN_KEYWORDS}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in combined:
                scores[domain] += 1
    best = max(scores, key=lambda d: scores[d])
    return best if scores[best] > 0 else None


def _check_danger_keywords(text: str) -> Optional[str]:
    """
    Scan text for danger keywords.
    Returns the matched keyword if found, None otherwise.
    Simple substring match — no regex needed.
    """
    text_lower = text.lower()
    for keyword in DANGER_KEYWORDS:
        if keyword in text_lower:
            return keyword
    return None


def _is_integrity_violation(keyword: str) -> bool:
    """Check if matched keyword represents an integrity violation."""
    return any(k in keyword for k in INTEGRITY_KEYWORDS)


def _keyword_to_product_area(keyword: str) -> str:
    """Map a matched danger keyword to an appropriate product area."""
    if any(k in keyword for k in [
        "fraud", "scam", "phishing", "identity",
        "stolen", "breach", "leak", "unauthorized",
    ]):
        return "fraud_and_security"
    if any(k in keyword for k in [
        "score", "result", "recruiter", "company",
        "next round", "move me",
    ]):
        return "assessment_integrity"
    if any(k in keyword for k in [
        "vulnerability", "bug bounty", "zero day", "cve",
    ]):
        return "safety_and_trust"
    if any(k in keyword for k in [
        "subscription", "cancel", "pause",
        "order id", "payment id", "refund",
    ]):
        return "billing_and_payments"
    if any(k in keyword for k in [
        "reschedul", "postpone", "alternative date",
    ]):
        return "assessments"
    if any(k in keyword for k in [
        "infosec", "compliance", "questionnaire", "security form",
    ]):
        return "security_and_compliance"
    if any(k in keyword for k in [
        "owner", "admin", "workspace",
    ]):
        return "account_access"
    if any(k in keyword for k in [
        "submission", "requests failing", "requests to claude",
        "stopped working completely",
    ]):
        return "platform_reliability"
    return "general"


def classify(
    issue: str,
    subject: str,
    company: str,
    safety_result: dict,
    retrieved_chunks: list[dict],
    retrieval_score: float,
) -> dict:
    """
    Hard classification pass.

    Escalates only on:
      1. Safety-flagged high-risk triggers
      2. Danger keyword match
      3. Ticket too vague
      4. Low retrieval confidence

    Returns:
      domain, should_escalate, escalation_reason,
      product_area, request_type
    """
    # Resolve domain
    normalised = company.strip().lower()
    domain = (
        infer_domain(issue, subject)
        if normalised in ("none", "", "unknown")
        else normalised
    )

    combined = f"{subject} {issue}"

    # 1. Safety-flagged high risk triggers
    high_risk = {
        "fraud", "identity_theft", "account_hacked",
        "security_vuln", "data_breach", "legal_threat",
    }
    triggered = set(safety_result.get("escalation_triggers", [])) & high_risk
    if triggered:
        return {
            "domain":           domain,
            "should_escalate":  True,
            "escalation_reason": f"High-risk trigger: {', '.join(triggered)}.",
            "product_area":     "fraud_and_security",
            "request_type":     "product_issue",
        }

    # 2. Danger keyword match
    matched_keyword = _check_danger_keywords(combined)
    if matched_keyword:
        request_type = (
            "invalid"
            if _is_integrity_violation(matched_keyword)
            else "product_issue"
        )
        return {
            "domain":           domain,
            "should_escalate":  True,
            "escalation_reason": (
                f"Danger keyword detected: '{matched_keyword}'. "
                f"Escalating for human review."
            ),
            "product_area":     _keyword_to_product_area(matched_keyword),
            "request_type":     request_type,
        }

    # 3. Vague ticket
    word_count = len(issue.strip().split())
    if word_count <= VAGUE_WORD_THRESHOLD:
        return {
            "domain":           domain,
            "should_escalate":  True,
            "escalation_reason": "Ticket too vague — insufficient information.",
            "product_area":     "general",
            "request_type":     "product_issue",
        }

    # 4. Low retrieval confidence
    LOW_CONFIDENCE_THRESHOLD = 0.15
    if retrieval_score < LOW_CONFIDENCE_THRESHOLD and domain:
        return {
            "domain":           domain,
            "should_escalate":  True,
            "escalation_reason": (
                f"No relevant documentation found in {domain} corpus "
                f"(confidence {retrieval_score:.2f})."
            ),
            "product_area":     "general",
            "request_type":     "product_issue",
        }

    # 5. Pass to confidence routing in main.py
    return {
        "domain":           domain,
        "should_escalate":  False,
        "escalation_reason": "",
        "product_area":     "",
        "request_type":     "product_issue",
    }