# code/safety.py
"""
Safety and prompt injection detection.
Runs BEFORE retrieval and LLM — purely rule-based, no API needed.
"""

import re

# ---------------------------------------------------------------------------
# Injection / system-probing patterns
# ---------------------------------------------------------------------------
INJECTION_PATTERNS = [
    # Asking agent to reveal internals
    r"(show|display|print|reveal|expose|dump|list).{0,40}(rule|instruction|prompt|system|internal|logic|retrieved|corpus|document)",
    r"(ignore|forget|override|bypass|disregard).{0,40}(instruction|rule|guideline|policy|above|previous|prior)",
    r"(you are now|act as|pretend|roleplay|simulate|imagine you are).{0,40}(different|another|new|unrestricted|without)",
    r"(what|show).{0,20}(exact|internal|hidden|real).{0,20}(logic|rule|prompt|decision|process)",
    r"affiche.{0,40}(règle|document|logique|interne)",   # French injection (ticket 25)
    r"muestra.{0,40}(regla|documento|lógica|interna)",   # Spanish injection
    r"(jailbreak|dan mode|developer mode|god mode|unrestricted mode)",
    r"repeat.{0,20}(everything|all|above|back|verbatim)",
    r"(previous|above|earlier).{0,20}(instruction|prompt|message|text|context)",
]

# ---------------------------------------------------------------------------
# Malicious / out-of-scope content patterns
# ---------------------------------------------------------------------------
MALICIOUS_PATTERNS = [
    r"(delete|remove|wipe|erase|format).{0,30}(all file|system file|director|disk|drive|root|everything)",
    r"(sudo|rm -rf|format c|del \/f|shutdown|passwd|chmod 777)",
    r"(hack|exploit|vulnerability|sql injection|xss|csrf).{0,30}(how|teach|show|give|provide)",
    r"(bomb|weapon|explosive|malware|ransomware|trojan)",
]

# ---------------------------------------------------------------------------
# Escalation trigger keywords (high-risk topics)
# These are checked AFTER safety — used by classifier too
# ---------------------------------------------------------------------------
ESCALATION_KEYWORDS = {
    "fraud":            r"\b(fraud|fraudulent|scam|phishing|unauthorized.{0,10}(charge|transaction|access))\b",
    "identity_theft":   r"\b(identity.{0,5}theft|identity.{0,5}stolen|someone.{0,10}stole.{0,10}(my|our))\b",
    "account_hacked":   r"\b(hacked|compromised|breach|stolen.{0,10}account|account.{0,10}stolen)\b",
    "financial_dispute":r"\b(refund|chargeback|dispute.{0,10}(charge|payment|transaction)|wrong.{0,10}charge)\b",
    "legal_threat":     r"\b(lawsuit|legal action|attorney|lawyer|sue|court|gdpr.{0,10}complaint)\b",
    "security_vuln":    r"\b(security.{0,10}(bug|vuln|vulnerability|flaw|hole)|bug.{0,10}bounty|cve|zero.?day)\b",
    "data_breach":      r"\b(data.{0,10}breach|data.{0,10}leak|exposed.{0,10}data|personal.{0,10}data.{0,10}(stolen|exposed))\b",
}

# ---------------------------------------------------------------------------
# Out-of-scope detection (not related to any of the three domains)
# ---------------------------------------------------------------------------
SUPPORTED_DOMAINS_HINT = [
    "hackerrank", "claude", "visa", "anthropic", "assessment", "test",
    "candidate", "interview", "subscription", "billing", "payment",
    "card", "account", "login", "access", "feature", "bug", "error",
    "support", "help", "issue", "problem", "refund", "certificate",
]


# ---------------------------------------------------------------------------
# Main safety check function
# ---------------------------------------------------------------------------

def check_safety(issue: str, subject: str = "", company: str = "None") -> dict:
    """
    Run all safety checks on a ticket.

    Returns:
        {
            "safe": bool,
            "injection": bool,
            "malicious": bool,
            "escalation_triggers": list[str],   # e.g. ["fraud", "identity_theft"]
            "out_of_scope": bool,
            "reason": str,                       # human-readable explanation
        }
    """
    combined_text = f"{subject} {issue}".lower()

    result = {
        "safe": True,
        "injection": False,
        "malicious": False,
        "escalation_triggers": [],
        "out_of_scope": False,
        "reason": "",
    }

    # 1. Prompt injection check
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, combined_text, re.IGNORECASE | re.DOTALL):
            result["safe"] = False
            result["injection"] = True
            result["reason"] = "Prompt injection or system-probing attempt detected."
            return result   # short-circuit — don't process further

    # 2. Malicious content check
    for pattern in MALICIOUS_PATTERNS:
        if re.search(pattern, combined_text, re.IGNORECASE):
            result["safe"] = False
            result["malicious"] = True
            result["reason"] = "Malicious or harmful content detected."
            return result

    # 3. Escalation trigger keywords (safe to continue, but flag for classifier)
    for trigger_name, pattern in ESCALATION_KEYWORDS.items():
        if re.search(pattern, combined_text, re.IGNORECASE):
            result["escalation_triggers"].append(trigger_name)

    # 4. Out-of-scope check (only when company is None or unrecognised)
    normalised_company = company.strip().lower()
    if normalised_company in ("none", "", "unknown"):
        has_domain_hint = any(
            hint in combined_text for hint in SUPPORTED_DOMAINS_HINT
        )
        if not has_domain_hint:
            result["out_of_scope"] = True
            result["reason"] = "No recognisable domain context found."

    return result


# ---------------------------------------------------------------------------
# Response helpers for unsafe tickets
# ---------------------------------------------------------------------------

def injection_response() -> dict:
    """Standard output row for a detected injection attempt."""
    return {
        "status": "replied",
        "product_area": "security",
        "request_type": "invalid",
        "response": (
            "I'm sorry, but I'm unable to process this request. "
            "It appears to contain content that attempts to alter or expose "
            "the internal workings of this support system. "
            "Please submit a genuine support query and we'll be happy to help."
        ),
        "justification": (
            "Prompt injection or system-probing attempt detected. "
            "Request blocked before retrieval or LLM processing."
        ),
    }


def malicious_response() -> dict:
    """Standard output row for malicious/harmful content."""
    return {
        "status": "replied",
        "product_area": "security",
        "request_type": "invalid",
        "response": (
            "I'm sorry, but this request is outside the scope of our support service "
            "and cannot be processed."
        ),
        "justification": (
            "Request contains malicious or harmful content unrelated to "
            "HackerRank, Claude, or Visa support."
        ),
    }


def out_of_scope_response() -> dict:
    """Standard output row for clearly out-of-scope tickets."""
    return {
        "status": "replied",
        "product_area": "general",
        "request_type": "invalid",
        "response": (
            "I'm sorry, but your request doesn't appear to relate to "
            "HackerRank, Claude, or Visa support. "
            "This service can only assist with issues in those three domains."
        ),
        "justification": (
            "No recognisable domain context found. "
            "Ticket is out of scope for this support agent."
        ),
    }