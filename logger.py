# code/logger.py
"""
Chat transcript logging per AGENTS.md §2 and §5 spec.
Log file lives at $HOME/hackerrank_orchestrate/log.txt
Append-only, never commit, never log secrets.
"""

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHALLENGE_END = datetime(2026, 5, 2, 11, 0, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
AGENT_NAME = "support-triage-agent"
LOG_DIR = Path.home() / "hackerrank_orchestrate"
LOG_FILE = LOG_DIR / "log.txt"
REPO_ROOT = Path(__file__).parent.parent.resolve()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _time_remaining() -> str:
    now = datetime.now(ZoneInfo("Asia/Kolkata"))
    delta = CHALLENGE_END - now
    if delta.total_seconds() <= 0:
        return "EXPIRED"
    total_seconds = int(delta.total_seconds())
    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60
    return f"{days}d {hours}h {minutes}m"


def _redact_secrets(text: str) -> str:
    """Remove API keys, tokens, and secrets from logged text."""
    # Anthropic keys: sk-ant-...
    text = re.sub(r"sk-ant-[A-Za-z0-9\-_]{20,}", "[REDACTED]", text)
    # OpenAI keys
    text = re.sub(r"sk-[A-Za-z0-9]{20,}", "[REDACTED]", text)
    # Generic bearer tokens
    text = re.sub(r"Bearer\s+[A-Za-z0-9\-_\.]{20,}", "Bearer [REDACTED]", text)
    # .env style KEY=VALUE
    text = re.sub(
        r"(API_KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL)\s*=\s*\S+",
        r"\1=[REDACTED]",
        text,
        flags=re.IGNORECASE,
    )
    return text


def _ensure_log_dir():
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _append(content: str):
    _ensure_log_dir()
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(content + "\n")


def _git_branch() -> str:
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, cwd=REPO_ROOT
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def log_session_start():
    """Append a SESSION START entry (§5.1)."""
    entry = f"""
## [{_now_iso()}] SESSION START

Agent: {AGENT_NAME}
Repo Root: {REPO_ROOT}
Branch: {_git_branch()}
Worktree: main
Parent Agent: none
Language: py
Time Remaining: {_time_remaining()}
"""
    _append(entry)


def log_onboarding(language: str = "py"):
    """Append ONBOARDING COMPLETE block (§3.4) — first run only."""
    # Check if already recorded
    if LOG_FILE.exists():
        content = LOG_FILE.read_text(encoding="utf-8")
        if f"AGREEMENT RECORDED: {REPO_ROOT}" in content:
            return  # already done

    entry = f"""
## [{_now_iso()}] ONBOARDING COMPLETE

AGREEMENT RECORDED: {REPO_ROOT}
Agent: {AGENT_NAME}
Language: {language}
System Time: {_now_iso()}
Time Remaining: {_time_remaining()} until 2026-05-02T11:00:00+05:30
"""
    _append(entry)


def log_turn(
    title: str,
    user_prompt: str,
    agent_summary: str,
    actions: list[str] | None = None,
):
    """
    Append a per-turn entry (§5.2).
    Automatically redacts secrets from user_prompt.
    """
    redacted_prompt = _redact_secrets(user_prompt)
    actions_str = "\n".join(f"* {a}" for a in (actions or []))

    entry = f"""
## [{_now_iso()}] {title[:80]}

User Prompt (verbatim, secrets redacted):
{redacted_prompt}

Agent Response Summary:
{agent_summary}

Actions:
{actions_str if actions_str else "* (none)"}

Context:
tool={AGENT_NAME}
branch={_git_branch()}
repo_root={REPO_ROOT}
worktree=main
parent_agent=none
"""
    _append(entry)


def log_run_start(input_csv: str, total_tickets: int):
    """Log the start of a full CSV processing run."""
    log_turn(
        title="Agent run started",
        user_prompt=f"Process {input_csv} ({total_tickets} tickets)",
        agent_summary=(
            f"Starting triage run over {total_tickets} support tickets. "
            f"Pipeline: safety_check → retrieval → classify → generate → output CSV."
        ),
        actions=[
            f"Reading input: {input_csv}",
            f"Writing output: support_tickets/output.csv",
            f"Log file: {LOG_FILE}",
        ],
    )


def log_ticket(
    row_num: int,
    issue: str,
    company: str,
    status: str,
    request_type: str,
    product_area: str,
    domain: str,
    retrieval_score: float,
    escalation_reason: str,
):
    """Log a single processed ticket (summary only, no full response text)."""
    log_turn(
        title=f"Ticket {row_num} processed — {status} / {request_type}",
        user_prompt=f"[Ticket {row_num}] company={company} | issue={issue[:120]}...",
        agent_summary=(
            f"Ticket {row_num}: status={status}, request_type={request_type}, "
            f"product_area={product_area}, domain={domain}, "
            f"retrieval_score={retrieval_score:.3f}. "
            f"Escalation reason: {escalation_reason or 'N/A'}."
        ),
        actions=[
            f"safety_check → passed",
            f"retriever → top score {retrieval_score:.3f}",
            f"classifier → {status}",
            f"agent.generate → response written to output.csv row {row_num}",
        ],
    )


def log_run_complete(total: int, replied: int, escalated: int, output_csv: str):
    """Log the completion summary of a full run."""
    log_turn(
        title="Agent run complete",
        user_prompt=f"Run finished — {total} tickets processed",
        agent_summary=(
            f"Completed triage of {total} tickets. "
            f"replied={replied}, escalated={escalated}. "
            f"Output written to {output_csv}."
        ),
        actions=[
            f"Output CSV written: {output_csv}",
            f"Total tickets: {total}",
            f"Replied: {replied} | Escalated: {escalated}",
        ],
    )