# code/main.py
"""
Entry point for the HackerRank Orchestrate support triage agent.

Usage:
    python code/main.py
    python code/main.py --input support_tickets/sample_support_tickets.csv
    python code/main.py --input support_tickets/support_tickets.csv
"""

import argparse
import csv
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, TextColumn,
    BarColumn, TaskProgressColumn,
)
from rich.table import Table

load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent))

from safety_check import (
    check_safety,
    injection_response,
    malicious_response,
    out_of_scope_response,
)
from retriever import Retriever
from classifier import classify, infer_domain
from agent import (
    generate,
    select_and_check_relevance,
    HIGH_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
)
from logger import (
    log_session_start,
    log_onboarding,
    log_run_start,
    log_ticket,
    log_run_complete,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_ROOT  = Path(__file__).parent.parent
INPUT_CSV  = REPO_ROOT / "support_tickets" / "support_tickets.csv"
OUTPUT_CSV = REPO_ROOT / "support_tickets" / "output.csv"
OUTPUT_FIELDS = [
    "issue", "subject", "company",
    "response", "product_area", "status",
    "request_type", "justification",
]

console = Console()


# ---------------------------------------------------------------------------
# Product area normalisation
# ---------------------------------------------------------------------------

def _normalise_product_area(area: str) -> str:
    """Standardise product area to underscore format."""
    if not area:
        return "general"
    return area.strip().lower().replace("-", "_").replace(" ", "_")


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def read_tickets(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [
            {
                "issue":   row.get("Issue",   row.get("issue",   "")).strip(),
                "subject": row.get("Subject", row.get("subject", "")).strip(),
                "company": row.get("Company", row.get("company", "None")).strip(),
            }
            for row in reader
        ]


def write_output(path: Path, rows: list[dict]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Single ticket processing
# ---------------------------------------------------------------------------

def process_ticket(
    ticket: dict,
    retriever: Retriever,
    row_num: int,
) -> dict:
    issue   = ticket["issue"]
    subject = ticket["subject"]
    company = ticket["company"]

    # ------------------------------------------------------------------
    # 1. Safety check
    # ------------------------------------------------------------------
    safety = check_safety(issue, subject, company)

    if safety["injection"]:
        result = injection_response()
        _log_ticket(row_num, ticket, result, 0.0, "injection_detected")
        return _build_row(ticket, result)

    if safety["malicious"]:
        result = malicious_response()
        _log_ticket(row_num, ticket, result, 0.0, "malicious_content")
        return _build_row(ticket, result)

    if safety["out_of_scope"] and not safety["escalation_triggers"]:
        result = out_of_scope_response()
        _log_ticket(row_num, ticket, result, 0.0, "out_of_scope")
        return _build_row(ticket, result)

    # ------------------------------------------------------------------
    # 2. Domain routing
    # ------------------------------------------------------------------
    normalised = company.strip().lower()
    domain = (
        infer_domain(issue, subject)
        if normalised in ("none", "", "unknown")
        else normalised
    )

    # ------------------------------------------------------------------
    # 3. Dual retrieval — BM25 and semantic separately
    # ------------------------------------------------------------------
    results         = retriever.retrieve_separate(issue, domain=domain, top_k=5)
    bm25_chunks     = results["bm25"]
    semantic_chunks = results["semantic"]

    # Confidence score from merged retrieval
    merged_chunks   = retriever.retrieve(issue, domain=domain, top_k=5)
    retrieval_score = merged_chunks[0]["score"] if merged_chunks else 0.0

    # ------------------------------------------------------------------
    # 4. Hard rule classification
    # ------------------------------------------------------------------
    hard = classify(
        issue=issue,
        subject=subject,
        company=company,
        safety_result=safety,
        retrieved_chunks=semantic_chunks,
        retrieval_score=retrieval_score,
    )
    domain            = hard["domain"] or domain
    hard_request_type = hard.get("request_type", "product_issue")

    # ------------------------------------------------------------------
    # 5. Three-band confidence routing
    # ------------------------------------------------------------------
    if hard["should_escalate"]:
        # Hard rule fired — escalate immediately
        should_escalate   = True
        escalation_reason = hard["escalation_reason"]
        routing_note      = "hard_rule"
        chunks            = semantic_chunks[:3]

    elif retrieval_score >= HIGH_CONFIDENCE_THRESHOLD:
        # Strong retrieval signal — trust it, skip relevance check
        should_escalate   = False
        escalation_reason = ""
        routing_note      = f"high_confidence({retrieval_score:.2f})"
        chunks            = semantic_chunks[:3]

    elif retrieval_score >= LOW_CONFIDENCE_THRESHOLD:
        # Borderline — ask LLM chunk selection + relevance check
        selection = select_and_check_relevance(
            issue, subject, domain,
            bm25_chunks, semantic_chunks,
        )
        chunks = selection["selected_chunks"]
        if selection["relevant"]:
            should_escalate   = False
            escalation_reason = ""
            routing_note      = f"relevance_yes({retrieval_score:.2f})"
        else:
            should_escalate   = True
            escalation_reason = (
                f"Documentation does not answer this request: "
                f"{selection['reason']}"
            )
            routing_note      = f"relevance_no({retrieval_score:.2f})"

    else:
        # Below threshold — no relevant docs found
        should_escalate   = True
        escalation_reason = (
            f"No relevant documentation found in "
            f"{domain or 'any'} corpus "
            f"(confidence {retrieval_score:.2f}). "
            f"Escalating to avoid unsupported response."
        )
        routing_note      = f"low_confidence({retrieval_score:.2f})"
        chunks            = []

    # Platform reliability override for outages caught by relevance check
    if should_escalate:
        issue_lower = issue.lower()
        if any(w in issue_lower for w in [
            "down", "not working", "failing",
            "stopped working", "broken", "resume builder",
        ]):
            if not hard.get("product_area") or hard.get("product_area") == "general":
                hard["product_area"] = "platform_reliability"

    # ------------------------------------------------------------------
    # 6. Generation
    # ------------------------------------------------------------------
    gen = generate(
        issue=issue,
        subject=subject,
        domain=domain,
        chunks=chunks,
        should_escalate=should_escalate,
        escalation_reason=escalation_reason,
        request_type=hard_request_type,
    )

    # ------------------------------------------------------------------
    # 7. Assemble output row
    # ------------------------------------------------------------------
    raw_product_area = (
        gen.get("product_area")
        or hard.get("product_area")
        or domain
        or "general"
    )

    result = {
        "status":        "escalated" if should_escalate else "replied",
        "product_area":  _normalise_product_area(raw_product_area),
        "request_type":  gen.get("request_type", hard_request_type),
        "response":      gen["response"],
        "justification": gen["justification"],
    }

    _log_ticket(
        row_num, ticket, result,
        retrieval_score, routing_note,
    )
    return _build_row(ticket, result)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_row(ticket: dict, result: dict) -> dict:
    return {
        "issue":         ticket["issue"],
        "subject":       ticket["subject"],
        "company":       ticket["company"],
        "response":      result["response"],
        "product_area":  _normalise_product_area(result.get("product_area", "general")),
        "status":        result["status"],
        "request_type":  result["request_type"],
        "justification": result["justification"],
    }


def _log_ticket(
    row_num: int,
    ticket: dict,
    result: dict,
    score: float,
    routing_note: str,
):
    log_ticket(
        row_num=row_num,
        issue=ticket["issue"],
        company=ticket["company"],
        status=result["status"],
        request_type=result["request_type"],
        product_area=result["product_area"],
        domain=ticket["company"].lower(),
        retrieval_score=score,
        escalation_reason=routing_note,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Support triage agent")
    parser.add_argument(
        "--input", type=Path, default=INPUT_CSV,
        help="Input CSV (default: support_tickets/support_tickets.csv)",
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_CSV,
        help="Output CSV (default: support_tickets/output.csv)",
    )
    args = parser.parse_args()

    log_session_start()
    log_onboarding()

    console.rule("[bold blue]HackerRank Orchestrate — Support Triage Agent")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        console.print(
            "[yellow]⚠  ANTHROPIC_API_KEY not set — running in stub mode.[/yellow]\n"
        )
    else:
        console.print("[green]✓  ANTHROPIC_API_KEY loaded.[/green]\n")

    tickets = read_tickets(args.input)
    console.print(
        f"[cyan]Loaded {len(tickets)} tickets from {args.input}[/cyan]"
    )

    console.rule("Building retrieval index")
    retriever = Retriever()

    log_run_start(str(args.input), len(tickets))

    console.rule("Processing tickets")
    output_rows     = []
    replied_count   = 0
    escalated_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Triaging...", total=len(tickets))

        for i, ticket in enumerate(tickets, 1):
            progress.update(
                task,
                description=(
                    f"[{i}/{len(tickets)}] "
                    f"{ticket['company']:12} | "
                    f"{ticket['issue'][:50]}..."
                ),
            )
            row = process_ticket(ticket, retriever, i)
            output_rows.append(row)

            if row["status"] == "replied":
                replied_count += 1
            else:
                escalated_count += 1

            progress.advance(task)

    write_output(args.output, output_rows)
    log_run_complete(
        len(tickets), replied_count,
        escalated_count, str(args.output),
    )

    console.rule("Results")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#",             style="dim", width=4)
    table.add_column("Company",       width=12)
    table.add_column("Status",        width=10)
    table.add_column("Type",          width=16)
    table.add_column("Product Area",  width=28)
    table.add_column("Issue (truncated)", width=45)

    for i, row in enumerate(output_rows, 1):
        style = "green" if row["status"] == "replied" else "red"
        table.add_row(
            str(i),
            row["company"],
            f"[{style}]{row['status']}[/{style}]",
            row["request_type"],
            row["product_area"],
            row["issue"][:45],
        )

    console.print(table)
    console.print(
        f"\n[bold green]✓ Done.[/bold green] "
        f"replied={replied_count}, escalated={escalated_count}"
    )
    console.print(f"[cyan]Output → {args.output}[/cyan]")


if __name__ == "__main__":
    main()