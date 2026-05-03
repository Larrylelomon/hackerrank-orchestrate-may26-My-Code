# Support Triage Agent
### HackerRank Orchestrate — May 2026

A terminal-based AI support triage agent that classifies and responds to
support tickets across three domains: **HackerRank**, **Claude**, and **Visa**.

---

## Quick Start

```bash
# Install dependencies
pip install -r code/requirements.txt

# Add your Anthropic API key
echo "ANTHROPIC_API_KEY=your-key-here" > .env

# Run against the main ticket file
python code/main.py

# Run against sample tickets (with expected outputs for validation)
python code/main.py --input support_tickets/sample_support_tickets.csv
```

---

## Architecture

The agent uses a five-layer pipeline per ticket:

```
Ticket
  │
  ▼
┌─────────────────────────────────────────────────────┐
│ 1. SAFETY CHECK (safety_check.py)                   │
│    • Prompt injection detection (multilingual)      │
│    • Malicious content detection                    │
│    • Out-of-scope detection                         │
│    Short-circuits immediately if triggered          │
└─────────────────────────┬───────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│ 2. HARD RULE CLASSIFICATION (classifier.py)         │
│    • Danger keyword matching (simple substring)     │
│    • Catches: fraud, identity theft, score          │
│      manipulation, security vulns, subscription     │
│      changes, billing disputes, platform outages,   │
│      non-owner access, infosec forms, rescheduling  │
│    • Returns request_type: invalid for integrity    │
│      violations                                     │
│    Short-circuits to escalated if any rule fires    │
└─────────────────────────┬───────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│ 3. DUAL RETRIEVAL (retriever.py)                    │
│    • BM25 keyword search (rank-bm25)                │
│    • Semantic search (sentence-transformers +       │
│      ChromaDB, all-MiniLM-L6-v2)                   │
│    • Vocabulary bridge for confirmed terminology    │
│      mismatches between user language and corpus    │
│    • Merged 0.7/0.3 semantic/BM25 for confidence   │
│    • Separate results passed to LLM for selection   │
└─────────────────────────┬───────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│ 4. THREE-BAND CONFIDENCE ROUTING (main.py)          │
│                                                     │
│    Score ≥ 0.35  →  HIGH CONFIDENCE                 │
│    Trust retrieval, reply without LLM check         │
│                                                     │
│    Score 0.15–0.35  →  BORDERLINE                   │
│    LLM chunk selection + relevance check            │
│    LLM receives BM25 and semantic results           │
│    separately, selects best chunks, decides         │
│    whether they answer the ticket                   │
│                                                     │
│    Score < 0.15  →  LOW CONFIDENCE                  │
│    Escalate — no corpus support found               │
└─────────────────────────┬───────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│ 5. GENERATION (agent.py)                            │
│    Escalated  →  deterministic template, no LLM     │
│    Replied    →  LLM generates grounded response    │
│                  using only retrieved chunks        │
│                  Also produces request_type and     │
│                  product_area in same call          │
└─────────────────────────┬───────────────────────────┘
                          │
                          ▼
                     output row
```

---

## File Structure

```
code/
├── main.py           # Entry point — CSV I/O, pipeline orchestration,
│                     # three-band confidence routing, terminal UI
├── safety_check.py   # Layer 1 — injection detection (multilingual regex),
│                     # malicious content, out-of-scope classification
├── retriever.py      # Layer 3 — corpus loading, 800-char chunking,
│                     # BM25 + ChromaDB indexing, vocabulary bridge,
│                     # dual retrieval (separate + merged)
├── classifier.py     # Layer 2 — danger keyword list, domain inference,
│                     # request_type override for integrity violations
├── agent.py          # Layers 4+5 — LLM chunk selection + relevance check
│                     # for borderline tickets, LLM response generation,
│                     # escalation template
├── logger.py         # AGENTS.md §5 compliant logging. Append-only.
│                     # Writes to $HOME/hackerrank_orchestrate/log.txt
├── requirements.txt  # Direct dependencies (7 packages)
└── README.md         # This file
```

---

## Installation

**Requirements:** Python 3.12+, Anthropic API key

```bash
# 1. Clone the repository
git clone https://github.com/interviewstreet/hackerrank-orchestrate-may26.git
cd hackerrank-orchestrate-may26

# 2. Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r code/requirements.txt

# 4. Configure API key
# Create .env in repo root (already gitignored):
echo "ANTHROPIC_API_KEY=your-key-here" > .env
```

On first run the `all-MiniLM-L6-v2` embedding model (~90MB) downloads
and caches automatically. Subsequent runs load from cache instantly.

---

## Running

```bash
# Main ticket file (produces support_tickets/output.csv)
python code/main.py

# Sample tickets with known expected outputs
python code/main.py --input support_tickets/sample_support_tickets.csv

# Custom paths
python code/main.py \
  --input support_tickets/support_tickets.csv \
  --output support_tickets/output.csv
```

---

## Output Schema

| Column | Values | Description |
|---|---|---|
| `status` | `replied` / `escalated` | Routing decision |
| `product_area` | underscore_formatted string | Most relevant support category |
| `request_type` | `product_issue` / `feature_request` / `bug` / `invalid` | Request classification |
| `response` | string | User-facing answer grounded in corpus |
| `justification` | string | Concise routing and response explanation |

---

## Design Decisions

### Why danger keywords instead of LLM routing for high-risk cases?

LLMs are persuadable. A fraud report worded sympathetically could
influence an LLM into replying instead of escalating. Danger keywords
cannot be persuaded — if "fraud" appears, the ticket escalates.
Period. Safety-critical routing must be deterministic and auditable.

The keyword list is intentionally narrow — only confirmed high-risk
signals where word presence alone is sufficient to warrant escalation
regardless of context.

### Why hybrid BM25 + semantic retrieval?

BM25 handles exact keyword matches reliably. Semantic search handles
intent matches when vocabulary differs. Neither alone is sufficient:

- BM25 without semantic: "mock interviews stopped" does not find
  billing/refund articles because the keywords do not match
- Semantic without BM25: exact product names and technical terms
  may not embed close enough to surface the right article

The 0.7/0.3 weighting favours semantic because intent matters more
than keyword overlap for support queries.

### Why send BM25 and semantic results separately to the LLM?

When retrieval is borderline, the two methods sometimes disagree.
BM25 might find "interviewer" in interview management articles while
semantic finds "remove person" in team management articles.

Sending both sets separately lets the LLM apply actual language
understanding to pick the right chunks — which is exactly what it is
better at than mathematical score merging.

### Why the vocabulary bridge?

The HackerRank corpus uses internal terminology that does not match
natural user language. Users say "remove employee" — the corpus says
"lock user access." Users say "apply tab" — the corpus says
"job search and applications."

This is not query expansion. It is terminology translation for
confirmed vocabulary mismatches. The bridge only activates for
specific patterns where the mismatch is documented and verified.

### Why three confidence bands instead of always running the LLM?

The LLM relevance check is non-deterministic — the same borderline
ticket may classify differently across runs. High-confidence tickets
(score ≥ 0.35) have clear corpus support and do not need the LLM to
verify — routing them directly removes unnecessary variance and cost.

Only genuinely uncertain cases (0.15–0.35) go through the relevance
check where the LLM's judgment actually adds value.

### Why does the corpus decide routing rather than the LLM?

Retrieval confidence is a deterministic, auditable signal. A score
of 0.6 means the corpus clearly covers this topic. An LLM deciding
whether to answer introduces persuadability — a clever ticket can
influence the decision through tone or framing.

Grounding the routing decision in retrieval score means: if the
corpus has a relevant answer, we reply. If it does not, we escalate.
The LLM only generates the response text — it never decides routing
on its own.

---

## Escalation Decision Logic

```
Injection / malicious content detected?          → replied (invalid)
Out of scope with no domain context?             → replied (invalid)
Danger keyword matched?                          → escalated
  • fraud / unauthorized transaction
  • identity theft / identity stolen
  • score / result manipulation                  → invalid request_type
  • security vulnerability / bug bounty
  • subscription cancel / pause
  • billing with specific order ID
  • non-owner requesting access restoration
  • platform-wide outage
  • infosec / compliance form filling
  • assessment rescheduling
  • explicit refund demand
  • data breach
Ticket too vague (≤ 5 words)?                   → escalated
Retrieval confidence < 0.15?                     → escalated
Retrieval confidence 0.15–0.35:
  LLM says docs answer the question?             → replied
  LLM says docs do not answer?                   → escalated
Retrieval confidence ≥ 0.35?                     → replied
```

---

## Known Limitations

**Visa corpus is thin** — only 14 documents covering a narrow topic
range. Visa tickets are more likely to hit the low-confidence
escalation threshold than HackerRank or Claude tickets.

**Vocabulary mismatches** — the HackerRank corpus uses internal
terminology ("lock user access") that does not match natural user
language ("remove employee"). The vocabulary bridge addresses known
mismatches but new phrasings may not be covered.

**LLM non-determinism** — the borderline relevance check may
classify the same ticket differently across runs. This affects only
tickets with retrieval scores between 0.15 and 0.35.

**Retrieval weights untuned** — the 0.7/0.3 semantic/BM25 merge
ratio is a principled default, not empirically optimised against
labelled data for this specific corpus.

**Chunk size fixed** — 800-character chunks with 100-character
overlap may split articles at inconvenient points, causing relevant
steps to appear across chunk boundaries.

---

## What Would Improve This With More Time

**Corpus enrichment** — adding synonym metadata to articles so
retrieval is self-correcting without vocabulary bridge patches.

**Empirical weight tuning** — calibrating the 0.7/0.3 retrieval
weights against labelled ticket-document pairs.

**Cosine similarity in ChromaDB** — the current L2 distance metric
is less accurate than cosine similarity for sentence-transformer
embeddings. Switching would improve semantic ranking.

**Persistent ChromaDB index** — currently rebuilt on every run.
Persisting to disk would make subsequent runs start in under 10 seconds.

**Parallel ticket processing** — tickets are currently processed
sequentially. Parallelising API calls would reduce total run time
significantly for large ticket files.

**Richer escalation templates** — current template is generic.
Domain-specific templates with actual contact channels would be
more useful to users.

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `anthropic` | 0.97.0 | Claude API client |
| `chromadb` | 1.5.8 | Vector database for semantic search |
| `sentence-transformers` | 5.4.1 | Local embedding model (all-MiniLM-L6-v2) |
| `rank-bm25` | 0.2.2 | BM25 keyword search |
| `rich` | 15.0.0 | Terminal progress bar and summary table |
| `python-dotenv` | 1.2.2 | Loads .env for API key |
| `tzdata` | 2026.2 | Timezone data (required on Windows) |

---

## Chat Transcript

Per AGENTS.md §2, all agent run entries are logged to:

```
# macOS / Linux
$HOME/hackerrank_orchestrate/log.txt

# Windows
%USERPROFILE%\hackerrank_orchestrate\log.txt
```
