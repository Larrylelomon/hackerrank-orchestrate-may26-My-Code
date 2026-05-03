# code/retriever.py
import re
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import chromadb

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_ROOT     = Path(__file__).parent.parent / "data"
DOMAINS       = ["claude", "hackerrank", "visa"]
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100
TOP_K         = 5
EMBED_MODEL   = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Vocabulary bridge
# Maps natural user language to HackerRank corpus terminology.
# Only used where there is a confirmed vocabulary mismatch between
# how users describe their problem and how the corpus is written.
# This is NOT query expansion — it is terminology translation.
# ---------------------------------------------------------------------------
VOCAB_BRIDGE = {
    # Users say "remove/fire/delete person" — corpus says "lock user access"
    r"remove.{0,20}(interviewer|employee|user|member|person|them|someone)":
        "lock user access manage team members",
    r"(fire|fired|left|leaving).{0,20}(employee|user|member|staff|colleague)":
        "lock user access manage team members",
    r"delete.{0,20}(user|account|member|employee|interviewer)":
        "lock user access manage team members",

    # Users say "apply tab" — corpus says "job search and applications"
    r"apply.{0,10}tab":
        "job search applications quick apply hackerrank community",

    # Users say "use my data to improve models" — corpus says "privacy training data"
    r"(use|using|allow).{0,20}(my data|data).{0,20}(improv|train|model)":
        "privacy training data retention anthropic policy",
    r"how long.{0,20}(data|information)":
        "privacy data retention policy anthropic",
}


def _apply_vocab_bridge(query: str) -> str:
    """
    Translate natural user language to corpus terminology.
    Only applies where there is a confirmed vocabulary mismatch.
    Returns augmented query string.
    """
    query_lower = query.lower()
    additions   = []
    for pattern, translation in VOCAB_BRIDGE.items():
        if re.search(pattern, query_lower):
            additions.append(translation)
    if additions:
        return f"{query} {' '.join(additions)}"
    return query


# ---------------------------------------------------------------------------
# Corpus loading and chunking
# ---------------------------------------------------------------------------

def load_documents(domain: str) -> list[dict]:
    """Load all .md files for a domain."""
    domain_path = DATA_ROOT / domain
    docs        = []
    for md_file in domain_path.rglob("*.md"):
        text = md_file.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            docs.append({
                "text":     text,
                "path":     str(md_file.relative_to(DATA_ROOT)),
                "domain":   domain,
                "filename": md_file.stem,
            })
    return docs


def chunk_document(doc: dict) -> list[dict]:
    """Split a document into overlapping character chunks."""
    text   = doc["text"]
    chunks = []
    start  = 0
    while start < len(text):
        end        = start + CHUNK_SIZE
        chunk_text = text[start:end]
        chunks.append({
            **doc,
            "text":     chunk_text,
            "chunk_id": f"{doc['path'].replace(chr(92), '/')}::{start}",
        })
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def load_all_chunks() -> dict[str, list[dict]]:
    """Return {domain: [chunks]} for all three domains."""
    all_chunks = {}
    for domain in DOMAINS:
        docs   = load_documents(domain)
        chunks = []
        for doc in docs:
            chunks.extend(chunk_document(doc))
        all_chunks[domain] = chunks
        print(f"  [{domain}] {len(docs)} docs → {len(chunks)} chunks")
    return all_chunks


# ---------------------------------------------------------------------------
# Retriever class
# ---------------------------------------------------------------------------

class Retriever:
    def __init__(self):
        print("Loading embedding model...")
        self.embedder = SentenceTransformer(EMBED_MODEL)

        self.chroma                              = chromadb.Client()
        self.collections: dict[str, any]         = {}
        self.bm25_indices: dict[str, BM25Okapi]  = {}
        self.chunks_by_domain: dict[str, list[dict]] = {}

        print("Loading corpus...")
        all_chunks = load_all_chunks()
        for domain, chunks in all_chunks.items():
            self._index_domain(domain, chunks)
        print("Retriever ready.\n")

    def _index_domain(self, domain: str, chunks: list[dict]):
        """Build ChromaDB collection + BM25 index for one domain."""
        self.chunks_by_domain[domain] = chunks

        # BM25
        tokenised = [self._tokenise(c["text"]) for c in chunks]
        self.bm25_indices[domain] = BM25Okapi(tokenised)

        # ChromaDB
        col       = self.chroma.create_collection(name=domain)
        texts     = [c["text"] for c in chunks]
        ids       = [c["chunk_id"] for c in chunks]
        metadatas = [
            {"path": c["path"], "filename": c["filename"]}
            for c in chunks
        ]

        batch_size = 64
        for i in range(0, len(texts), batch_size):
            embeddings = self.embedder.encode(
                texts[i:i + batch_size],
                show_progress_bar=False,
            ).tolist()
            col.add(
                documents=texts[i:i + batch_size],
                embeddings=embeddings,
                ids=ids[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
            )

        self.collections[domain] = col

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def retrieve(
        self,
        query: str,
        domain: Optional[str] = None,
        top_k: int = TOP_K,
    ) -> list[dict]:
        """
        Hybrid BM25 + semantic retrieval merged at 0.7/0.3.
        Used for confidence score calculation.
        Applies vocabulary bridge before searching.
        """
        domains_to_search = [domain] if domain else DOMAINS
        all_results       = []

        bridged_query   = _apply_vocab_bridge(query)
        query_embedding = self.embedder.encode(
            [bridged_query], show_progress_bar=False
        ).tolist()[0]
        query_tokens = self._tokenise(bridged_query)

        for d in domains_to_search:
            chunks = self.chunks_by_domain[d]
            if not chunks:
                continue

            # Semantic
            col         = self.collections[d]
            sem_results = col.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k * 2, len(chunks)),
            )
            sem_scores: dict[str, float] = {
                chunk_id: 1 / (1 + distance)
                for chunk_id, distance in zip(
                    sem_results["ids"][0],
                    sem_results["distances"][0],
                )
            }

            # BM25
            raw_scores = self.bm25_indices[d].get_scores(query_tokens)
            max_bm25   = max(raw_scores) if max(raw_scores) > 0 else 1.0
            bm25_scores: dict[str, float] = {
                chunks[i]["chunk_id"]: raw_scores[i] / max_bm25
                for i in range(len(chunks))
            }

            # Merge 0.7 semantic + 0.3 BM25
            for chunk in chunks:
                cid      = chunk["chunk_id"]
                sem      = sem_scores.get(cid, 0.0)
                bm25_s   = bm25_scores.get(cid, 0.0)
                combined = 0.7 * sem + 0.3 * bm25_s
                all_results.append({**chunk, "score": combined})

        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]

    def retrieve_separate(
        self,
        query: str,
        domain: Optional[str] = None,
        top_k: int = TOP_K,
    ) -> dict:
        """
        Returns BM25 and semantic results separately for LLM selection.
        Applies vocabulary bridge before searching.
        Returns {"bm25": [...], "semantic": [...]}
        """
        domains_to_search = [domain] if domain else DOMAINS
        bm25_results      = []
        semantic_results  = []

        bridged_query   = _apply_vocab_bridge(query)
        query_embedding = self.embedder.encode(
            [bridged_query], show_progress_bar=False
        ).tolist()[0]
        query_tokens = self._tokenise(bridged_query)

        for d in domains_to_search:
            chunks = self.chunks_by_domain[d]
            if not chunks:
                continue

            # Semantic
            col         = self.collections[d]
            sem_results = col.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, len(chunks)),
            )
            for chunk_id, distance in zip(
                sem_results["ids"][0],
                sem_results["distances"][0],
            ):
                chunk = next(
                    (c for c in chunks if c["chunk_id"] == chunk_id),
                    None,
                )
                if chunk:
                    semantic_results.append({
                        **chunk,
                        "score": 1 / (1 + distance),
                    })

            # BM25
            raw_scores  = self.bm25_indices[d].get_scores(query_tokens)
            max_score   = max(raw_scores) if max(raw_scores) > 0 else 1.0
            top_indices = sorted(
                range(len(raw_scores)),
                key=lambda i: raw_scores[i],
                reverse=True,
            )[:top_k]
            for i in top_indices:
                bm25_results.append({
                    **chunks[i],
                    "score": raw_scores[i] / max_score,
                })

        bm25_results.sort(key=lambda x: x["score"], reverse=True)
        semantic_results.sort(key=lambda x: x["score"], reverse=True)

        return {
            "bm25":     bm25_results[:top_k],
            "semantic": semantic_results[:top_k],
        }

    def get_top_score(
        self,
        query: str,
        domain: Optional[str] = None,
    ) -> float:
        """Return highest retrieval score for confidence check."""
        results = self.retrieve(query, domain, top_k=1)
        return results[0]["score"] if results else 0.0