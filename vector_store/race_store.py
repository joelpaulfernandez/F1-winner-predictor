"""pgvector-backed race similarity search.

Embeds driver-performance summaries with a local HuggingFace sentence-transformer
and stores them in PostgreSQL + pgvector via LangChain's PGVector integration.

Enables semantic queries like "find races similar to Monaco 2023".

Setup:
    # Start PostgreSQL with pgvector (Docker):
    docker run -d --name pgvector \
        -e POSTGRES_PASSWORD=f1pass \
        -p 5432:5432 pgvector/pgvector:pg16

    # Set env var (or add to .env):
    export PGVECTOR_CONNECTION_STRING="postgresql+psycopg2://postgres:f1pass@localhost:5432/f1"

Usage:
    python -m vector_store.race_store          # demo: insert samples + search
    python -m vector_store.race_store --query "street circuit tight corners"
"""
from __future__ import annotations

import argparse
import os
from typing import Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document

COLLECTION_NAME = "f1_races"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_CONNECTION = os.getenv(
    "PGVECTOR_CONNECTION_STRING",
    "postgresql+psycopg2://postgres:f1pass@localhost:5432/f1",
)


def _embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


class RaceVectorStore:
    """Thin wrapper around LangChain PGVector for F1 race embeddings."""

    def __init__(self, connection_string: str = DEFAULT_CONNECTION):
        self._conn = connection_string
        self._emb = _embeddings()
        self._store = PGVector(
            connection_string=connection_string,
            embedding_function=self._emb,
            collection_name=COLLECTION_NAME,
        )

    def add_race(
        self,
        race_id: str,
        summary: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """Embed and store a race summary.

        Args:
            race_id: Unique race identifier e.g. "2023_monaco".
            summary: Free-text description of the race / driver performance.
            metadata: Optional dict of extra fields (year, circuit, winner, etc.).
        """
        doc = Document(
            page_content=summary,
            metadata={"race_id": race_id, **(metadata or {})},
        )
        self._store.add_documents([doc])
        print(f"Stored: {race_id}")

    def find_similar(self, query: str, k: int = 5) -> list[dict]:
        """Return the k most semantically similar races for a free-text query.

        Args:
            query: Natural-language description e.g. "wet race unpredictable result".
            k: Number of results to return.

        Returns:
            List of dicts with keys: race_id, summary, score, metadata.
        """
        results = self._store.similarity_search_with_score(query, k=k)
        output = []
        for doc, score in results:
            output.append(
                {
                    "race_id": doc.metadata.get("race_id", "unknown"),
                    "summary": doc.page_content,
                    "similarity_score": round(float(score), 4),
                    "metadata": {k: v for k, v in doc.metadata.items() if k != "race_id"},
                }
            )
        return output

    def add_races_from_dataframe(self, df) -> None:
        """Bulk-insert races from a DataFrame with columns: race_id, summary, + optional metadata."""
        import pandas as pd  # noqa: F401 — only needed here

        docs = []
        for _, row in df.iterrows():
            meta = {k: v for k, v in row.items() if k not in ("race_id", "summary")}
            docs.append(
                Document(
                    page_content=str(row["summary"]),
                    metadata={"race_id": str(row["race_id"]), **meta},
                )
            )
        self._store.add_documents(docs)
        print(f"Stored {len(docs)} races.")


# ---------------------------------------------------------------------------
# Demo / CLI
# ---------------------------------------------------------------------------

SAMPLE_RACES = [
    {
        "race_id": "2023_monaco",
        "summary": (
            "Narrow street circuit through Monte Carlo. Verstappen dominated from pole "
            "in dry conditions. Alonso held P3 with strong Aston Martin pace. "
            "Almost no overtaking — qualifying largely decided the result."
        ),
        "metadata": {"year": 2023, "circuit": "Monaco", "winner": "Verstappen", "wet": False},
    },
    {
        "race_id": "2021_belgium",
        "summary": (
            "Race declared behind safety car due to heavy rain and poor visibility at Spa. "
            "Verstappen awarded win after two laps behind safety car. "
            "Controversy over half-points. No racing lap completed."
        ),
        "metadata": {"year": 2021, "circuit": "Spa-Francorchamps", "winner": "Verstappen", "wet": True},
    },
    {
        "race_id": "2023_singapore",
        "summary": (
            "Street circuit with tight barriers. Sainz won from pole for Ferrari. "
            "Red Bull struggled on the bumpy surface. Hamilton and Norris completed podium. "
            "Huge upset — Verstappen finished 5th."
        ),
        "metadata": {"year": 2023, "circuit": "Marina Bay", "winner": "Sainz", "wet": False},
    },
    {
        "race_id": "2021_hungary",
        "summary": (
            "Chaotic first lap — Bottas triggered a multi-car collision eliminating Verstappen. "
            "Ocon took surprise win for Alpine. Hamilton charged from last to second. "
            "Wet-dry strategy played a key role."
        ),
        "metadata": {"year": 2021, "circuit": "Hungaroring", "winner": "Ocon", "wet": True},
    },
    {
        "race_id": "2023_bahrain",
        "summary": (
            "Season opener at Sakhir. Verstappen won comfortably from pole. "
            "Ferrari struggled with tyre degradation. Alonso surprised with P3 on debut for Aston Martin."
        ),
        "metadata": {"year": 2023, "circuit": "Bahrain", "winner": "Verstappen", "wet": False},
    },
]


def demo(query: str = "chaotic wet race with safety car", k: int = 3, connection: str = DEFAULT_CONNECTION):
    print(f"Connecting to: {connection}")
    store = RaceVectorStore(connection_string=connection)

    print("\nInserting sample races...")
    for r in SAMPLE_RACES:
        store.add_race(r["race_id"], r["summary"], r["metadata"])

    print(f'\nSearching for: "{query}"')
    results = store.find_similar(query, k=k)

    print(f"\nTop {k} similar races:")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['race_id']}  (score={r['similarity_score']})")
        print(f"     {r['summary'][:100]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1 pgvector demo")
    parser.add_argument("--query", default="chaotic wet race with safety car")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--connection", default=DEFAULT_CONNECTION)
    args = parser.parse_args()
    demo(query=args.query, k=args.k, connection=args.connection)
