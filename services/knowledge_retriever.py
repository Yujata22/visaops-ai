from pathlib import Path
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "visaops_knowledge"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class KnowledgeBaseError(Exception):
    """Raised when the VisaOPS knowledge base is unavailable."""


class VisaKnowledgeRetriever:
    """Search the locally stored VisaOPS knowledge base."""

    def __init__(self) -> None:
        if not CHROMA_DIR.exists():
            raise KnowledgeBaseError(
                "The Chroma database does not exist. "
                "Run `python scripts/ingest_documents.py` first."
            )

        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

        try:
            self.collection = self.client.get_collection(
                name=COLLECTION_NAME
            )
        except Exception as exc:
            raise KnowledgeBaseError(
                "The VisaOPS collection could not be loaded. "
                "Run the ingestion script again."
            ) from exc

    def search(
        self,
        query: str,
        number_of_results: int = 3,
        visa_category: Optional[str] = None,
    ) -> list[dict]:
        """
        Find knowledge chunks semantically related to a question.
        """

        cleaned_query = query.strip()

        if not cleaned_query:
            return []

        if number_of_results < 1:
            raise ValueError(
                "number_of_results must be at least 1."
            )

        collection_size = self.collection.count()

        if collection_size == 0:
            return []

        query_embedding = self.embedding_model.encode(
            cleaned_query,
            normalize_embeddings=True,
        ).tolist()

        query_arguments = {
            "query_embeddings": [query_embedding],
            "n_results": min(
                number_of_results,
                collection_size,
            ),
            "include": [
                "documents",
                "metadatas",
                "distances",
            ],
        }

        if visa_category:
            query_arguments["where"] = {
                "visa_category": visa_category.lower()
            }

        results = self.collection.query(
            **query_arguments
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        matches = []

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        ):
            metadata = metadata or {}

            matches.append(
                {
                    "content": document,
                    "metadata": {
                        "source": metadata.get(
                            "source",
                            "Unknown source",
                        ),
                        "file_name": metadata.get(
                            "file_name",
                            "Unknown file",
                        ),
                        "title": metadata.get(
                            "title",
                            metadata.get(
                                "file_name",
                                "Unknown source",
                            ),
                        ),
                        "source_url": metadata.get(
                            "source_url",
                            "",
                        ),
                        "visa_category": metadata.get(
                            "visa_category",
                            "general",
                        ),
                        "topic": metadata.get(
                            "topic",
                            "",
                        ),
                        "last_reviewed": metadata.get(
                            "last_reviewed",
                            "",
                        ),
                        "chunk_index": metadata.get(
                            "chunk_index",
                            0,
                        ),
                    },
                    "distance": float(distance),
                }
            )

        return matches