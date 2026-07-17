from pathlib import Path
import hashlib
import shutil

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "visaops_knowledge"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 120


def read_markdown_files() -> list[Path]:
    """Return all Markdown knowledge documents except README files."""

    files = [
        file
        for file in KNOWLEDGE_BASE_DIR.rglob("*.md")
        if file.name.lower() != "readme.md"
    ]

    return sorted(files)


def clean_text(text: str) -> str:
    """Normalize spacing while keeping the document readable."""

    lines = []

    for line in text.splitlines():
        cleaned_line = line.strip()

        if cleaned_line:
            lines.append(cleaned_line)

    return "\n".join(lines)


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into overlapping chunks.

    The overlap helps preserve context across chunk boundaries.
    """

    if not text:
        return []

    if overlap >= chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]

        if end < len(text):
            possible_breaks = [
                chunk.rfind("\n\n"),
                chunk.rfind("\n"),
                chunk.rfind(". "),
                chunk.rfind(" "),
            ]

            best_break = max(possible_breaks)

            if best_break > chunk_size // 2:
                end = start + best_break + 1
                chunk = text[start:end]

        cleaned_chunk = chunk.strip()

        if cleaned_chunk:
            chunks.append(cleaned_chunk)

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


def create_chunk_id(
    source_path: str,
    chunk_index: int,
    chunk_text: str,
) -> str:
    """Create a stable unique identifier for a document chunk."""

    raw_id = f"{source_path}:{chunk_index}:{chunk_text}"
    return hashlib.sha256(raw_id.encode("utf-8")).hexdigest()


def infer_visa_category(file_path: Path) -> str:
    """Use the parent folder name as the visa category."""

    return file_path.parent.name.lower()


def reset_chroma_database() -> None:
    """Delete the previous local database before rebuilding it."""

    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)


def ingest_documents() -> None:
    markdown_files = read_markdown_files()

    if not markdown_files:
        raise FileNotFoundError(
            f"No Markdown documents were found in {KNOWLEDGE_BASE_DIR}"
        )

    print(f"Found {len(markdown_files)} knowledge documents.")

    reset_chroma_database()

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "VisaOPS educational immigration knowledge base",
            "embedding_model": EMBEDDING_MODEL_NAME,
        },
    )

    all_ids = []
    all_documents = []
    all_embeddings = []
    all_metadata = []

    for file_path in markdown_files:
        raw_text = file_path.read_text(
            encoding="utf-8"
        )

        cleaned_text = clean_text(raw_text)
        chunks = split_text(cleaned_text)

        relative_path = file_path.relative_to(
            PROJECT_ROOT
        ).as_posix()

        visa_category = infer_visa_category(file_path)

        print(
            f"Processing {relative_path}: "
            f"{len(chunks)} chunk(s)"
        )

        for chunk_index, chunk_text in enumerate(chunks):
            chunk_id = create_chunk_id(
                source_path=relative_path,
                chunk_index=chunk_index,
                chunk_text=chunk_text,
            )

            embedding = embedding_model.encode(
                chunk_text,
                normalize_embeddings=True,
            ).tolist()

            all_ids.append(chunk_id)
            all_documents.append(chunk_text)
            all_embeddings.append(embedding)

            all_metadata.append(
                {
                    "source": relative_path,
                    "file_name": file_path.name,
                    "visa_category": visa_category,
                    "chunk_index": chunk_index,
                }
            )

    if not all_documents:
        raise ValueError(
            "Documents were found, but no text chunks were created."
        )

    collection.add(
        ids=all_ids,
        documents=all_documents,
        embeddings=all_embeddings,
        metadatas=all_metadata,
    )

    print()
    print("Knowledge-base ingestion complete.")
    print(f"Documents indexed: {len(markdown_files)}")
    print(f"Chunks indexed: {len(all_documents)}")
    print(f"Database location: {CHROMA_DIR}")
    print(f"Collection size: {collection.count()}")


if __name__ == "__main__":
    ingest_documents()