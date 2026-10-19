"""
Local GraphRAG — Document Ingestion Pipeline
Extracts text from PDF/DOCX, chunks it, and embeds into ChromaDB.
Designed for incremental ingestion — adds to existing collections, never overwrites.
"""

import os
import hashlib
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from core.config import (
    CHROMA_DIR,
    CHROMA_COLLECTION,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    processing_status,
)
from core.retrievers import LMStudioEmbeddings


# ═══════════════════════════════════════════════════════════════════════════════
#  TEXT EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_text_from_pdf(file_path: Path) -> list[Document]:
    """Extract text from a PDF file, one Document per page."""
    from pypdf import PdfReader

    reader = PdfReader(str(file_path))
    documents = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            documents.append(Document(
                page_content=text.strip(),
                metadata={
                    "source": file_path.name,
                    "page": i + 1,
                    "total_pages": len(reader.pages),
                    "file_type": "pdf",
                },
            ))

    return documents


def extract_text_from_docx(file_path: Path) -> list[Document]:
    """Extract text from a DOCX file, one Document per non-empty paragraph group."""
    from docx import Document as DocxDocument

    doc = DocxDocument(str(file_path))
    documents = []
    current_text = []
    paragraph_count = 0

    for para in doc.paragraphs:
        if para.text.strip():
            current_text.append(para.text.strip())
            paragraph_count += 1
        elif current_text:
            # Group paragraphs into logical sections
            documents.append(Document(
                page_content="\n".join(current_text),
                metadata={
                    "source": file_path.name,
                    "section": len(documents) + 1,
                    "file_type": "docx",
                },
            ))
            current_text = []

    # Don't forget the last section
    if current_text:
        documents.append(Document(
            page_content="\n".join(current_text),
            metadata={
                "source": file_path.name,
                "section": len(documents) + 1,
                "file_type": "docx",
            },
        ))

    return documents


def extract_text_from_pptx(file_path: Path) -> list[Document]:
    """Extract text from a PowerPoint PPTX file, one Document per slide."""
    try:
        from pptx import Presentation
        prs = Presentation(str(file_path))
        documents = []
        for i, slide in enumerate(prs.slides):
            slide_texts = []
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for paragraph in shape.text_frame.paragraphs:
                        text = "".join(run.text for run in paragraph.runs).strip()
                        if text:
                            slide_texts.append(text)
            if slide_texts:
                documents.append(Document(
                    page_content="\n".join(slide_texts),
                    metadata={
                        "source": file_path.name,
                        "slide": i + 1,
                        "total_slides": len(prs.slides),
                        "file_type": "pptx",
                    }
                ))
        return documents
    except Exception as e:
        return [Document(
            page_content=f"PowerPoint Presentation: {file_path.name} (Error parsing slides: {e})",
            metadata={"source": file_path.name, "file_type": "pptx", "total_pages": 1}
        )]


def extract_text_from_plain(file_path: Path) -> list[Document]:
    """Extract text from plain text, Markdown, CSV, or code files."""
    text = file_path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []
    return [Document(
        page_content=text,
        metadata={
            "source": file_path.name,
            "file_type": file_path.suffix.lstrip(".").lower(),
            "total_pages": 1,
        }
    )]


def extract_text_from_image(file_path: Path) -> list[Document]:
    """Extract metadata, EXIF, and descriptions from images for multimodal indexing."""
    from PIL import Image
    info_lines = [f"Image File: {file_path.name}"]
    try:
        with Image.open(str(file_path)) as img:
            info_lines.append(f"Dimensions: {img.width}x{img.height}")
            info_lines.append(f"Format: {img.format}")
            info_lines.append(f"Color Mode: {img.mode}")
            exif = img.getexif()
            if exif:
                items = [f"{k}: {v}" for k, v in exif.items() if isinstance(v, (str, int, float))]
                if items:
                    info_lines.append(f"EXIF Metadata: {', '.join(items[:8])}")
    except Exception as e:
        info_lines.append(f"Image read info: {e}")

    return [Document(
        page_content="\n".join(info_lines),
        metadata={
            "source": file_path.name,
            "file_type": "image",
            "total_pages": 1,
        }
    )]


def extract_text_from_video(file_path: Path) -> list[Document]:
    """Register video files with metadata descriptor."""
    size_mb = file_path.stat().st_size / (1024 * 1024)
    content = f"Video Asset: {file_path.name}\nSize: {size_mb:.2f} MB\nExtension: {file_path.suffix}\nIndexed for multimodal context references."
    return [Document(
        page_content=content,
        metadata={
            "source": file_path.name,
            "file_type": "video",
            "total_pages": 1,
        }
    )]


def extract_text(file_path: Path) -> list[Document]:
    """Route extraction based on file extension."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(file_path)
    elif suffix in (".docx", ".doc"):
        return extract_text_from_docx(file_path)
    elif suffix in (".pptx", ".ppt"):
        return extract_text_from_pptx(file_path)
    elif suffix in (".txt", ".md", ".csv", ".json", ".py", ".html"):
        return extract_text_from_plain(file_path)
    elif suffix in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"):
        return extract_text_from_image(file_path)
    elif suffix in (".mp4", ".mkv", ".mov", ".avi"):
        return extract_text_from_video(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CHUNKING
# ═══════════════════════════════════════════════════════════════════════════════

def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    # Add chunk IDs based on content hash for dedup
    for chunk in chunks:
        content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()[:12]
        chunk.metadata["chunk_id"] = f"{chunk.metadata.get('source', 'unknown')}_{content_hash}"

    return chunks


# ═══════════════════════════════════════════════════════════════════════════════
#  EMBEDDING INTO CHROMADB
# ═══════════════════════════════════════════════════════════════════════════════

def embed_chunks(chunks: list[Document]) -> int:
    """
    Embed text chunks into ChromaDB. Additive — never overwrites existing data.
    Uses content-hash-based IDs for automatic deduplication.

    Handles two deduplication layers:
    1. Cross-batch: skips chunks already persisted in ChromaDB
    2. Within-batch: skips duplicate chunks from the same ingestion run
       (identical content produces identical chunk_ids)

    Returns:
        Number of new chunks added.
    """
    if not chunks:
        return 0

    embeddings = LMStudioEmbeddings()

    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )

    # Use chunk_id as the document ID for dedup
    ids = [chunk.metadata.get("chunk_id", str(i)) for i, chunk in enumerate(chunks)]

    # Check which IDs already exist to avoid re-embedding
    existing = vectorstore.get(ids=ids)
    existing_ids = set(existing.get("ids", []))

    # Filter to new chunks AND deduplicate within the batch
    seen = set()
    new_chunks = []
    new_ids = []
    for c, cid in zip(chunks, ids):
        if cid not in existing_ids and cid not in seen:
            seen.add(cid)
            new_chunks.append(c)
            new_ids.append(cid)

    if new_chunks:
        vectorstore.add_documents(documents=new_chunks, ids=new_ids)

    return len(new_chunks)


# ═══════════════════════════════════════════════════════════════════════════════
#  FULL INGESTION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def ingest_file(file_path: Path) -> dict:
    """
    Full pipeline: Extract → Chunk → Embed for a single file.
    Updates processing_status at each stage.
    
    Returns:
        Dict with ingestion stats.
    """
    file_name = file_path.name
    stats = {
        "file": file_name,
        "pages_extracted": 0,
        "chunks_created": 0,
        "chunks_embedded": 0,
        "status": "success",
        "error": None,
    }

    try:
        # Stage 1: Extract
        processing_status.update(
            current_file=file_name,
            stage="extracting",
            message=f"📄 Extracting text from {file_name}...",
        )
        documents = extract_text(file_path)
        stats["pages_extracted"] = len(documents)

        if not documents:
            stats["status"] = "skipped"
            stats["error"] = "No text content found"
            return stats

        # Stage 2: Chunk
        processing_status.update(
            stage="chunking",
            message=f"✂️ Splitting {file_name} into chunks...",
        )
        chunks = chunk_documents(documents)
        stats["chunks_created"] = len(chunks)

        # Stage 3: Embed
        processing_status.update(
            stage="embedding",
            message=f"🧮 Embedding {len(chunks)} chunks from {file_name}...",
        )
        new_count = embed_chunks(chunks)
        stats["chunks_embedded"] = new_count

    except Exception as e:
        stats["status"] = "error"
        stats["error"] = str(e)

    return stats


def ingest_files(file_paths: list[Path]) -> list[dict]:
    """
    Ingest multiple files sequentially with progress tracking.
    
    Returns:
        List of per-file stat dicts.
    """
    processing_status.start(total_files=len(file_paths))
    results = []

    for i, file_path in enumerate(file_paths):
        processing_status.update(
            processed_files=i,
            progress=(i / len(file_paths)),
        )

        result = ingest_file(file_path)
        results.append(result)

        processing_status.update(
            processed_files=i + 1,
            progress=((i + 1) / len(file_paths)),
            files_ingested=processing_status.get().get("files_ingested", []) + [result],
        )

        if result["status"] == "error":
            processing_status.fail(f"Failed on {file_path.name}: {result['error']}")
            return results

    processing_status.finish()
    return results
