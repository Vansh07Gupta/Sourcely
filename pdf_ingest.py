import os
import tempfile
from typing import List, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

_DENSE_PAGE_CHARS = 3000
_SPARSE_PAGE_CHARS = 1000
_DENSE_CHUNK_SIZE = 800
_TYPICAL_CHUNK_SIZE = 1200
_SPARSE_CHUNK_SIZE = 1600
_OVERLAP_RATIO = 0.125
_MIN_OVERLAP = 100


def extract_pdf_chunks(pdf_file) -> List[Tuple[str, str]]:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_path = tmp_file.name
        pages = PyPDFLoader(tmp_path).load()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    if not pages:
        return []

    chunk_size, chunk_overlap = _auto_chunk_params(pages)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(pages)
    if not chunks:
        return []

    return [
        (_chunk_label(chunk, idx, len(chunks)), chunk.page_content)
        for idx, chunk in enumerate(chunks)
    ]


def _auto_chunk_params(pages: List) -> Tuple[int, int]:
    total_chars = sum(len(page.page_content) for page in pages)
    avg_chars_per_page = total_chars / len(pages) if pages else 0

    if avg_chars_per_page > _DENSE_PAGE_CHARS:
        chunk_size = _DENSE_CHUNK_SIZE
    elif avg_chars_per_page < _SPARSE_PAGE_CHARS:
        chunk_size = _SPARSE_CHUNK_SIZE
    else:
        chunk_size = _TYPICAL_CHUNK_SIZE

    chunk_overlap = max(_MIN_OVERLAP, round(chunk_size * _OVERLAP_RATIO))
    return chunk_size, chunk_overlap


def _chunk_label(chunk, idx: int, total: int) -> str:
    page = chunk.metadata.get('page')
    page_part = f"page {page + 1}" if isinstance(page, int) else "page ?"
    return f"{page_part}, chunk {idx + 1}/{total}"
