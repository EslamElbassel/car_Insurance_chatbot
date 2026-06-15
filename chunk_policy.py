"""
chunk_policy.py
===============
Smart structural chunker for the AROPE Aman Gold Motors Insurance PDF.

Strategy:
---------
The document has a clear Arabic hierarchy:
    القسم (Section) → الفصل (Chapter) → البند (Article/Clause)

Instead of splitting blindly by character count, we:
1. Load the PDF page by page.
2. Detect section/chapter/article boundaries using Arabic regex patterns.
3. Accumulate text until a boundary is hit, then emit a chunk.
4. Apply a safety fallback splitter only for oversized chunks.
5. Attach rich metadata (page, section, chapter, article) to every chunk.
6. Build and persist the FAISS index from the resulting chunks.
"""

import os
import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PDF_PATH        = "AMG_cond.pdf"
FAISS_INDEX_DIR = "motors_terms_index"
INDEX_NAME      = "motors_terms_index"
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")

# Safety net: if any structural chunk exceeds this, split it further
MAX_CHUNK_SIZE  = 2000
CHUNK_OVERLAP   = 300

# ---------------------------------------------------------------------------
# Arabic structural boundary patterns
# Priority: higher index = more specific / higher priority
# ---------------------------------------------------------------------------

BOUNDARY_PATTERNS = [
    # Top-level sections  e.g. "القسم الأول", "القسم الثانى"
    re.compile(r"القسم\s+(الأول|الثاني|الثانى|الثالث|الرابع|الخامس)"),

    # Chapters  e.g. "الفصل الأول", "الفصل الثاني"
    re.compile(r"الفصل\s+(الأول|الثاني|الثالث|الرابع|الخامس)"),

    # General conditions header
    re.compile(r"الشروط\s+العامة"),

    # Numbered top-level articles  e.g. "-1 تغير الخطر:", "-2 الإحتياطات"
    re.compile(r"^-\d+\s+[\u0600-\u06FF]"),

    # Exclusions headers
    re.compile(r"(استثناءات|إستثناءات)\s+(خاصة|عامة|ال\s*ت)"),

    # Liability limits header
    re.compile(r"حدود\s+مسئولية\s+الشركة"),

    # Settlement basis header
    re.compile(r"أسس\s+تسوية\s+التعويضات"),

    # Coverage scope header
    re.compile(r"نطاق\s+التأمين"),

    # Risks header
    re.compile(r"الأخطار\s+ال(مغطاة|تي)"),

    # Additional coverages
    re.compile(r"التغطيات\s+(الأساسية|الإضافية|اإلضافية)"),
]


def is_boundary(line: str) -> bool:
    """Return True if the line starts a new structural section."""
    stripped = line.strip()
    if not stripped:
        return False
    return any(p.search(stripped) for p in BOUNDARY_PATTERNS)


def detect_metadata(text: str, current_meta: dict) -> dict:
    """
    Parse a boundary line and update the running metadata dict.
    Returns an updated copy.
    """
    meta = current_meta.copy()

    if re.search(r"القسم\s+(الأول|الثاني|الثانى|الثالث|الرابع|الخامس)", text):
        meta["section"] = text.strip()
        meta["chapter"] = ""
        meta["article"] = ""

    elif re.search(r"الفصل\s+(الأول|الثاني|الثالث|الرابع|الخامس)", text):
        meta["chapter"] = text.strip()
        meta["article"] = ""

    elif re.search(r"^-\d+\s+", text.strip()):
        meta["article"] = text.strip()

    elif re.search(r"الشروط\s+العامة", text):
        meta["chapter"] = text.strip()
        meta["article"] = ""

    elif re.search(r"(استثناءات|إستثناءات)", text):
        meta["article"] = text.strip()

    elif re.search(r"حدود\s+مسئولية|أسس\s+تسوية|نطاق\s+التأمين", text):
        meta["article"] = text.strip()

    return meta


# ---------------------------------------------------------------------------
# Core structural chunker
# ---------------------------------------------------------------------------

def structural_chunk(pages: list[Document]) -> list[Document]:
    """
    Walk through all pages line by line.
    Emit a chunk every time a structural boundary is detected.
    """
    chunks: list[Document] = []

    current_lines: list[str] = []
    current_meta: dict = {
        "section": "",
        "chapter": "",
        "article": "",
        "source":  PDF_PATH,
        "page":    1,
    }
    current_page = 1

    def flush(lines, meta, page):
        text = "\n".join(lines).strip()
        if len(text) > 30:           # discard noise / page headers
            chunks.append(Document(
                page_content=text,
                metadata={**meta, "page": page},
            ))

    for page_doc in pages:
        page_num  = page_doc.metadata.get("page", current_page) + 1
        page_text = page_doc.page_content

        for line in page_text.splitlines():
            if is_boundary(line):
                # Save the accumulated chunk before starting a new one
                flush(current_lines, current_meta, current_page)
                current_meta  = detect_metadata(line, current_meta)
                current_page  = page_num
                current_lines = [line]
            else:
                current_lines.append(line)

    # Flush the last pending chunk
    flush(current_lines, current_meta, current_page)

    return chunks


# ---------------------------------------------------------------------------
# Safety-net splitter for oversized structural chunks
# ---------------------------------------------------------------------------

def apply_safety_split(chunks: list[Document]) -> list[Document]:
    """
    Any structural chunk that is still too large gets split further
    by RecursiveCharacterTextSplitter which respects Arabic sentence
    boundaries (newline → Arabic period → space).
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", "،", " "],
        chunk_size=MAX_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        keep_separator=True,
    )

    final: list[Document] = []
    for chunk in chunks:
        if len(chunk.page_content) > MAX_CHUNK_SIZE:
            sub_chunks = splitter.split_documents([chunk])
            # Propagate metadata to sub-chunks
            for i, sc in enumerate(sub_chunks):
                sc.metadata = {**chunk.metadata, "sub_chunk": i + 1}
            final.extend(sub_chunks)
        else:
            final.append(chunk)

    return final


# ---------------------------------------------------------------------------
# Build and persist FAISS index
# ---------------------------------------------------------------------------

def build_and_save_index(chunks: list[Document]) -> FAISS:
    print(f"[Index] Embedding {len(chunks)} chunks with text-embedding-3-large …")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=OPENAI_API_KEY,
    )

    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)

    Path(FAISS_INDEX_DIR).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(folder_path=FAISS_INDEX_DIR, index_name=INDEX_NAME)
    print(f"[Index] Saved to '{FAISS_INDEX_DIR}/{INDEX_NAME}'.")
    return vectorstore


# ---------------------------------------------------------------------------
# Chunk preview (optional — useful for debugging)
# ---------------------------------------------------------------------------

def preview_chunks(chunks: list[Document], n: int = 5) -> None:
    print(f"\n{'='*60}")
    print(f"Total chunks: {len(chunks)}")
    print(f"{'='*60}")
    for i, chunk in enumerate(chunks[:n]):
        meta = chunk.metadata
        print(f"\n--- Chunk {i+1} ---")
        print(f"  Section : {meta.get('section', '')}")
        print(f"  Chapter : {meta.get('chapter', '')}")
        print(f"  Article : {meta.get('article', '')}")
        print(f"  Page    : {meta.get('page', '')}")
        print(f"  Length  : {len(chunk.page_content)} chars")
        print(f"  Preview : {chunk.page_content[:200].strip()} …")
    print(f"\n{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # 1. Load PDF
    print(f"[Load] Reading '{PDF_PATH}' …")
    loader = PyPDFLoader(PDF_PATH)
    pages  = loader.load()
    print(f"[Load] Loaded {len(pages)} pages.")

    # 2. Structural chunking
    print("[Chunk] Applying structural chunking …")
    chunks = structural_chunk(pages)
    print(f"[Chunk] Produced {len(chunks)} structural chunks.")

    # 3. Safety split oversized chunks
    print("[Split] Applying safety split for oversized chunks …")
    chunks = apply_safety_split(chunks)
    print(f"[Split] Final chunk count: {len(chunks)}")

    # 4. Preview first 5 chunks
    preview_chunks(chunks, n=5)

    # 5. Build & save FAISS index
    build_and_save_index(chunks)

    print("\n✅ Done! FAISS index is ready for the RAG chatbot.")


if __name__ == "__main__":
    main()
