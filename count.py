"""Diagnostic: inspect core_identity ChromaDB collection."""
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    encode_kwargs={"normalize_embeddings": True}
)
identity_db = Chroma(
    collection_name="core_identity",
    persist_directory="./ai_memory_db",
    embedding_function=embeddings,
    collection_metadata={"hnsw:space": "cosine"},
)

col = identity_db._collection
data = col.get(include=["documents", "metadatas"])
print(f"\n🧬 Total identity facts: {col.count()}\n")
for i, (doc, meta) in enumerate(zip(data["documents"], data["metadatas"]), 1):
    src = meta.get("source_snippet", meta.get("source", ""))[:60]
    conf = meta.get("confidence", "?")
    cat = meta.get("category", "?")
    extracted = meta.get("extracted_at", "?")[:19]
    print(f"  {i:>2}. [{conf:>6}] [{cat:<12}] {doc}")
    print(f"       extracted: {extracted}  source: {src}")
