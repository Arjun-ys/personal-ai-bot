"""Diagnostic: inspect core_identity ChromaDB collection."""
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
identity_db = Chroma(
    collection_name="core_identity",
    persist_directory="./ai_memory_db",
    embedding_function=embeddings,
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
