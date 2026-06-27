"""Patch notebooks 02-05 for E5 embeddings and token chunking."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent / "notebooks"

SAFE_REPLACEMENTS = [
    ("GeminiEmbedder", "E5Embedder"),
    ("settings.gemini_embedding_model", "settings.embedding_model"),
    ("Gemini embedding", "E5 embedding"),
    ("Gemini embeddings", "E5 embeddings"),
    ("Initialising Gemini embedder", "Loading E5 embedder"),
]

for path in sorted(root.glob("*.ipynb")):
    if path.name == "01_data_exploration.ipynb":
        continue
    text = path.read_text(encoding="utf-8")
    original = text
    for old, new in SAFE_REPLACEMENTS:
        text = text.replace(old, new)
    if text != original:
        path.write_text(text, encoding="utf-8")
        print("patched", path.name)

print("done")
