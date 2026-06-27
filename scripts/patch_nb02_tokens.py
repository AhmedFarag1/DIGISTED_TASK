"""Add token-chunking cells to notebook 02."""
import json
from pathlib import Path

path = Path("notebooks/02_preprocessing_pipeline.ipynb")
nb = json.loads(path.read_text(encoding="utf-8"))

insert_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "## Step 4 (updated): Token-based chunking — multilingual-e5-large\n",
        "\n",
        "`CHUNK_SIZE` and `CHUNK_OVERLAP` in `.env` are **token counts**, counted with the same tokenizer as the embedding model.\n",
    ],
}

insert_code = {
    "cell_type": "code",
    "metadata": {},
    "outputs": [],
    "execution_count": None,
    "source": [
        "from app.rag.tokenizer_utils import get_embedding_tokenizer, count_tokens, count_embed_passage_tokens\n",
        "\n",
        "tok = get_embedding_tokenizer(settings.embedding_model)\n",
        "SAMPLE_TEXT = clean_text(df['long_answers'].iloc[5])\n",
        "print(f'Answer chars: {len(SAMPLE_TEXT)} | tokens: {count_tokens(SAMPLE_TEXT, tok)}')\n",
        "\n",
        "for cs, co in [(128, 16), (256, 32), (384, 48)]:\n",
        "    chunks = chunk_text(SAMPLE_TEXT, cs, co, tokenizer=tok)\n",
        "    tok_counts = [count_tokens(c, tok) for c in chunks]\n",
        "    print(f'chunk_size={cs} tok, overlap={co} -> {len(chunks)} chunks, token counts={tok_counts}')\n",
    ],
}

# Insert before "## Step 5" if exists, else append
idx = len(nb["cells"])
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if "Step 5" in src and cell["cell_type"] == "markdown":
        idx = i
        break

nb["cells"].insert(idx, insert_code)
nb["cells"].insert(idx, insert_md)
path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Updated", path)
