"""Regenerate notebooks/01_data_exploration.ipynb (token-based EDA)."""
import json
from pathlib import Path


def md(s: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}


def code(s: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": s.splitlines(keepends=True),
    }


cells = [
    md(
        """# Notebook 1 — EDA & Token Analysis (multilingual-e5-large)

**Goal:** Explore Natural Questions and choose `CHUNK_SIZE` / `CHUNK_OVERLAP` in **tokens**
using the **same tokenizer** as `intfloat/multilingual-e5-large`.

- E5 max context: **512 tokens**
- Default chunk budget: **256 tokens** (answer body), overlap **32 tokens**
- Full passage embedded as: `passage: Question: ...\\nAnswer: ...`
"""
    ),
    code(
        """import sys, os
sys.path.append(os.path.abspath('..'))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    'figure.facecolor': '#0f1117', 'axes.facecolor': '#1a1d27',
    'axes.edgecolor': '#3a3f55', 'axes.labelcolor': '#e0e0e0',
    'xtick.color': '#b0b0b0', 'ytick.color': '#b0b0b0',
    'text.color': '#e0e0e0', 'grid.color': '#2a2f45', 'grid.linestyle': '--', 'grid.alpha': 0.5,
})
ACCENT, ACCENT2, ACCENT3 = '#6366f1', '#22d3ee', '#f59e0b'

from app.config import get_settings
from app.rag.preprocessing import clean_text
from app.rag.tokenizer_utils import (
    get_embedding_tokenizer, count_tokens,
    count_embed_passage_tokens,
)

settings = get_settings()
TOKENIZER = get_embedding_tokenizer(settings.embedding_model)
print('Embedding model :', settings.embedding_model)
print('Vocab size      :', TOKENIZER.vocab_size)
print('Max model tokens:', settings.embedding_max_tokens)
print('chunk_size (tok):', settings.chunk_size, '| overlap:', settings.chunk_overlap)
"""
    ),
    md("## 1. Load dataset"),
    code(
        """CSV_PATH = settings.resolve(settings.dataset_path)
df_raw = pd.read_csv(CSV_PATH)
print(f'Shape: {df_raw.shape[0]:,} rows')
print('Columns:', list(df_raw.columns))
df_raw.head(3)
"""
    ),
    md("## 2. Token lengths — questions & answers\n\nWe count tokens with the **E5 tokenizer**, not words or characters."),
    code(
        """SAMPLE = min(5000, len(df_raw))
df = df_raw.head(SAMPLE).copy()
df['q']  = df['question'].apply(clean_text)
df['la'] = df['long_answers'].apply(clean_text)
df['sa'] = df['short_answers'].apply(clean_text)

df['q_tok']  = df['q'].apply(lambda t: count_tokens(t, TOKENIZER))
df['la_tok'] = df['la'].apply(lambda t: count_tokens(t, TOKENIZER))
df['sa_tok'] = df['sa'].apply(lambda t: count_tokens(t, TOKENIZER))

df['passage_tok'] = df.apply(
    lambda r: count_embed_passage_tokens(r['q'], r['la'] or r['sa'], TOKENIZER), axis=1
)

df[['q_tok','la_tok','sa_tok','passage_tok']].describe(percentiles=[.25,.5,.75,.9,.95,.99]).round(1)
"""
    ),
    md("## 3. Distribution of answer length (tokens) — basis for chunk_size"),
    code(
        """la = df['la_tok'].replace(0, np.nan).dropna()
percentiles = {p: np.percentile(la, p) for p in [50, 75, 90, 95, 99]}

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
axes[0].hist(la.clip(upper=600), bins=70, color=ACCENT, edgecolor='none', alpha=0.85)
for p, v in percentiles.items():
    axes[0].axvline(v, lw=1.6, linestyle='--', label=f'p{p}={v:.0f} tok')
axes[0].set_title('Long answer token length'); axes[0].legend(fontsize=9); axes[0].grid(True)

sorted_la = np.sort(la.values)
cdf = np.arange(1, len(sorted_la)+1) / len(sorted_la)
axes[1].plot(sorted_la, cdf, color=ACCENT2, lw=2)
for p, v in percentiles.items():
    axes[1].axvline(v, lw=1.2, linestyle='--', label=f'p{p}={v:.0f}')
axes[1].set_xlim(0, 700); axes[1].set_title('CDF — answer tokens')
axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
axes[1].legend(fontsize=8); axes[1].grid(True)
plt.suptitle('Token distribution (E5 tokenizer)', fontsize=13, y=1.02)
plt.tight_layout(); plt.show()

for p, v in percentiles.items():
    print(f'p{p}: {v:.0f} tokens')
"""
    ),
    md("## 4. Full passage tokens (must stay ≤ 512)"),
    code(
        """pt = df['passage_tok']
over = (pt > settings.embedding_max_tokens).sum()
print(f'Passages over {settings.embedding_max_tokens} tokens: {over} ({100*over/len(df):.1f}%)')

fig, ax = plt.subplots(figsize=(12, 4), facecolor='#0f1117')
ax.set_facecolor('#1a1d27')
ax.hist(pt.clip(upper=600), bins=60, color=ACCENT3, edgecolor='none', alpha=0.85)
ax.axvline(settings.embedding_max_tokens, color='#ef4444', lw=2, linestyle='--', label=f'max={settings.embedding_max_tokens}')
ax.axvline(settings.chunk_size, color='#22c55e', lw=2, linestyle=':', label=f'CHUNK_SIZE={settings.chunk_size}')
ax.legend(); ax.grid(True); plt.tight_layout(); plt.show()
"""
    ),
    md("## 5. Chunk heatmap (token chunk_size × overlap)"),
    code(
        """from app.rag.preprocessing import chunk_text
from app.rag.tokenizer_utils import max_answer_tokens_for_question

chunk_sizes = [128, 192, 256, 384, 512]
overlaps = [16, 32, 48, 64]
sample_texts = df['la'].tolist()[:500]
questions = df['q'].tolist()[:500]

rows = []
for cs in chunk_sizes:
    for co in overlaps:
        if co >= cs:
            continue
        total = 0
        for q, t in zip(questions, sample_texts):
            if not t:
                continue
            budget = max_answer_tokens_for_question(q, cs, settings.embedding_max_tokens, TOKENIZER)
            total += len(chunk_text(t, budget, co, tokenizer=TOKENIZER))
        rows.append({'chunk_size_tok': cs, 'overlap_tok': co,
                     'avg_chunks': round(total / max(len(sample_texts), 1), 2)})

res = pd.DataFrame(rows)
pivot = res.pivot(index='chunk_size_tok', columns='overlap_tok', values='avg_chunks')
display(pivot)

fig, ax = plt.subplots(figsize=(9, 5), facecolor='#0f1117')
ax.set_facecolor('#1a1d27')
im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(pivot.columns)))
ax.set_yticks(range(len(pivot.index)))
ax.set_xticklabels([f'ov={c}' for c in pivot.columns], color='#b0b0b0')
ax.set_yticklabels([f'sz={r}' for r in pivot.index], color='#b0b0b0')
plt.colorbar(im, ax=ax); plt.tight_layout(); plt.show()
"""
    ),
    md("## 6. Recommendation"),
    code(
        """p50, p90 = np.percentile(la, 50), np.percentile(la, 90)
print('='*58)
print('TOKEN chunk_size guide (E5-large, max 512 tokens)')
print('='*58)
print(f'p50 answer tokens: {p50:.0f} | p90: {p90:.0f}')
print('CHUNK_SIZE=256 + OVERLAP=32  -> recommended default')
print(f'Current: CHUNK_SIZE={settings.chunk_size}, OVERLAP={settings.chunk_overlap}')
print('='*58)
"""
    ),
    md("## 7. Chars vs tokens — why we use tokens"),
    code(
        """sample = df['la'].iloc[100]
print(sample[:200], '...')
print('Chars :', len(sample))
print('Tokens:', count_tokens(sample, TOKENIZER))
print('Words :', len(sample.split()))
"""
    ),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.14.0"},
    },
    "cells": cells,
}

out = Path(__file__).resolve().parent.parent / "notebooks" / "01_data_exploration.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Wrote", out)
