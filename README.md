# Recipe "Generator" with Word2Vec + Semantic Similarity

A step-by-step guide to building, understanding, and deploying a recipe recommendation engine that trains a **Word2Vec** model on a combined corpus of Western and Indian cooking recipes, converts every recipe into a document vector, and uses **cosine similarity** to retrieve the closest recipes to a user's query — presented as a "generated" recipe card in a **Gradio** web app hosted on **Hugging Face Spaces**. A **Streamlit** ingredient pairing UI is also included, with an Indian cuisine toggle.

> **Honest framing:** this is *retrieval*, not true text generation. The "generation" is the rendering of the most semantically similar recipe from the corpus. Section 10 covers how to upgrade it to true generation (RAG / LLM).

---

## 1. What it does (user's view)

1. User types what they have or crave: *"chicken tomato basil pasta"*.
2. The app vectorizes the query with the trained Word2Vec model.
3. It compares that vector against all ~13K recipe vectors (a single matrix multiply).
4. The top-k most similar recipes are rendered as full recipe cards (ingredients + steps).

A separate **Streamlit** UI lets users explore ingredient-to-ingredient pairings with an **Indian cuisine toggle** that filters results to Indian spices, legumes, and staples.

## 2. How it works (under the hood)

```
 general recipes (6,118) + Indian recipes (6,871)
        │  lowercase, strip punctuation, drop tokens ≤ 2 chars
        ▼
 one token list per recipe  ──────────────►  Word2Vec (Skip-gram, 128-dim)
        │                                          │
        │                                          ▼
        │                            mean of word vectors per recipe
        │                                          │
        ▼                                          ▼
   metadata.jsonl ◄──────────────  recipe_vectors.npy (L2-normalized)
                                                    ▲
 user query ──► same tokenization ──► mean vector ──┘
                                                    │
                                    dot product = cosine similarity
                                                    ▼
                                          top-k recipes → UI
```

| Stage | What happens | Why |
|---|---|---|
| **Corpus** | `recipe_nlg_lite` (6,118 general) + `IndianFoodDataset` (6,871 Indian) = **12,989 recipes** | Covers both Western home cooking and Indian cuisine for richer embeddings |
| **Tokenize** | Each recipe = one document of name + NER tags + ingredients + steps | Co-occurrence of food words in the same document is the signal |
| **Word2Vec** | Skip-gram (`sg=1`), 128-dim, window 8, min_count 2, 15 epochs | Skip-gram works well on smaller corpora; rare-ish food words still get good vectors |
| **Doc vector** | Mean of all in-vocab word vectors, then L2-normalize | Simple, fast, surprisingly strong baseline (centroid of the recipe's semantics) |
| **Query vector** | Same averaging on the user's input | Puts query and docs in the same space |
| **Retrieval** | `recipe_vectors @ query_vec` → top-k | Both sides L2-normalized, so dot product **is** cosine similarity |

**Why L2-normalize?** After normalization, `cos(a,b) = a·b`, so the entire similarity search is one NumPy matrix-vector product over 13K×128 floats — sub-millisecond.

---

## 3. Repository structure

```
recipe-word2vec-hf/
├── README.md              ← you are here
├── requirements.txt
├── data_loader.py         ← downloads + parses both datasets into clean dicts
├── analyze_dataset.py     ← EDA: stats + charts into analysis/
├── train.py               ← Word2Vec training on combined corpus + recipe vectors
├── app.py                 ← Gradio web app for HF Spaces
├── streamlit_app.py       ← Streamlit ingredient pairing UI (with Indian toggle)
├── analysis/              ← generated: stats.json, charts, samples
└── data/                  ← generated: raw dataset archives + CSVs
```

---

## 4. Step 1 — Install dependencies

Python **3.10–3.12** recommended.

```bash
pip install -r requirements.txt
```

Key pins: `gensim>=4.3.3` with `numpy<2.0` and `scipy<1.14` (a known-good compatibility trio for gensim on Python 3.12).

> **Note on `datasets`:** the original design used `load_dataset("m3hrdadfi/recipe_nlg_lite")`, but that repo ships a legacy loading **script** which `datasets>=3.x` no longer executes. This repo instead downloads the underlying archive directly (via `gdown`) and parses it with `data_loader.py` — no `datasets` dependency, works on any modern install.

## 5. Step 2 — Fetch & analyze the datasets

```bash
python analyze_dataset.py
```

First run downloads both datasets to `data/` and writes EDA artifacts to `analysis/`.

### Datasets

| Dataset | Source | Recipes | License |
|---|---|---|---|
| `recipe_nlg_lite` | [HuggingFace (m3hrdadfi)](https://huggingface.co/datasets/m3hrdadfi/recipe_nlg_lite) | 6,118 train + 1,080 test | MIT |
| `IndianFoodDataset` | [Mendeley Data (xsphgmmh7b)](https://data.mendeley.com/datasets/xsphgmmh7b/1) via [GitHub mirror](https://github.com/nileshely/Indian-Food) | 6,871 | CC BY 4.0 |

### Raw format gotchas handled by `data_loader.py`

**General dataset:** CSVs store `ner`, `ingredients`, `steps` as comma-separated strings, not lists. `steps` is a single text blob whose sentences are separated by `". "` while commas appear *inside* sentences.

- `ner`, `ingredients` → split on `", "`
- `steps` → split on `". "` into individual instruction sentences

**Indian dataset:** Ingredients contain quantities, units, Hindi names in parentheses, and prep instructions after ` - `. The loader strips all of these to extract clean ingredient names (e.g. `3 tablespoon Gram flour (besan)` → `gram flour`).

(The original spec silently produced empty lists for string fields — this loader fixes that latent bug.)

### Analysis findings (general dataset)

| Metric | Train | Test |
|---|---|---|
| Recipes | **6,118** | 1,080 (7,198 total) |
| Ingredients/recipe (mean / median / max) | 10.2 / 9 / 62 | 10.0 / 9 / 39 |
| Instruction sentences/recipe (mean / median / max) | 11.6 / 10 / 107 | 11.5 / 10 / 108 |
| Tokens/recipe after tokenizer (mean / p90 / max) | 172 / 300 / 1,564 | 170 / 296 / 1,407 |
| Total training tokens | **1,052,047** | 183,392 |
| Vocabulary (raw → `min_count=2`) | 13,980 → **9,059** | 6,840 → 4,544 |
| Token occurrences kept at `min_count=2` | **99.5%** | 98.7% |

Key takeaways:

- **All distributions are right-skewed** — most recipes are compact (≈9 ingredients, ≈10 steps) with a long tail. See `analysis/distributions.png`.
- **`min_count=2` is validated by the data**: discards 35% of vocabulary *types* but only 0.5% of token *occurrences*.
- **Top ingredients** (`analysis/top_ingredients.png`): `salt` (2,115), `olive oil` (1,119), `garlic` (1,050), `water`, `black pepper`, `sugar`, `butter`…
- **Data cleaning:** 1,298 train recipes carried the literal NER tag `"none"` — filtered out in `data_loader.py`.

### Analysis findings (Indian dataset)

| Metric | Value |
|---|---|
| Recipes | **6,871** |
| Unique cuisines | Indian (plus regional: South Indian, Punjabi, Gujarati, Bengali, etc.) |
| Dietary categories | Vegetarian, Diabetic Friendly, Gluten Free, etc. |
| Top ingredients (cleaned) | `salt`, `turmeric`, `cumin`, `ginger`, `garlic`, `onion`, `chili`, `coriander`, `ghee`, `coconut` |

![Dataset distributions](analysis/distributions.png)

![Top 25 ingredients](analysis/top_ingredients.png)

Artifacts produced: `analysis/stats.json`, `analysis/distributions.png`, `analysis/top_ingredients.png`, `analysis/sample_recipes.json`.

## 6. Step 3 — Train

```bash
python train.py
```

Pipeline (see `train.py`):

1. Load **6,118 general** recipes via `data_loader.load_split("train")`.
2. Load **6,871 Indian** recipes via `data_loader.load_indian()`.
3. Combine into a single 12,989-recipe corpus.
4. Tokenize each recipe into one document (name + NER + ingredients + steps).
5. Train Skip-gram Word2Vec: `vector_size=128, window=8, min_count=2, epochs=15, workers=4`. Runs in ~2–3 minutes on CPU.
6. Average word vectors per recipe → 12,989×128 matrix → L2-normalize rows.
7. Save to `hf_model_repo/`:
   - `word2vec.model` — gensim model
   - `recipe_vectors.npy` — the searchable matrix
   - `metadata.jsonl` — one JSON per line: `uid, name, link, ner, ingredients, steps`

Sanity check printed at the end: `most_similar("chicken")` should return food-adjacent words (`breasts`, `thighs`...). If it returns units like `cup`, training went wrong.

### Verified Indian pairings (post-training)

| Query | Top 5 (filtered to Indian ingredients) |
|---|---|
| `garlic` | cumin, ginger, coriander, fennel |
| `cumin` | coriander, turmeric, fennel, garam, ginger |
| `turmeric` | garam, cumin, ginger, coriander |
| `ginger` | cumin, garlic, turmeric, garam, fennel |

## 7. Step 4 — Push artifacts to the Hugging Face Hub

Create a write token at <https://huggingface.co/settings/tokens>, then either uncomment in `train.py`:

```python
push_to_hub(repo_id="your-username/recipe-word2vec")
```

or upload manually:

```bash
huggingface-cli login
huggingface-cli upload your-username/recipe-word2vec hf_model_repo/word2vec.model --repo-type model
huggingface-cli upload your-username/recipe-word2vec hf_model_repo/recipe_vectors.npy --repo-type model
huggingface-cli upload your-username/recipe-word2vec hf_model_repo/metadata.jsonl --repo-type model
```

## 8. Step 5 — Deploy the Gradio app on HF Spaces

1. Edit `app.py` line: `MODEL_REPO = "your-username/recipe-word2vec"`.
2. Test locally: `python app.py` → open `http://127.0.0.1:7860`.
3. Create a Space at <https://huggingface.co/new-space> (SDK: **Gradio**, blank).
4. Commit `app.py` + `requirements.txt` to the Space repo and push:

```bash
git add app.py requirements.txt
git commit -m "init recipe app"
git push
```

HF builds the container, pip-installs deps, downloads your three artifacts at startup, and serves the UI. Try the bundled examples (`chicken tomato cheese pasta`, `chocolate flour egg butter sugar`, ...).

## 9. Streamlit — Ingredient Pairing UI

A lightweight alternative focused on **ingredient-to-ingredient** similarity (no recipe retrieval). Uses `word2vec.most_similar()` to find what goes with a given ingredient.

```bash
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`. Type an ingredient (e.g. `garlic`, `chocolate`, `cumin`) and get the top 5 semantically closest ingredients.

**Indian cuisine toggle:** flip the switch to filter results to Indian spices, legumes, dairy, and dishes. The model was trained on 6,871 Indian recipes, so Indian ingredient embeddings are strong natively. The toggle fetches top 30 candidates and filters to a curated list of ~80 Indian ingredients for clean results.

---

## 10. Extending

- **True generation:** fine-tune GPT-2 / a small LLM on the corpus and feed the retrieved top-3 recipes as in-context examples (RAG).
- **Better embeddings:** swap Word2Vec for `sentence-transformers/all-MiniLM-L6-v2` (`util.cos_sim`) — less custom code, usually stronger retrieval.
- **Diet filters:** post-filter retrieved hits by NER tags (and remember to drop that `"none"` tag first).
- **Faster startup:** save gensim `KeyedVectors` only (`model.wv.save`) instead of the full model.

## 11. Troubleshooting

| Symptom | Fix |
|---|---|
| Google Drive download fails / quota hit | Re-run later, or place the extracted CSVs in `data/recipe_nlg_lite/` yourself |
| Indian dataset download fails | Place `IndianFoodDataset.csv` in `data/` manually |
| `gensim` import error on numpy 2.x | Keep `numpy<2.0` + `scipy<1.14` from `requirements.txt` |
| All similarities ≈ 0 | Query words missing from vocab (min_count=2); use common food words |
| Space build fails on gensim | Same numpy/scipy pins apply in the Space's `requirements.txt` |

---

*Datasets: [m3hrdadfi/recipe_nlg_lite](https://huggingface.co/datasets/m3hrdadfi/recipe_nlg_lite) (MIT, 7,198 recipes) + [Kanishka Jain's Indian Food Dataset](https://data.mendeley.com/datasets/xsphgmmh7b/1) (CC BY 4.0, 6,871 recipes). Combined corpus: 12,989 recipes.*
