# analyze_dataset.py
import json
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_loader import DATASET_NAME, load_all

OUT_DIR = Path(__file__).parent / "analysis"
OUT_DIR.mkdir(exist_ok=True)

plt.style.use("ggplot")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def tokenize(text: str) -> list[str]:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 2]


def doc_tokens(recipe: dict) -> list[str]:
    parts = [str(recipe.get("name", ""))]
    for field in ("ner", "ingredients", "steps"):
        parts.extend(str(p) for p in recipe.get(field, []))
    return tokenize(" ".join(parts))


def describe_lengths(values, label):
    arr = np.asarray(values)
    return {
        label: {
            "mean": round(float(arr.mean()), 2),
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
            "min": int(arr.min()),
            "max": int(arr.max()),
        }
    }


def analyze_split(rows):
    n_ing = [len(r.get("ingredients", [])) for r in rows]
    n_steps = [len(r.get("steps", [])) for r in rows]

    ner_counter = Counter()
    token_counter = Counter()
    doc_len = []

    for r in rows:
        ner_counter.update(tag.lower() for tag in r.get("ner", []))
        toks = doc_tokens(r)
        doc_len.append(len(toks))
        token_counter.update(toks)

    vocab_full = len(token_counter)
    vocab_min2 = sum(1 for c in token_counter.values() if c >= 2)
    total_tokens = sum(token_counter.values())
    coverage = sum(c for c in token_counter.values() if c >= 2) / total_tokens * 100

    stats = {
        "num_rows": len(rows),
        "ingredients_per_recipe": describe_lengths(n_ing, "count"),
        "steps_per_recipe": describe_lengths(n_steps, "count"),
        "tokens_per_recipe": describe_lengths(doc_len, "count"),
        "corpus_tokens_total": total_tokens,
        "vocab_size_raw": vocab_full,
        "vocab_size_min_count_2": vocab_min2,
        "token_coverage_at_min_count_2_pct": round(coverage, 1),
        "top_25_ner_ingredients": dict(ner_counter.most_common(25)),
        "top_25_tokens": dict(token_counter.most_common(25)),
    }

    charts = {"n_ing": n_ing, "n_steps": n_steps, "doc_len": doc_len}
    sample = [{k: r.get(k) for k in ("uid", "name", "ner", "ingredients")} for r in rows[:3]]
    return stats, charts, sample


def save_charts(train_charts, top_ing):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(f"{DATASET_NAME} - train split distributions", fontsize=13)

    axes[0].hist(train_charts["n_ing"], bins=range(0, max(train_charts["n_ing"]) + 2), color="#e07a5f")
    axes[0].set_title("Ingredients per recipe")
    axes[0].set_xlabel("# ingredients")

    axes[1].hist(train_charts["n_steps"], bins=range(0, max(train_charts["n_steps"]) + 2), color="#3d405b")
    axes[1].set_title("Instruction sentences per recipe")
    axes[1].set_xlabel("# steps")

    axes[2].hist(train_charts["doc_len"], bins=50, color="#81b29a")
    axes[2].set_title("Tokens per recipe (training tokenizer)")
    axes[2].set_xlabel("# tokens")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "distributions.png", dpi=130)
    plt.close(fig)

    names = list(top_ing.keys())[::-1]
    counts = list(top_ing.values())[::-1]
    fig2, ax = plt.subplots(figsize=(8, 7))
    ax.barh(names, counts, color="#f2cc8f")
    ax.set_title("Top 25 ingredients (NER tags)")
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "top_ingredients.png", dpi=130)
    plt.close(fig2)


def main():
    print(f"Loading {DATASET_NAME} from local cache (downloads if missing) ...")
    data = load_all()

    all_stats = {"dataset": DATASET_NAME, "splits": {}}

    for split_name, rows in data.items():
        print(f"\n=== Analyzing split: {split_name} ({len(rows)} recipes) ===")
        stats, charts, sample = analyze_split(rows)
        all_stats["splits"][split_name] = stats
        print(json.dumps(stats, indent=2)[:2400])
        if split_name == "train":
            train_charts = charts
            train_top_ing = stats["top_25_ner_ingredients"]
            with open(OUT_DIR / "sample_recipes.json", "w", encoding="utf-8") as f:
                json.dump(sample, f, indent=2, ensure_ascii=False)

    save_charts(train_charts, train_top_ing)

    with open(OUT_DIR / "stats.json", "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)

    print(f"\nSaved stats    -> analysis/stats.json")
    print(f"Saved charts   -> analysis/distributions.png, analysis/top_ingredients.png")
    print(f"Saved samples  -> analysis/sample_recipes.json")


if __name__ == "__main__":
    main()
