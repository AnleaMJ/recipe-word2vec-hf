# train.py
import re
import json
import numpy as np
from pathlib import Path

from gensim.models import Word2Vec
from sklearn.preprocessing import normalize
from huggingface_hub import HfApi, create_repo, upload_file

from data_loader import DATASET_NAME, load_split, load_indian
SAVE_DIR = Path("hf_model_repo")
SAVE_DIR.mkdir(exist_ok=True)


def tokenize_text(text: str) -> list[str]:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    return [t for t in tokens if len(t) > 2]


def preprocess_recipe(recipe: dict) -> list[str]:
    """
    Combine name, NER tags, raw ingredients, and instructions
    into one token list representing the recipe document.
    """
    parts = []

    parts.append(recipe.get("name", ""))

    ner = recipe.get("ner", [])
    if isinstance(ner, list):
        parts.extend(ner)
    elif isinstance(ner, str):
        parts.append(ner)

    ingredients = recipe.get("ingredients", [])
    if isinstance(ingredients, list):
        parts.extend(ingredients)
    elif isinstance(ingredients, str):
        parts.append(ingredients)

    steps = recipe.get("steps", [])
    if isinstance(steps, list):
        parts.extend(steps)
    elif isinstance(steps, str):
        parts.append(steps)

    full_text = " ".join(str(p) for p in parts if p)
    return tokenize_text(full_text)


def train():
    print(f"Loading general dataset: {DATASET_NAME} ...")
    general = load_split("train")
    print(f"Loading Indian dataset ...")
    indian = load_indian()

    all_recipes = general + indian
    print(f"Combined corpus: {len(general)} general + {len(indian)} Indian = {len(all_recipes)} recipes")

    print("Preprocessing recipes...")
    corpus_tokens = []
    metadata = []

    for ex in all_recipes:
        tokens = preprocess_recipe(ex)
        if not tokens:
            continue
        corpus_tokens.append(tokens)

        metadata.append({
            "uid": ex.get("uid"),
            "name": ex.get("name"),
            "link": ex.get("link"),
            "ner": ex.get("ner", []),
            "ingredients": ex.get("ingredients", []),
            "steps": ex.get("steps", []),
        })

    print(f"Corpus size: {len(corpus_tokens)} recipes")

    print("Training Word2Vec...")
    w2v_model = Word2Vec(
        sentences=corpus_tokens,
        vector_size=128,
        window=8,
        min_count=2,
        workers=4,
        epochs=15,
        sg=1,
    )

    print("Building recipe vectors...")
    recipe_vectors = []
    vocab = w2v_model.wv

    for tokens in corpus_tokens:
        vecs = [vocab[t] for t in tokens if t in vocab]
        if vecs:
            vec = np.mean(vecs, axis=0)
        else:
            vec = np.zeros(w2v_model.vector_size)
        recipe_vectors.append(vec)

    recipe_vectors = np.vstack(recipe_vectors)
    recipe_vectors = normalize(recipe_vectors, norm="l2", axis=1)

    w2v_model.save(str(SAVE_DIR / "word2vec.model"))
    np.save(SAVE_DIR / "recipe_vectors.npy", recipe_vectors)
    with open(SAVE_DIR / "metadata.jsonl", "w", encoding="utf-8") as f:
        for m in metadata:
            f.write(json.dumps(m) + "\n")

    print(f"Saved artifacts to ./{SAVE_DIR}")
    print("Sample similar words:", w2v_model.wv.most_similar("chicken", topn=3))


def push_to_hub(repo_id: str):
    """Upload the three artifacts to a Hugging Face Model repo."""
    create_repo(repo_id, repo_type="model", exist_ok=True)
    api = HfApi()

    for file in ["word2vec.model", "recipe_vectors.npy", "metadata.jsonl"]:
        upload_file(
            path_or_fileobj=SAVE_DIR / file,
            path_in_repo=file,
            repo_id=repo_id,
            repo_type="model",
        )
    print(f"Pushed to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    train()

    # push_to_hub(repo_id="your-username/recipe-word2vec")
