# app.py
import json
import numpy as np
import gradio as gr

from gensim.models import Word2Vec
from sklearn.preprocessing import normalize
from huggingface_hub import hf_hub_download


MODEL_REPO = "your-username/recipe-word2vec"  # CHANGE THIS
EMBED_DIM = 128


def load_artifacts():
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename="word2vec.model", repo_type="model")
    vectors_path = hf_hub_download(repo_id=MODEL_REPO, filename="recipe_vectors.npy", repo_type="model")
    meta_path = hf_hub_download(repo_id=MODEL_REPO, filename="metadata.jsonl", repo_type="model")

    model = Word2Vec.load(model_path)
    recipe_vectors = np.load(vectors_path)

    metadata = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            metadata.append(json.loads(line))

    return model, recipe_vectors, metadata


print("Downloading artifacts from Hub...")
w2v_model, RECIPE_VECTORS, METADATA = load_artifacts()
VOCAB = w2v_model.wv


def vectorize_query(text: str):
    tokens = [t for t in text.lower().split() if len(t) > 2]
    vecs = [VOCAB[t] for t in tokens if t in VOCAB]
    if not vecs:
        return np.zeros(EMBED_DIM)
    vec = np.mean(vecs, axis=0)
    vec = normalize(vec.reshape(1, -1), norm="l2")[0]
    return vec


def recommend(query: str, top_k: int = 3):
    if not query.strip():
        return "Please enter some ingredients or a dish description."

    q_vec = vectorize_query(query)
    sims = RECIPE_VECTORS @ q_vec
    top_idx = np.argsort(sims)[::-1][:top_k]

    cards = []
    for rank, idx in enumerate(top_idx, 1):
        rec = METADATA[idx]
        score = float(sims[idx])

        ingredients_md = "\n".join(f"- {ing}" for ing in rec.get("ingredients", [])[:12])
        steps_md = "\n".join(f"{i+1}. {step}" for i, step in enumerate(rec.get("steps", [])[:6]))

        card = f"""
### {rank}. {rec['name']} (similarity: {score:.3f})
**Ingredients**
{ingredients_md}

**Steps**
{steps_md}
"""
        cards.append(card)

    return "\n---\n".join(cards)


with gr.Blocks(title="Recipe Recommender") as demo:
    gr.Markdown("""
    # 🍲 Recipe Generator (Word2Vec + Semantic Similarity)
    Enter ingredients you have (e.g. *"chicken tomato basil pasta"*) or a dish idea.
    The engine finds the semantically closest recipe from a 7K-recipe corpus.
    """)

    with gr.Row():
        query_input = gr.Textbox(
            label="Your ingredients / idea",
            placeholder="chicken garlic lemon rosemary...",
            lines=2,
        )
        top_k = gr.Slider(1, 5, value=3, step=1, label="Number of recipes")

    generate_btn = gr.Button("Generate Recipe", variant="primary")
    output = gr.Markdown(label="Generated Recipe")

    generate_btn.click(fn=recommend, inputs=[query_input, top_k], outputs=output)

    gr.Examples(
        examples=[
            ["chicken tomato cheese pasta", 3],
            ["chocolate flour egg butter sugar", 3],
            ["lentil spinach cumin garlic", 2],
            ["spicy shrimp noodle soup", 2],
        ],
        inputs=[query_input, top_k],
    )

if __name__ == "__main__":
    demo.launch()
