# streamlit_app.py
import streamlit as st
from pathlib import Path
from gensim.models import Word2Vec

MODEL_PATH = Path(__file__).parent / "hf_model_repo" / "word2vec.model"


@st.cache_resource
def load_model():
    return Word2Vec.load(str(MODEL_PATH))


st.set_page_config(page_title="Ingredient Pairing", page_icon=":herb:")
st.title("Herb & Spice Pairing Engine")
st.caption("Type an ingredient — the model finds what goes best with it, learned from 6K recipes.")

model = load_model()
vocab = model.wv

query = st.text_input("Ingredient", placeholder="garlic")

examples = ["garlic", "chocolate", "lemon", "ginger", "basil", "salmon", "honey", "cumin", "chili"]
st.caption("Try one:")
cols = st.columns(len(examples))
for i, ex in enumerate(examples):
    if cols[i].button(ex, key=ex, use_container_width=True):
        query = ex

if query:
    token = query.strip().lower()
    if token in vocab:
        st.subheader(f"Top 5 pairings for {token}")
        results = vocab.most_similar(token, topn=5)
        for rank, (word, score) in enumerate(results, 1):
            st.markdown(f"**{rank}.** {word} — similarity {score:.3f}")
    else:
        st.warning(f"`{token}` not in the model vocabulary. Try a more common ingredient (e.g. garlic, chicken, sugar).")
