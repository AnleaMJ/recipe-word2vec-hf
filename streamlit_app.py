# streamlit_app.py
import streamlit as st
from pathlib import Path
from gensim.models import Word2Vec

MODEL_PATH = Path(__file__).parent / "hf_model_repo" / "word2vec.model"

INDIAN_INGREDIENTS = {
    "cumin", "coriander", "turmeric", "garam", "masala", "chili", "mustard",
    "fenugreek", "asafoetida", "hing", "cardamom", "cinnamon", "cloves",
    "bay", "nutmeg", "mace", "saffron", "curry", "tamarind", "fennel",
    "ajwain", "kalonji", "nigella", "methi", "kashmiri", "paprika",
    "lentil", "chickpea", "paneer", "dal", "chana", "moong", "urad",
    "basmati", "naan", "roti", "paratha", "puri", "dosa", "idli",
    "ghee", "yogurt", "raita", "buttermilk", "dahi", "lassi",
    "ginger", "garlic", "onion", "tomato", "coconut", "mango",
    "pickle", "chutney", "sambar", "rasam",
    "besan", "gram", "semolina", "rava", "atta", "maida",
    "jaggery", "cashew", "almond", "pistachio",
    "cilantro", "mint", "tulsi",
    "khoya", "malai",
    "biryani", "pulao", "khichdi", "uttapam", "pakora", "samosa",
    "vada", "bhel", "chaat", "tikka", "korma", "vindaloo", "jalfrezi",
    "madras", "tandoori", "rogan", "josh", "makhani",
    "seekh", "kebab", "chapati", "kulcha", "bhatura",
    "poha", "upma", "dhokla", "khandvi", "thepla",
    "achar", "launji",
    "kootu", "poriyal", "avial", "thoran", "erissery",
    "halwa", "barfi", "laddu", "gulab", "jamun", "jalebi", "rasgulla",
    "sandesh", "kulfi", "falooda", "payasam",
}


@st.cache_resource
def load_model():
    return Word2Vec.load(str(MODEL_PATH))


st.set_page_config(page_title="Ingredient Pairing", page_icon=":herb:")
st.title("Herb & Spice Pairing Engine")
st.caption("Type an ingredient — the model finds what goes best with it, learned from 6K recipes.")

model = load_model()
vocab = model.wv

indian_mode = st.toggle("Indian cuisine mode", value=False)

query = st.text_input("Ingredient", placeholder="garlic")

default_examples = ["garlic", "chocolate", "lemon", "ginger", "basil", "salmon", "honey", "cumin", "chili"]
indian_examples = ["cumin", "turmeric", "coriander", "garam masala", "cardamom", "ginger", "chili", "fenugreek", "tamarind"]

examples = indian_examples if indian_mode else default_examples
st.caption("Try one:")
cols = st.columns(len(examples))
for i, ex in enumerate(examples):
    if cols[i].button(ex, key=f"ex_{ex}", use_container_width=True):
        query = ex

if query:
    token = query.strip().lower()
    if token in vocab:
        topn = 30 if indian_mode else 5
        all_results = vocab.most_similar(token, topn=topn)

        if indian_mode:
            results = [(w, s) for w, s in all_results if w in INDIAN_INGREDIENTS][:5]
            if not results:
                st.info("No Indian cuisine ingredients found in the top matches. Try a spice (cumin, turmeric, coriander).")
            else:
                st.subheader(f"Top 5 Indian pairings for {token}")
                for rank, (word, score) in enumerate(results, 1):
                    st.markdown(f"**{rank}.** {word} — similarity {score:.3f}")
        else:
            st.subheader(f"Top 5 pairings for {token}")
            for rank, (word, score) in enumerate(all_results, 1):
                st.markdown(f"**{rank}.** {word} — similarity {score:.3f}")
    else:
        st.warning(f"`{token}` not in the model vocabulary. Try a more common ingredient (e.g. garlic, chicken, sugar).")
