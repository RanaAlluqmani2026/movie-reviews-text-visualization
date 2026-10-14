import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path
import urllib.request
import tarfile

st.set_page_config(page_title="Movie Reviews Explorer", layout="wide")

st.title("Movie Reviews Text Visualization Explorer")
st.write(
    "Explore word frequencies and patterns in real IMDb movie reviews."
)

# Load a sample of real IMDb movie reviews
@st.cache_data
def load_reviews():
    data_dir = Path("aclImdb")
    if not data_dir.exists():
        url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
        archive = Path("/tmp/aclImdb_v1.tar.gz")
        urllib.request.urlretrieve(url, archive)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(path=".", filter="data")

    dataset = load_files(
        str(data_dir / "test"),
        categories=["pos", "neg"],
        encoding="utf-8",
        random_state=42
    )
    df = pd.DataFrame({
        "review": dataset.data,
        "sentiment": [dataset.target_names[i] for i in dataset.target]
    })
    return df.groupby("sentiment", group_keys=False).sample(
        n=250, random_state=42
    ).reset_index(drop=True)

df = load_reviews()

st.sidebar.header("Controls")
sentiment = st.sidebar.selectbox(
    "Select Review Type",
    ["All Reviews", "Positive", "Negative"]
)

if sentiment == "Positive":
    filtered_df = df[df["sentiment"] == "pos"]
elif sentiment == "Negative":
    filtered_df = df[df["sentiment"] == "neg"]
else:
    filtered_df = df

st.metric("Number of Reviews", len(filtered_df))

st.subheader("Most Frequent Words")

top_n = st.slider("Number of Words", 5, 30, 15)

vectorizer = CountVectorizer(
    stop_words="english",
    max_features=1000
)

word_matrix = vectorizer.fit_transform(filtered_df["review"])
frequencies = pd.DataFrame({
    "Word": vectorizer.get_feature_names_out(),
    "Frequency": word_matrix.sum(axis=0).A1
})

frequencies = frequencies.nlargest(top_n, "Frequency")
frequencies = frequencies.sort_values("Frequency")

fig = px.bar(
    frequencies,
    x="Frequency",
    y="Word",
    orientation="h",
    title="Top Words in Movie Reviews"
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Preprocessing: lowercase tokenization and English stop-word removal. "
    "Bars show raw word counts, not sentiment or importance."
)

st.markdown("### What Does This Visualization Reveal?")
st.write(
    "This chart shows the most frequently used words in the selected "
    "movie reviews and allows accurate comparisons between words."
)

st.markdown("### How Could It Mislead?")
st.write(
    "High frequency does not necessarily mean high importance. "
    "Removing stop words may also remove meaningful words such as 'not'."
)

st.subheader("Dataset Preview")
st.dataframe(filtered_df.head(10))
