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
import re

filtered_df = filtered_df.copy()
filtered_df["review"] = filtered_df["review"].apply(
    lambda text: re.sub(r"<[^>]+>", " ", text)
)
vectorizer = CountVectorizer(
    stop_words="english",
    max_features=1000,
    token_pattern=r"(?u)\b[a-zA-Z]{2,}\b"
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

from sklearn.decomposition import LatentDirichletAllocation

st.divider()
st.header("Topic Modeling Explorer")

st.write(
    "Discover hidden topics in IMDb movie reviews using "
    "Latent Dirichlet Allocation (LDA)."
)

@st.cache_resource
def train_topic_model(reviews):
    topic_vectorizer = CountVectorizer(
        stop_words="english",
        max_features=1000,
        min_df=2,
        token_pattern=r"(?u)\b[a-zA-Z]{2,}\b"
    )

    topic_matrix = topic_vectorizer.fit_transform(reviews)

    lda = LatentDirichletAllocation(
        n_components=5,
        random_state=42,
        max_iter=10
    )

    lda.fit(topic_matrix)

    return lda, topic_vectorizer

lda_model, topic_vectorizer = train_topic_model(
    tuple(df["review"].str.replace(
        r"<[^>]+>", " ", regex=True
    ))
)

selected_topic = st.selectbox(
    "Select Topic",
    [f"Topic {i + 1}" for i in range(5)]
)

topic_index = int(selected_topic.split()[-1]) - 1

words = topic_vectorizer.get_feature_names_out()
topic_weights = lda_model.components_[topic_index]

top_indices = topic_weights.argsort()[-10:][::-1]

topic_df = pd.DataFrame({
    "Word": words[top_indices],
    "Weight": topic_weights[top_indices]
}).sort_values("Weight")

topic_fig = px.bar(
    topic_df,
    x="Weight",
    y="Word",
    orientation="h",
    title=f"Top Words in {selected_topic}"
)

st.plotly_chart(topic_fig, use_container_width=True)

st.markdown("### What Does This Visualization Reveal?")
st.write(
    "The topic model identifies groups of words that frequently "
    "appear together in movie reviews, revealing recurring themes."
)

st.markdown("### How Could It Mislead?")
st.write(
    "LDA topics are statistical patterns rather than fixed meanings. "
    "The number of topics is chosen in advance, and common words "
    "may appear in multiple topics."
)

st.caption(
    "Method: LDA with 5 topics, CountVectorizer, "
    "English stop-word removal, and random_state=42."
)

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

st.divider()
st.header("Raw Word Counts vs. TF-IDF")

st.write(
    "Compare frequently used words with words that are "
    "more distinctive across IMDb movie reviews."
)

@st.cache_data
def calculate_tfidf(reviews):
    tfidf_vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=1000,
        token_pattern=r"(?u)\b[a-zA-Z]{2,}\b"
    )

    tfidf_matrix = tfidf_vectorizer.fit_transform(reviews)

    return pd.DataFrame({
        "Word": tfidf_vectorizer.get_feature_names_out(),
        "TF-IDF": np.asarray(
            tfidf_matrix.mean(axis=0)
        ).ravel()
    })

tfidf_df = calculate_tfidf(
    tuple(filtered_df["review"])
)

comparison_df = frequencies.merge(
    tfidf_df,
    on="Word",
    how="inner"
)

comparison_df = comparison_df.rename(
    columns={"Frequency": "Raw Count"}
)

comparison_df = comparison_df.sort_values(
    "Raw Count",
    ascending=False
).head(15)

st.subheader("Comparison of Top 15 Words")

metric = st.radio(
    "Choose Metric",
    ["Raw Count", "TF-IDF"],
    horizontal=True
)

chart_df = comparison_df.sort_values(metric)

fig_comparison = px.bar(
    chart_df,
    x=metric,
    y="Word",
    orientation="h",
    title=f"Top 15 Words by {metric}"
)

st.plotly_chart(
    fig_comparison,
    use_container_width=True
)

st.markdown("### What Does This Visualization Reveal?")
st.write(
    "Raw counts show how often words appear, while TF-IDF "
    "reduces the influence of words that appear across many "
    "reviews and highlights more distinctive terms."
)

st.markdown("### How Could It Mislead?")
st.write(
    "TF-IDF does not measure sentiment or true importance. "
    "Also, this comparison uses the top 15 words selected "
    "by raw frequency, so distinctive but less frequent "
    "words may be excluded."
)

st.caption(
    "TF-IDF scores are averaged across reviews. "
    "Raw counts and TF-IDF scores use different scales."
)
