from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import altair as alt

DATA_DIR = Path("data")
CSV_PATH = DATA_DIR / "webscraping_dev_data.csv"
PRODUCTS_JSON = DATA_DIR / "products.json"
REVIEWS_JSON = DATA_DIR / "reviews.json"
TESTIMONIALS_JSON = DATA_DIR / "testimonials.json"

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


# -----------------------
# Loaders
# -----------------------
@st.cache_data
def load_csv() -> pd.DataFrame:
    if not CSV_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(CSV_PATH)


@st.cache_data
def load_json(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_json(path)


@st.cache_resource
def get_sentiment_pipe():
    from transformers import pipeline
    return pipeline("sentiment-analysis", model=MODEL_NAME)


def month_labels_2023() -> List[str]:
    return [datetime(2023, m, 1).strftime("%b %Y") for m in range(1, 13)]


def parse_date_column(df: pd.DataFrame) -> pd.Series:
    """
    Create a unified datetime column even if the file uses different names.
    """
    if "date_iso" in df.columns:
        return pd.to_datetime(df["date_iso"], errors="coerce")
    if "date" in df.columns:
        return pd.to_datetime(df["date"], errors="coerce")
    return pd.to_datetime(pd.Series([None] * len(df)), errors="coerce")


def find_type_column(df: pd.DataFrame) -> Optional[str]:
    """
    Your CSV might not have 'type'. We try common alternatives.
    """
    candidates = ["type", "section", "category", "source", "dataset"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def get_text_col(df: pd.DataFrame) -> str:
    for c in ["text", "review_text", "content", "message"]:
        if c in df.columns:
            return c
    # fallback
    df["text"] = ""
    return "text"


# -----------------------
# App Pages
# -----------------------
def show_products(df_csv: pd.DataFrame):
    st.subheader("Products")

    type_col = find_type_column(df_csv)
    if type_col:
        products = df_csv[df_csv[type_col].astype(str).str.lower().eq("product")].copy()
        if not products.empty:
            st.dataframe(products, use_container_width=True, hide_index=True)
            return

    # fallback: load JSON
    products = load_json(PRODUCTS_JSON)
    if products.empty:
        st.error(f"Could not find Products in CSV (no '{type_col}' match) and missing {PRODUCTS_JSON}.")
        return
    st.dataframe(products, use_container_width=True, hide_index=True)


def show_testimonials(df_csv: pd.DataFrame):
    st.subheader("Testimonials")

    type_col = find_type_column(df_csv)
    if type_col:
        t = df_csv[df_csv[type_col].astype(str).str.lower().eq("testimonial")].copy()
        if not t.empty:
            st.dataframe(t, use_container_width=True, hide_index=True)
            return

    t = load_json(TESTIMONIALS_JSON)
    if t.empty:
        st.error(f"Could not find Testimonials in CSV and missing {TESTIMONIALS_JSON}.")
        return
    st.dataframe(t, use_container_width=True, hide_index=True)


def show_reviews(df_csv: pd.DataFrame):
    st.subheader("Reviews (Core Feature)")

    # Load reviews (CSV if possible, else JSON)
    type_col = find_type_column(df_csv)
    reviews = pd.DataFrame()

    if type_col:
        reviews = df_csv[df_csv[type_col].astype(str).str.lower().eq("review")].copy()

    if reviews.empty:
        reviews = load_json(REVIEWS_JSON).copy()

    if reviews.empty:
        st.error("No reviews found (neither in CSV nor reviews.json).")
        return

    # Parse date and filter to 2023
    reviews["date_dt"] = parse_date_column(reviews)
    reviews_2023 = reviews[reviews["date_dt"].dt.year == 2023].copy()

    if reviews_2023.empty:
        st.warning("No reviews with a 2023 date. (Check scraping date extraction.)")
        st.dataframe(reviews.head(20), use_container_width=True, hide_index=True)
        return

    reviews_2023["month_label"] = reviews_2023["date_dt"].dt.strftime("%b %Y")

    # Month slider (rubric requirement)
    allowed_months = month_labels_2023()
    existing = [m for m in allowed_months if m in set(reviews_2023["month_label"])]
    if not existing:
        existing = sorted(
            reviews_2023["month_label"].unique(),
            key=lambda x: pd.to_datetime(x, format="%b %Y")
        )

    selected = st.select_slider("Select a month in 2023", options=existing)

    filtered = reviews_2023[reviews_2023["month_label"] == selected].copy()
    st.write(f"Showing **{len(filtered)}** reviews for **{selected}**")

    text_col = get_text_col(filtered)
    filtered[text_col] = filtered[text_col].astype(str)

    # NLP (rubric requirement): apply dynamically to filtered text
    with st.spinner("Running Hugging Face sentiment model..."):
        pipe = get_sentiment_pipe()
        preds = pipe(filtered[text_col].tolist())

    filtered["sentiment_label"] = [p["label"] for p in preds]
    filtered["sentiment_score"] = [float(p["score"]) for p in preds]

    # Confidence metrics (rubric requirement)
    st.metric("Avg confidence", f"{filtered['sentiment_score'].mean():.3f}")
    st.metric("Positive %", f"{(filtered['sentiment_label'].eq('POSITIVE').mean() * 100):.1f}%")
    st.metric("Negative %", f"{(filtered['sentiment_label'].eq('NEGATIVE').mean() * 100):.1f}%")

    # Visualization: bar chart (rubric requirement)
    chart_df = (
        filtered.groupby("sentiment_label", as_index=False)
        .agg(count=("sentiment_label", "size"), avg_conf=("sentiment_score", "mean"))
    )

    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("sentiment_label:N", title="Sentiment"),
            y=alt.Y("count:Q", title="Count"),
            tooltip=[
                alt.Tooltip("sentiment_label:N", title="Sentiment"),
                alt.Tooltip("count:Q", title="Count"),
                alt.Tooltip("avg_conf:Q", title="Avg confidence", format=".3f"),
            ],
        )
    )
    st.altair_chart(chart, use_container_width=True)

    # Table
    show_cols = [c for c in ["date_dt", text_col, "sentiment_label", "sentiment_score", "rating", "review_id"] if c in filtered.columns]
    st.dataframe(filtered[show_cols], use_container_width=True, hide_index=True)

    # BONUS: word cloud
    st.subheader("Word Cloud (Bonus)")
    blob = " ".join(filtered[text_col].tolist()).strip()
    if blob:
        wc = WordCloud(width=1200, height=450, background_color="white").generate(blob)
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.info("No text available for word cloud.")


# -----------------------
# Main
# -----------------------
def main():
    st.set_page_config(page_title="Brand Reputation Monitor (2023)", layout="wide")
    st.title("Brand Reputation Monitor (2023)")
    st.caption("Scraped from web-scraping.dev — Products, Reviews, Testimonials")

    df_csv = load_csv()
    if df_csv.empty:
        st.warning("CSV not found or empty. The app will use JSON files in /data.")

    section = st.sidebar.radio("Navigate", ["Products", "Testimonials", "Reviews"])
    st.sidebar.info("Loads locally from /data (fast, no re-scraping).")

    if section == "Products":
        show_products(df_csv)
    elif section == "Testimonials":
        show_testimonials(df_csv)
    else:
        show_reviews(df_csv)


if __name__ == "__main__":
    main()
