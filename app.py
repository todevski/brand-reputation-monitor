from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import List, Optional

import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# -----------------------------
# Settings
# -----------------------------
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


# -----------------------------
# Helpers
# -----------------------------
def month_labels_2023() -> List[str]:
    # Always show Jan..Dec 2023 on the slider (even if some months have 0 reviews)
    return [datetime(2023, m, 1).strftime("%b %Y") for m in range(1, 13)]


def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_excel_strict(name: str) -> pd.DataFrame:
    """
    Strict loader: reads data/<name>.xlsx using openpyxl.
    Returns empty df if missing.
    """
    path = DATA_DIR / f"{name}.xlsx"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_excel(path, engine="openpyxl")


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a standard datetime column called date_dt if possible.
    """
    if df.empty:
        return df

    date_col = pick_col(df, ["date", "date_iso", "created_at", "createdAt", "timestamp"])
    df = df.copy()
    df["date_dt"] = pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT
    return df


# -----------------------------
# Cached data loaders
# -----------------------------
@st.cache_data
def load_products() -> pd.DataFrame:
    return load_excel_strict("product")


@st.cache_data
def load_reviews() -> pd.DataFrame:
    df = load_excel_strict("review")
    return parse_dates(df)


@st.cache_data
def load_testimonials() -> pd.DataFrame:
    return load_excel_strict("testimonial")


# -----------------------------
# Hugging Face (cached)
# -----------------------------
@st.cache_resource
def get_sentiment_pipe():
    from transformers import pipeline
    return pipeline("sentiment-analysis", model=MODEL_NAME)


@st.cache_data
def run_sentiment(texts: List[str]):
    pipe = get_sentiment_pipe()
    preds = []
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        preds.extend(pipe(texts[i : i + batch_size]))
    return preds


# -----------------------------
# UI Pages
# -----------------------------
def page_products():
    st.subheader("Products")
    df = load_products()
    if df.empty:
        st.error("Products file missing or empty. Expected: data/product.xlsx")
        return
    st.dataframe(df, use_container_width=True, hide_index=True)


def page_testimonials():
    st.subheader("Testimonials")
    df = load_testimonials()
    if df.empty:
        st.error("Testimonials file missing or empty. Expected: data/testimonial.xlsx")
        return
    st.dataframe(df, use_container_width=True, hide_index=True)


def page_reviews():
    st.subheader("Reviews (Core Feature)")

    df = load_reviews()
    if df.empty:
        st.error("Reviews file missing or empty. Expected: data/review.xlsx")
        return

    text_col = pick_col(df, ["text", "review_text", "content", "message"])
    if not text_col:
        st.error("No review text column found. Expected: text / review_text / content / message")
        st.write("Columns found:", list(df.columns))
        return

    if "date_dt" not in df.columns or df["date_dt"].isna().all():
        st.error("No valid date column found/parsed. Make sure review.xlsx has a date column.")
        st.write("Columns found:", list(df.columns))
        return

    # keep only 2023 reviews
    df_2023 = df[df["date_dt"].dt.year == 2023].copy()
    if df_2023.empty:
        st.warning("No reviews from 2023 were found. Check your dates in review.xlsx.")
        return

    df_2023["month_label"] = df_2023["date_dt"].dt.strftime("%b %Y")

    # Slider must show ALL months Jan..Dec (even if no reviews in later months)
    all_months = month_labels_2023()
    selected_month = st.select_slider("Select a month in 2023", options=all_months, value="Jan 2023")

    filtered = df_2023[df_2023["month_label"] == selected_month].copy()

    if filtered.empty:
        st.info(f"No reviews found for **{selected_month}**.")
        return

    st.write(f"Showing **{len(filtered)}** reviews for **{selected_month}**")

    # sentiment
    with st.spinner("Running sentiment analysis (Hugging Face Transformers)..."):
        texts = filtered[text_col].astype(str).tolist()
        preds = run_sentiment(texts)

    filtered["sentiment_label"] = [p["label"] for p in preds]
    filtered["sentiment_score"] = [float(p["score"]) for p in preds]

    # metrics
    avg_conf = float(filtered["sentiment_score"].mean())
    pos_pct = float((filtered["sentiment_label"] == "POSITIVE").mean() * 100)
    neg_pct = float((filtered["sentiment_label"] == "NEGATIVE").mean() * 100)

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg confidence", f"{avg_conf:.3f}")
    c2.metric("Positive %", f"{pos_pct:.1f}%")
    c3.metric("Negative %", f"{neg_pct:.1f}%")

    # bar chart with avg confidence tooltip
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

    # table
    show_cols = [c for c in ["date_dt", text_col, "sentiment_label", "sentiment_score", "rating", "review_id"] if c in filtered.columns]
    st.dataframe(filtered[show_cols], use_container_width=True, hide_index=True)

    # bonus word cloud
    st.subheader("Word Cloud (Bonus)")
    blob = " ".join(filtered[text_col].astype(str).tolist()).strip()
    if blob:
        wc = WordCloud(width=1200, height=450, background_color="white").generate(blob)
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.info("No review text available to generate a word cloud.")


# -----------------------------
# Main app
# -----------------------------
def main():
    st.set_page_config(page_title="Brand Reputation Monitor (2023)", layout="wide")
    st.title("Brand Reputation Monitor (2023)")
    st.caption("Loads local Excel files: product.xlsx, review.xlsx, testimonial.xlsx")

    # Sidebar navigation only (debug removed)
    st.sidebar.markdown("## Navigate")
    section = st.sidebar.radio("Navigate", ["Products", "Testimonials", "Reviews"])

    if section == "Products":
        page_products()
    elif section == "Testimonials":
        page_testimonials()
    else:
        page_reviews()


if __name__ == "__main__":
    main()
