import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from pathlib import Path
from datetime import datetime

# -----------------------------
# App config
# -----------------------------
st.set_page_config(
    page_title="Brand Reputation Monitor (2023)",
    page_icon="📊",
    layout="wide",
)

DATA_DIR = Path(__file__).parent / "data"

PRODUCT_FILE = DATA_DIR / "product.xlsx"
TESTIMONIAL_FILE = DATA_DIR / "testimonial.xlsx"
REVIEWS_SCORED_FILE = DATA_DIR / "review_scored.xlsx"


# -----------------------------
# Helpers
# -----------------------------
def month_range_2023():
    return pd.date_range("2023-01-01", "2023-12-01", freq="MS")


def ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def safe_read_excel(path: Path) -> pd.DataFrame:
    # engine fixed to avoid the "format cannot be determined" issue
    return pd.read_excel(path, engine="openpyxl")


@st.cache_data(show_spinner=False)
def load_products() -> pd.DataFrame:
    if not PRODUCT_FILE.exists():
        return pd.DataFrame()
    df = safe_read_excel(PRODUCT_FILE)
    return df


@st.cache_data(show_spinner=False)
def load_testimonials() -> pd.DataFrame:
    if not TESTIMONIAL_FILE.exists():
        return pd.DataFrame()
    df = safe_read_excel(TESTIMONIAL_FILE)
    df = ensure_datetime(df, "date")
    return df


@st.cache_data(show_spinner=False)
def load_reviews_scored() -> pd.DataFrame:
    if not REVIEWS_SCORED_FILE.exists():
        return pd.DataFrame()

    df = safe_read_excel(REVIEWS_SCORED_FILE)

    # Typical columns expected:
    # text, date, rating, username, sentiment, sentiment_score (or sentiment_label, sentiment_score)
    df = ensure_datetime(df, "date")

    # Normalize sentiment column names if needed
    if "sentiment" not in df.columns and "sentiment_label" in df.columns:
        # map POSITIVE/NEGATIVE -> Positive/Negative
        df["sentiment"] = df["sentiment_label"].map({"POSITIVE": "Positive", "NEGATIVE": "Negative"})

    # If sentiment is still missing, create empty columns so app doesn't crash
    if "sentiment" not in df.columns:
        df["sentiment"] = None
    if "sentiment_score" not in df.columns:
        df["sentiment_score"] = None

    return df


def render_wordcloud(texts):
    combined = " ".join([t for t in texts if isinstance(t, str) and t.strip()])
    if not combined.strip():
        st.info("No text available for word cloud in this month.")
        return

    wc = WordCloud(width=1200, height=500, background_color="white").generate(combined)
    fig = plt.figure(figsize=(12, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(fig, use_container_width=True)


# -----------------------------
# Pages
# -----------------------------
def page_products():
    st.header("Products")
    df = load_products()

    if df.empty:
        st.error("Products file not found or empty. Expected: data/product.xlsx")
        return

    st.dataframe(df, use_container_width=True)


def page_testimonials():
    st.header("Testimonials")
    df = load_testimonials()

    if df.empty:
        st.error("Testimonials file not found or empty. Expected: data/testimonial.xlsx")
        return

    st.dataframe(df, use_container_width=True)


def page_reviews():
    st.header("Reviews (Core Feature)")
    df = load_reviews_scored()

    if df.empty:
        st.error("Scored reviews file not found or empty. Expected: data/review_scored.xlsx")
        st.caption("Tip: Run precompute_sentiment.py locally to create review_scored.xlsx.")
        return

    # Make sure we have date + text
    if "date" not in df.columns:
        st.error("Your reviews file must have a 'date' column.")
        st.write("Columns found:", list(df.columns))
        return
    if "text" not in df.columns:
        st.error("Your reviews file must have a 'text' column.")
        st.write("Columns found:", list(df.columns))
        return

    months = month_range_2023()

    # Slider shows Jan-Dec even if no data later
    month_labels = [d.strftime("%b %Y") for d in months]
    selected_label = st.select_slider("Select a month in 2023", options=month_labels, value="Jan 2023")
    selected_month = datetime.strptime(selected_label, "%b %Y")

    start = pd.Timestamp(selected_month.year, selected_month.month, 1)
    end = start + pd.offsets.MonthBegin(1)

    filtered = df[(df["date"] >= start) & (df["date"] < end)].copy()

    st.subheader(f"Showing {len(filtered)} reviews for {selected_label}")

    # If sentiment columns exist, compute metrics + chart
    # Clean sentiment values to exactly Positive/Negative
    filtered["sentiment"] = filtered["sentiment"].astype(str).str.strip()
    filtered.loc[~filtered["sentiment"].isin(["Positive", "Negative"]), "sentiment"] = None

    pos_count = int((filtered["sentiment"] == "Positive").sum())
    neg_count = int((filtered["sentiment"] == "Negative").sum())
    total = max(pos_count + neg_count, 1)

    avg_conf = pd.to_numeric(filtered["sentiment_score"], errors="coerce").dropna()
    avg_conf_value = float(avg_conf.mean()) if len(avg_conf) else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg confidence", f"{avg_conf_value:.3f}")
    c2.metric("Positive %", f"{(pos_count/total)*100:.1f}%")
    c3.metric("Negative %", f"{(neg_count/total)*100:.1f}%")

    chart_df = pd.DataFrame(
        {"Sentiment": ["Positive", "Negative"], "Count": [pos_count, neg_count]}
    )

    chart = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("Sentiment:N", sort=None),
            y=alt.Y("Count:Q"),
            tooltip=["Sentiment", "Count"],
        )
        .properties(height=320)
    )

    st.altair_chart(chart, use_container_width=True)

    # Show table
    with st.expander("Show filtered reviews table"):
        st.dataframe(filtered, use_container_width=True)

    st.subheader("Word Cloud (Bonus)")
    render_wordcloud(filtered["text"].fillna("").astype(str).tolist())


# -----------------------------
# Main
# -----------------------------
def main():
    st.title("Brand Reputation Monitor (2023)")
    st.caption("Loads local Excel files from /data: product.xlsx, review_scored.xlsx, testimonial.xlsx")

    # Sidebar navigation
    st.sidebar.header("Navigate")
    page = st.sidebar.radio("Navigate", ["Products", "Testimonials", "Reviews"], index=0)

    if page == "Products":
        page_products()
    elif page == "Testimonials":
        page_testimonials()
    else:
        page_reviews()


if __name__ == "__main__":
    main()
