import json
import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

st.set_page_config(page_title="Customer Feedback Classifier", layout="wide")
st.title("Customer Feedback Classifier")
st.write("Upload a feedback CSV to classify each row into **category** (`delivery`, `quality`, `price`) and **priority** (`High`, `Medium`, `Low`).")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY not found. Please add it to your `.env` file.")
    st.stop()

client = genai.Client(api_key=api_key)

CLASSIFICATION_PROMPT = """
You are a customer support analyst. Classify the following customer feedback into:
1. "category": strictly one of ["delivery", "quality", "price"]
2. "priority": strictly one of ["High", "Medium", "Low"]

Priority rules:
- "High": severe delays/missing orders, defective/unsafe/broken items, billing errors or major pricing overcharges.
- "Medium": noticeable inconvenience, minor defects, or fair complaints about value/discounts.
- "Low": general observations, mild suggestions, or minor comments.

Return ONLY valid JSON with keys "category" and "priority". No markdown or extra text.
Feedback: {feedback_text}
""".strip()


def classify_feedback(text: str) -> dict:
    if not text or not str(text).strip():
        return {"category": "quality", "priority": "Low"}
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=CLASSIFICATION_PROMPT.format(feedback_text=str(text).strip()),
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )
        data = json.loads(response.text)
        category = str(data.get("category", "quality")).lower().strip()
        priority = str(data.get("priority", "Medium")).capitalize().strip()

        if category not in {"delivery", "quality", "price"}:
            category = "quality"
        if priority not in {"High", "Medium", "Low"}:
            priority = "Medium"

        return {"category": category, "priority": priority}
    except Exception:
        return {"category": "quality", "priority": "Medium"}


uploaded_file = st.file_uploader("Upload Feedback CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    text_columns = df.select_dtypes(include=["object", "string"]).columns.tolist()
    default_col_idx = 0
    for idx, col in enumerate(text_columns):
        if any(word in col.lower() for word in ["feedback", "review", "comment", "text", "message"]):
            default_col_idx = idx
            break

    feedback_col = st.selectbox(
        "Select the column containing customer feedback",
        options=text_columns if text_columns else df.columns.tolist(),
        index=default_col_idx if text_columns else 0,
    )

    if st.button("Classify Feedback"):
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_rows = len(df)

        for i, row_val in enumerate(df[feedback_col]):
            status_text.text(f"Classifying row {i + 1} of {total_rows}...")
            results.append(classify_feedback(row_val))
            progress_bar.progress((i + 1) / total_rows)

        status_text.text("Classification complete!")
        classified_df = df.copy()
        classified_df["category"] = [r["category"] for r in results]
        classified_df["priority"] = [r["priority"] for r in results]
        st.session_state["classified_df"] = classified_df

if "classified_df" in st.session_state:
    classified_df = st.session_state["classified_df"]

    st.subheader("Filter & View Results")
    col1, col2 = st.columns(2)
    with col1:
        selected_category = st.selectbox(
            "Filter by Category",
            options=["All", "delivery", "quality", "price"],
        )
    with col2:
        selected_priority = st.selectbox(
            "Filter by Priority",
            options=["All", "High", "Medium", "Low"],
        )

    filtered_df = classified_df.copy()
    if selected_category != "All":
        filtered_df = filtered_df[filtered_df["category"] == selected_category]
    if selected_priority != "All":
        filtered_df = filtered_df[filtered_df["priority"] == selected_priority]

    st.dataframe(filtered_df, use_container_width=True)

    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Filtered Results as CSV",
        data=csv_data,
        file_name="classified_feedback.csv",
        mime="text/csv",
    )