import os
import sys
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from src.preprocess import preprocess_single, combine_features
from src.predict import predict, models_exist, load_results
from src.scraper import scrape_job_text

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake Job Detection",
    page_icon="🔍",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .result-real {
        background: #d4edda; border-left: 6px solid #28a745;
        padding: 1rem 1.5rem; border-radius: 6px; margin: 1rem 0;
    }
    .result-fake {
        background: #f8d7da; border-left: 6px solid #dc3545;
        padding: 1rem 1.5rem; border-radius: 6px; margin: 1rem 0;
    }
    .result-title { font-size: 1.6rem; font-weight: 700; margin-bottom: 0.25rem; }
    .conf-badge {
        display: inline-block; background: rgba(0,0,0,0.08);
        border-radius: 999px; padding: 2px 12px; font-size: 0.9rem;
    }
    .word-chip-fake {
        display: inline-block; background: #f8d7da; color: #721c24;
        border-radius: 999px; padding: 2px 10px; margin: 3px; font-size: 0.85rem;
    }
    .word-chip-real {
        display: inline-block; background: #d4edda; color: #155724;
        border-radius: 999px; padding: 2px 10px; margin: 3px; font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔍 Hybrid AI Fake Job Posting Detector")
st.caption(
    "Powered by SVM + TF-IDF + LIME explainability · Built with Streamlit"
)

# ── Sidebar: Model training ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Model Setup")

    if not models_exist():
        st.warning("No trained model found. Upload the dataset and train first.")
    else:
        st.success("Model is ready ✓")

    uploaded = st.file_uploader(
        "Upload fake_job_postings.csv", type=["csv"], key="csv_upload"
    )

    if uploaded and st.button("🚀 Train Model"):
        import tempfile, pandas as pd
        from src.train import train

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        with st.spinner("Training all models — this may take 1–3 minutes…"):
            try:
                results = train(tmp_path)
                st.success("Training complete!")
            except Exception as e:
                st.error(f"Training failed: {e}")
        os.unlink(tmp_path)

    # Model comparison table
    st.markdown("---")
    st.subheader("📊 Model Comparison")
    results = load_results()
    if results:
        import pandas as pd
        rows = [
            {"Model": k, "Accuracy %": v["accuracy"], "Recall %": v["recall"], "F1 %": v["f1"]}
            for k, v in results.items()
        ]
        st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)
    else:
        st.info("Train a model to see comparison.")

# ── Main: Input ───────────────────────────────────────────────────────────────
tab_text, tab_url = st.tabs(["📝 Paste Job Text", "🌐 Job URL"])

raw_text = ""

with tab_text:
    raw_text_input = st.text_area(
        "Paste the full job posting here",
        height=220,
        placeholder="Job Title, Description, Requirements, Benefits…",
        key="text_input",
    )

with tab_url:
    url_input = st.text_input(
        "Enter job posting URL",
        placeholder="https://www.example.com/jobs/123",
        key="url_input",
    )
    if url_input:
        if st.button("🔗 Fetch Job Text"):
            with st.spinner("Fetching content from URL…"):
                try:
                    fetched = scrape_job_text(url_input)
                    st.session_state["fetched_text"] = fetched
                    st.success(f"Fetched {len(fetched)} characters.")
                    st.text_area("Extracted text (preview)", fetched[:800] + "…", height=180, disabled=True)
                except Exception as e:
                    st.error(f"Could not scrape URL: {e}")

# ── Determine active text ─────────────────────────────────────────────────────
active_text = raw_text_input or st.session_state.get("fetched_text", "")

# ── Analyze button ────────────────────────────────────────────────────────────
st.markdown("---")
analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

if analyze_btn:
    if not models_exist():
        st.error("Please train the model first using the sidebar.")
    elif not active_text.strip():
        st.warning("Please enter a job description or fetch from a URL.")
    else:
        with st.spinner("Analyzing…"):
            clean = preprocess_single(active_text)

            if not clean.strip():
                st.warning("Text was empty after preprocessing. Please provide more content.")
                st.stop()

            result = predict(clean)

        label = result["label"]
        conf = result["confidence"]
        is_fake = result["prediction"] == 1
        css_class = "result-fake" if is_fake else "result-real"
        icon = "❌" if is_fake else "✅"

        st.markdown(
            f"""
            <div class="{css_class}">
                <div class="result-title">{icon} {label} Job Posting</div>
                {"<span class='conf-badge'>Confidence: " + str(conf) + "%</span>" if conf else ""}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── LIME explanation ─────────────────────────────────────────────────
        with st.expander("🧠 Explainable AI — Why this prediction?", expanded=True):
            with st.spinner("Generating LIME explanation…"):
                try:
                    from src.explain import explain
                    exp = explain(clean, num_features=12)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**🚩 Fake indicators**")
                        if exp["fake_indicators"]:
                            chips = "".join(
                                f"<span class='word-chip-fake'>{w} ({s})</span>"
                                for w, s in exp["fake_indicators"][:8]
                            )
                            st.markdown(chips, unsafe_allow_html=True)
                        else:
                            st.write("None detected")

                    with col2:
                        st.markdown("**✅ Real indicators**")
                        if exp["real_indicators"]:
                            chips = "".join(
                                f"<span class='word-chip-real'>{w} ({s})</span>"
                                for w, s in exp["real_indicators"][:8]
                            )
                            st.markdown(chips, unsafe_allow_html=True)
                        else:
                            st.write("None detected")

                    # Full LIME HTML in iframe-style component
                    st.markdown("---")
                    st.markdown("**Full LIME explanation:**")
                    st.components.v1.html(exp["html"], height=300, scrolling=True)

                except Exception as e:
                    st.warning(f"LIME explanation unavailable: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "⚠️ This tool is for informational purposes only. Always verify job postings independently."
)
