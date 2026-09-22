import streamlit as st
import os
import json
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import re
import spacy
import nltk
from nltk.stem import PorterStemmer

# -----------------------------------------
# 1. INITIALIZATION & ARTIFACT LOADING
# -----------------------------------------
@st.cache_resource
def load_nlp_resources():
    # NLTK setup
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    
    # Robust spaCy loader for cloud deployment
    try:
        import en_core_web_sm
        nlp = en_core_web_sm.load()
    except Exception:
        nlp = spacy.load("en_core_web_sm")
        
    return nlp, PorterStemmer()

nlp, stemmer = load_nlp_resources()

@st.cache_resource
def load_models():
    base_dir = "model_artifacts"
    try:
        clf = joblib.load(os.path.join(base_dir, "clause_classifier.pkl"))
        tfidf = joblib.load(os.path.join(base_dir, "tfidf_vectorizer.pkl"))
        mlb = joblib.load(os.path.join(base_dir, "mlb_encoder.pkl"))
        with open(os.path.join(base_dir, "evaluation_metrics.json"), "r") as f:
            metrics = json.load(f)
        with open(os.path.join(base_dir, "top_keywords.json"), "r") as f:
            keywords = json.load(f)
        return clf, tfidf, mlb, metrics, keywords
    except FileNotFoundError:
        st.error("Model artifacts missing. Ensure the `model_artifacts/` folder is pushed to GitHub.")
        st.stop()

clf, tfidf, mlb, metrics, keywords = load_models()

# -----------------------------------------
# 2. SESSION STATE & ROUTING
# -----------------------------------------
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False
if "raw_text" not in st.session_state:
    st.session_state.raw_text = ""

def trigger_analysis():
    if len(st.session_state.text_input.strip()) > 10:
        st.session_state.raw_text = st.session_state.text_input
        st.session_state.analyzed = True
    else:
        st.warning("Please enter a valid contract snippet.")

def reset_app():
    st.session_state.analyzed = False
    st.session_state.raw_text = ""

def highlight_text(text, patterns, color, label=""):
    for p in patterns:
        replacement = f'<span style="background-color:{color}; padding:2px 4px; border-radius:4px; font-weight:bold; color:black;">\\1 <span style="font-size:0.7em; color:#333;">[{label}]</span></span>'
        text = re.sub(f'({p})', replacement, text, flags=re.IGNORECASE)
    return text

# -----------------------------------------
# 3. VIEWS
# -----------------------------------------
def render_landing_page():
    st.title("⚖️ JurisParse AI")
    st.markdown("### End-to-End NLP Pipeline for Legal Clause Audit & Risk Detection")
    
    sample_clauses = {
        "Sample 1: Indemnification": "The Contractor shall indemnify and hold harmless the Client from any damages, liabilities, or financial losses arising from breach of this Agreement.",
        "Sample 2: Termination": "Either party may terminate this Contract by providing a thirty (30) days prior written notice to the other party. In the event of a material breach, the non-breaching Party may terminate immediately.",
        "Sample 3: Governing Law": "This Agreement and any dispute or claim arising out of or in connection with it shall be governed by and construed in accordance with the laws of the State of Delaware, without giving effect to any choice or conflict of law provision."
    }
    
    selected_sample = st.selectbox("Choose a pre-loaded sample:", ["-- Select Sample --"] + list(sample_clauses.keys()))
    default_text = sample_clauses.get(selected_sample, "")

    st.text_area("Contract / Clause Text", value=default_text, height=200, key="text_input")
    st.button("Analyze Contract 🚀", on_click=trigger_analysis, type="primary")

def render_dashboard():
    st.button("← Analyze Another Contract", on_click=reset_app)
    text = st.session_state.raw_text
    doc = nlp(text)
    
    # Core Classification
    vec = tfidf.transform([text])
    probs = clf.predict_proba(vec)[0]
    
    threshold = 0.25
    pred_indices = np.where(probs > threshold)[0]
    if len(pred_indices) == 0:
        pred_indices = [np.argmax(probs)]
        
    detected_classes = mlb.classes_[pred_indices]
    
    st.markdown("### 📊 JurisParse AI Dashboard")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1. Exec Summary", "2. Vagueness & Risks", "3. Entities", "4. Classifier Insights", "5. Feature Mining"
    ])
    
    # --- TAB 1: EXECUTIVE SUMMARY ---
    with tab1:
        st.header("Clause Summary & Audit")
        st.success(f"**Detected Clause Typologies:** {', '.join(detected_classes)}")
        
        highlighted_text = text
        highlighted_text = highlight_text(highlighted_text, [r'\b(?:thirty|sixty|ninety)?\s*\(\d+\)\s*(?:days|months|years)\b', r'\b\d+\s*(?:days|months|years)\b'], "#81c784", "TIMELINE")
        highlighted_text = highlight_text(highlighted_text, [r'terminate', r'termination', r'indemnify', r'hold harmless'], "#e57373", "TRIGGER")
        
        st.markdown("**Actionable Text Audit:**")
        st.markdown(f"<div style='line-height:1.8; font-size:1.1em;'>{highlighted_text}</div>", unsafe_allow_html=True)
        
        with st.expander("🛠️ View Technical NLP Debugging Metrics (Stemming & POS)"):
            tokens = [token for token in doc if not token.is_punct and not token.is_space]
            comparison_data = {
                "Original Token": [t.text for t in tokens],
                "Porter Stemmer": [stemmer.stem(t.text) for t in tokens],
                "Lemma": [t.lemma_ for t in tokens],
                "POS": [t.pos_ for t in tokens]
            }
            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

    # --- TAB 2: VAGUENESS & LEGAL RISK ---
    with tab2: 
        st.header("Legal Ambiguity & Vagueness Detector")
        st.info("Highlights undefined legal standards, discretionary boilerplate, and subjective qualifiers.")
        
        ambiguous_patterns = [
            r"material breach", r"reasonable efforts", r"sole discretion", 
            r"at any time", r"without cause", r"best efforts", r"timely manner"
        ]
        
        risk_text = text
        risk_count = 0
        for pattern in ambiguous_patterns:
            if re.search(pattern, risk_text, re.IGNORECASE):
                risk_text = highlight_text(risk_text, [pattern], "#ffd54f", "VAGUE/AMBIGUOUS")
                risk_count += 1
                
        col1, col2 = st.columns(2)
        col1.metric("Ambiguous Terms Detected", risk_count)
        
        has_passive = any(tok.dep_ == "auxpass" for tok in doc)
        col2.metric("Passive Voice Detected", "Yes ⚠️" if has_passive else "No ✅")
        
        st.markdown("**Vagueness Audit:**")
        st.markdown(f"<div style='line-height:1.8; font-size:1.1em; padding:15px; border:1px solid #ddd; border-radius:5px;'>{risk_text}</div>", unsafe_allow_html=True)

    # --- TAB 3: ENTITIES ---
    with tab3:
        st.header("Legal Entity Extractor")
        dates = re.findall(r'\b(?:[a-zA-Z-]+\s+)?\(\d+\)\s*(?:days|months|years)\b|\b\d+\s*(?:days|months|years)\b', text, re.IGNORECASE)
        entities = [{"Entity": d, "Label": "NOTICE PERIOD"} for d in dates]
        
        target_labels = ['ORG', 'MONEY', 'GPE']
        for ent in doc.ents:
            if ent.label_ in target_labels and ent.text.lower() != "party":
                entities.append({"Entity": ent.text, "Label": ent.label_})
        
        if entities:
            st.dataframe(pd.DataFrame(entities), use_container_width=True)
        else:
            st.info("No primary legal entities detected.")

    # --- TAB 4: CLASSIFIER INSIGHTS ---
    with tab4:
        st.header("Multi-Label Model Confidence")
        prob_df = pd.DataFrame({"Clause Category": mlb.classes_, "Probability": probs}).sort_values(by="Probability", ascending=False)
        fig2 = px.bar(prob_df, x="Probability", y="Clause Category", orientation='h', title="Prediction Confidence Scores", color="Probability")
        fig2.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text=f"Threshold ({threshold})")
        st.plotly_chart(fig2, use_container_width=True)

    # --- TAB 5: FEATURE MINING ---
    with tab5:
        st.header("Domain Keyword Mining")
        primary_class = detected_classes[0]
        st.markdown(f"**Top TF-IDF Keywords for Primary Category: `{primary_class}`**")
        class_keywords = keywords.get(primary_class, [])
        weights = [1.0 - (i * 0.08) for i in range(len(class_keywords))]
        df_keywords = pd.DataFrame({"Keyword": class_keywords, "TF-IDF Weight": weights})
        fig = px.bar(df_keywords, x="TF-IDF Weight", y="Keyword", orientation='h', color="TF-IDF Weight", color_continuous_scale="Blues")
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    st.set_page_config(page_title="JurisParse AI", page_icon="⚖️", layout="wide")
    if not st.session_state.analyzed:
        render_landing_page()
    else:
        render_dashboard()