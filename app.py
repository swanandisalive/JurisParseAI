import streamlit as st
import os
import json
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import re
import spacy
from spacy.pipeline import EntityRuler
import nltk
from nltk.stem import PorterStemmer
import PyPDF2
import docx
from io import BytesIO

# -----------------------------------------
# 1. INITIALIZATION & ARTIFACT LOADING
# -----------------------------------------
@st.cache_resource
def load_nlp_resources():
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    
    try:
        import en_core_web_sm
        nlp = en_core_web_sm.load()
    except Exception:
        nlp = spacy.load("en_core_web_sm")
    
    # IMPROVEMENT: Add Custom Legal Entity Ruler
    if not nlp.has_pipe("entity_ruler"):
        ruler = nlp.add_pipe("entity_ruler", before="ner")
        patterns = [
            {"label": "JURISDICTION", "pattern": [{"LOWER": "the", "OP": "?"}, {"LOWER": "state"}, {"LOWER": "of"}, {"ENT_TYPE": "GPE", "OP": "+"}]}
        ]
        ruler.add_patterns(patterns)
        
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
# 2. SESSION STATE, HELPER FUNCTIONS & ROUTING
# -----------------------------------------
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False
if "raw_text" not in st.session_state:
    st.session_state.raw_text = ""

def trigger_analysis(text_source):
    if len(text_source.strip()) > 10:
        st.session_state.raw_text = text_source
        st.session_state.analyzed = True
    else:
        st.warning("Please provide valid contract text.")

def reset_app():
    st.session_state.analyzed = False
    st.session_state.raw_text = ""

def highlight_text(text, patterns, color, label=""):
    for p in patterns:
        replacement = f'<span style="background-color:{color}; padding:2px 4px; border-radius:4px; font-weight:bold; color:black;">\\1 <span style="font-size:0.7em; color:#333;">[{label}]</span></span>'
        text = re.sub(f'({p})', replacement, text, flags=re.IGNORECASE)
    return text

def extract_file_text(uploaded_file):
    text = ""
    try:
        if uploaded_file.name.endswith(".pdf"):
            reader = PyPDF2.PdfReader(uploaded_file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        elif uploaded_file.name.endswith(".docx"):
            doc = docx.Document(uploaded_file)
            for para in doc.paragraphs:
                text += para.text + "\n"
    except Exception as e:
        st.error(f"Error parsing file: {e}")
    return text

# -----------------------------------------
# 3. VIEWS
# -----------------------------------------
def render_landing_page():
    st.title("⚖️ JurisParse AI")
    st.markdown("### End-to-End NLP Pipeline for Legal Clause Audit & Risk Detection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Upload Document")
        uploaded_file = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])
        if uploaded_file and st.button("Analyze Uploaded File 🚀", type="primary"):
            trigger_analysis(extract_file_text(uploaded_file))
            st.rerun()
            
    with col2:
        st.subheader("2. Or Paste / Use Snippet")
        sample_clauses = {
            "Sample 1: Indemnification": "The Contractor shall indemnify and hold harmless the Client from any damages, liabilities, or financial losses arising from breach of this Agreement.",
            "Sample 2: Termination": "Either party may terminate this Contract by providing a thirty (30) days prior written notice to the other party. In the event of a material breach, the non-breaching Party may terminate immediately.",
            "Sample 3: Governing Law": "This Agreement and any dispute or claim arising out of or in connection with it shall be governed by and construed in accordance with the internal laws of the State of Delaware, without regard to its conflict of laws principles. Any dispute shall be resolved first through good-faith executive negotiations for a period of thirty (30) days. If unresolved, the dispute shall be submitted to final and binding arbitration administered by the American Arbitration Association (AAA) under its Commercial Arbitration Rules, with venue in Wilmington, Delaware, and judgment on the award rendered by the arbitrator(s) may be entered in any court having jurisdiction thereof."
        }
        selected_sample = st.selectbox("Choose a pre-loaded sample:", ["-- Select Sample --"] + list(sample_clauses.keys()))
        default_text = sample_clauses.get(selected_sample, "")
        
        text_input = st.text_area("Contract Text", value=default_text, height=150)
        if st.button("Analyze Snippet 🚀"):
            trigger_analysis(text_input)
            st.rerun()

def render_dashboard():
    st.button("← Analyze Another Contract", on_click=reset_app)
    text = st.session_state.raw_text
    doc = nlp(text)
    
    vec = tfidf.transform([text])
    probs = clf.predict_proba(vec)[0]
    
    threshold = 0.25
    pred_indices = np.where(probs > threshold)[0]
    if len(pred_indices) == 0:
        pred_indices = [np.argmax(probs)]
    detected_classes = mlb.classes_[pred_indices]
    
    st.markdown("### 📊 JurisParse AI Dashboard")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "1. Exec Summary", "2. Vagueness & Risks", "3. Entities", "4. Drafting Strategy"
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

    # --- TAB 2: VAGUENESS & LEGAL RISK ---
    with tab2: 
        st.header("Legal Ambiguity & Vagueness Detector")
        st.info("Highlights undefined legal standards and passive voice obligations.")
        
        ambiguous_patterns = [
            r"material breach", r"reasonable efforts", r"sole discretion", 
            r"at any time", r"without cause", r"best efforts", r"timely manner"
        ]
        
        risk_text = text
        risk_count = 0
        for pattern in ambiguous_patterns:
            if re.search(pattern, risk_text, re.IGNORECASE):
                risk_text = highlight_text(risk_text, [pattern], "#ffd54f", "VAGUE")
                risk_count += 1
                
        # IMPROVEMENT: Highlight Passive Voice dynamically
        passive_count = 0
        for tok in doc:
            if tok.dep_ == "auxpass":
                passive_phrase = f"{tok.text} {tok.head.text}"
                risk_text = highlight_text(risk_text, [re.escape(passive_phrase)], "#ffb74d", "PASSIVE VOICE")
                passive_count += 1
                
        col1, col2 = st.columns(2)
        col1.metric("Ambiguous Terms Detected", risk_count)
        col2.metric("Passive Phrasing Detected", passive_count)
        
        st.markdown("**Vagueness & Passive Audit:**")
        st.markdown(f"<div style='line-height:1.8; font-size:1.1em; padding:15px; border:1px solid #ddd; border-radius:5px;'>{risk_text}</div>", unsafe_allow_html=True)

    # --- TAB 3: ENTITIES ---
    with tab3:
        st.header("Legal Entity Extractor")
        
        dates = re.findall(r'\b(?:[a-zA-Z-]+\s+)?\(\d+\)\s*(?:days|months|years)\b|\b\d+\s*(?:days|months|years)\b', text, re.IGNORECASE)
        entities = [{"Entity": d, "Label": "NOTICE PERIOD"} for d in dates]
        
        # IMPROVEMENT: Target JURISDICTION custom label too
        target_labels = ['ORG', 'MONEY', 'GPE', 'JURISDICTION']
        for ent in doc.ents:
            if ent.label_ in target_labels and ent.text.lower() != "party":
                entities.append({"Entity": ent.text, "Label": ent.label_})
        
        if entities:
            # Deduplicate list of dicts
            entities = [dict(t) for t in {tuple(d.items()) for d in entities}]
            st.dataframe(pd.DataFrame(entities), use_container_width=True)
        else:
            st.info("No primary legal entities detected.")

    # --- TAB 4: DRAFTING STRATEGY ---
    with tab4:
        st.header("Drafting Strategy & Missing Elements")
        st.info("Rule-based checks mapping the ML classification to practical legal requirements.")
        
        primary_class = detected_classes[0]
        st.markdown(f"**Primary Clause Type:** `{primary_class}`")
        
        if primary_class == "Governing Laws":
            st.write("- **Venue / Jurisdiction Choice:**", "✅ Detected" if re.search(r'\b(venue|jurisdiction|courts of)\b', text, re.IGNORECASE) else "❌ Missing (Consider specifying exclusive venue)")
            st.write("- **Jury Trial Waiver:**", "✅ Detected" if re.search(r'\b(jury trial|waiver of jury)\b', text, re.IGNORECASE) else "❌ Missing (Consider adding waiver)")
            st.write("- **Conflict of Laws Exclusion:**", "✅ Detected" if re.search(r'\b(conflict of laws|choice of law)\b', text, re.IGNORECASE) else "❌ Missing (Consider excluding conflict of laws principles)")
        
        elif primary_class == "Termination":
            st.write("- **Cure Period:**", "✅ Detected" if re.search(r'\b(cure|remedy)\b', text, re.IGNORECASE) else "❌ Missing (Consider adding a notice and cure period)")
            st.write("- **Transition Assistance:**", "✅ Detected" if re.search(r'\b(transition|assist)\b', text, re.IGNORECASE) else "❌ Missing (Consider post-termination transition duties)")
        
        elif primary_class == "Indemnification":
            st.write("- **Defense Obligation:**", "✅ Detected" if re.search(r'\b(defend)\b', text, re.IGNORECASE) else "❌ Missing (Indemnify alone may not cover defense costs)")
            st.write("- **Third-Party Claims:**", "✅ Detected" if re.search(r'\b(third-party|third party)\b', text, re.IGNORECASE) else "❌ Missing (Specify application to third-party claims)")
            
        else:
            st.write(f"No specific strategic rules configured for `{primary_class}` clauses yet.")

if __name__ == "__main__":
    st.set_page_config(page_title="JurisParse AI", page_icon="⚖️", layout="wide")
    if not st.session_state.analyzed:
        render_landing_page()
    else:
        render_dashboard()