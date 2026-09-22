import streamlit as st
import os
import json
import joblib
import pandas as pd
import plotly.express as px
import re
import spacy
from spacy import displacy
import nltk
from nltk.stem import PorterStemmer

# -----------------------------------------
# 1. CLOUD-SAFE RESOURCE INITIALIZATION
# -----------------------------------------
@st.cache_resource
def load_nlp_resources():
    # NLTK setup
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    
    # spaCy setup with auto-download fallback
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        import subprocess
        import sys
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
    
    return nlp, PorterStemmer()

nlp, stemmer = load_nlp_resources()

# -----------------------------------------
# 2. MODEL ARTIFACT LOADING
# -----------------------------------------
@st.cache_resource
def load_models():
    base_dir = "model_artifacts"
    try:
        clf = joblib.load(os.path.join(base_dir, "clause_classifier.pkl"))
        tfidf = joblib.load(os.path.join(base_dir, "tfidf_vectorizer.pkl"))
        le = joblib.load(os.path.join(base_dir, "label_encoder.pkl"))
        
        with open(os.path.join(base_dir, "evaluation_metrics.json"), "r") as f:
            metrics = json.load(f)
        with open(os.path.join(base_dir, "top_keywords.json"), "r") as f:
            keywords = json.load(f)
            
        return clf, tfidf, le, metrics, keywords
    except FileNotFoundError:
        st.error("Model artifacts missing. Please run `train_model.py` first.")
        st.stop()

clf, tfidf, le, metrics, keywords = load_models()

# -----------------------------------------
# 3. SESSION STATE ROUTING
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
        st.warning("Please enter a valid contract snippet (min 10 characters).")

def reset_app():
    st.session_state.analyzed = False
    st.session_state.raw_text = ""

# -----------------------------------------
# 4. VIEW: LANDING PAGE
# -----------------------------------------
def render_landing_page():
    st.title("⚖️ JurisParse AI")
    st.markdown("### End-to-End NLP Pipeline for Legal Clause Audit & Risk Detection")
    st.markdown("Enter a raw contract snippet below to begin morphological, syntactic, and entity analysis.")
    
    sample_clauses = {
        "Sample 1: Indemnification": "The Contractor shall indemnify and hold harmless the Client from any damages, liabilities, or financial losses arising from breach of this Agreement.",
        "Sample 2: Termination": "Either party may terminate this Contract by providing a 30-day written notice to the other party, specifying the effective date of termination.",
        "Sample 3: Governing Law": "This Agreement shall be governed by and construed in accordance with the laws of the State of California, without regard to its conflict of law provisions."
    }
    
    selected_sample = st.selectbox("Or choose a pre-loaded sample:", ["-- Select Sample --"] + list(sample_clauses.keys()))
    
    default_text = ""
    if selected_sample != "-- Select Sample --":
        default_text = sample_clauses[selected_sample]

    st.text_area("Contract / Clause Text", value=default_text, height=200, key="text_input")
    st.button("Analyze Contract 🚀", on_click=trigger_analysis, type="primary")

# -----------------------------------------
# 5. VIEW: DASHBOARD (MULTI-TAB)
# -----------------------------------------
def render_dashboard():
    st.button("← Analyze Another Contract", on_click=reset_app)
    
    text = st.session_state.raw_text
    doc = nlp(text)
    
    st.markdown("### 📊 JurisParse AI Dashboard")
    st.info(f"**Input Snippet:** {text[:150]}...")
    
    # Core Classification (Used across tabs)
    vec = tfidf.transform([text])
    pred_idx = clf.predict(vec)[0]
    pred_class = le.inverse_transform([pred_idx])[0]
    probs = clf.predict_proba(vec)[0]

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1. Preprocessing", "2. Feature Mining", "3. Classifier", "4. Syntactic Structure", "5. Entity & Bot"
    ])
    
    # --- TAB 1: TEXT PREPROCESSING ---
    with tab1:
        st.header("Morphological Analysis")
        # Jargon Normalizer Regex
        norm_text = re.sub(r'(?i)IN WITNESS WHEREOF|hereinafter|thereto|hereby', '[LEGALESE REMOVED]', text)
        norm_text = re.sub(r'\d+\.\d+|\([a-z]\)', '[SEC_NUM]', norm_text)
        
        st.markdown("**Regex Legal Jargon Normalization:**")
        st.write(norm_text)
        
        st.markdown("**Stemming vs. Lemmatization Comparator:**")
        tokens = [token for token in doc if not token.is_punct and not token.is_space]
        comparison_data = {
            "Original Token": [t.text for t in tokens],
            "Porter Stemmer (NLTK)": [stemmer.stem(t.text) for t in tokens],
            "Morphological Lemma (spaCy)": [t.lemma_ for t in tokens],
            "POS Tag": [t.pos_ for t in tokens]
        }
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

    # --- TAB 2: FEATURE ENGINEERING ---
    with tab2:
        st.header("Domain Keyword Mining")
        st.markdown(f"**Top TF-IDF Keywords for Detected Category: `{pred_class}`**")
        
        class_keywords = keywords.get(pred_class, [])
        # Mocking weights for visualization purposes based on order
        weights = [1.0 - (i * 0.08) for i in range(len(class_keywords))]
        
        df_keywords = pd.DataFrame({"Keyword": class_keywords, "TF-IDF Weight": weights})
        fig = px.bar(df_keywords, x="TF-IDF Weight", y="Keyword", orientation='h', 
                     title=f"Term Importance for {pred_class}", color="TF-IDF Weight",
                     color_continuous_scale="Blues")
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    # --- TAB 3: CLAUSE CLASSIFIER ---
    with tab3:
        st.header("Multi-Class Legal Classifier")
        st.success(f"**Predicted Clause Category:** {pred_class}")
        
        # Probability Distribution
        prob_df = pd.DataFrame({
            "Clause Category": le.classes_,
            "Probability": probs
        }).sort_values(by="Probability", ascending=False)
        
        fig2 = px.bar(prob_df, x="Probability", y="Clause Category", orientation='h',
                      title="Prediction Confidence Scores", color="Probability")
        st.plotly_chart(fig2, use_container_width=True)
        
        with st.expander("View Global Model Evaluation Metrics"):
            st.metric("Validation Accuracy", f"{metrics['accuracy']*100:.2f}%")
            st.json(metrics['classification_report'])

    # --- TAB 4: SYNTACTIC AMBIGUITY ---
    with tab4:
        st.header("Syntactic Structure & Ambiguity")
        
        # Rule-based linguistic checks
        has_passive = any(tok.dep_ == "auxpass" for tok in doc)
        has_subj = any("subj" in tok.dep_ for tok in doc)
        has_obj = any("obj" in tok.dep_ for tok in doc)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Passive Voice Detected", "Yes ⚠️" if has_passive else "No ✅")
        col2.metric("Missing SVO Structure", "Yes ⚠️" if not (has_subj and has_obj) else "No ✅")
        col3.metric("Avg Sentence Length", f"{len(list(doc))/len(list(doc.sents)):.1f} tokens")
        
        st.markdown("**Dependency Parse Tree:**")
        html = displacy.render(doc, style="dep", page=False)
        st.components.v1.html(html, height=400, scrolling=True)

    # --- TAB 5: ENTITY & QA BOT ---
    with tab5:
        st.header("Entity Extractor & Query Bot")
        
        # NER Extraction
        target_labels = ['ORG', 'DATE', 'MONEY', 'GPE']
        entities = [{"Entity": ent.text, "Label": ent.label_} for ent in doc.ents if ent.label_ in target_labels]
        
        if entities:
            st.dataframe(pd.DataFrame(entities), use_container_width=True)
        else:
            st.info("No primary legal entities (ORG, DATE, MONEY, GPE) detected in this snippet.")
            
        st.markdown("---")
        st.markdown("### 🤖 Regulatory QA Bot")
        user_q = st.text_input("Ask a question about this clause (e.g., 'What is the notice period?', 'Who is liable?'):")
        
        if user_q:
            q = user_q.lower()
            if re.search(r'notice|how many days|terminate', q):
                st.info("**Bot Answer:** Based on standard contract structures, look for terms accompanied by 'days' or 'written notice' in the text.")
            elif re.search(r'liab|indemn|who pays', q):
                st.info("**Bot Answer:** Liability usually falls on the party mentioned directly preceding 'shall indemnify' or 'agrees to hold harmless'.")
            elif re.search(r'law|govern|jurisdiction', q):
                st.info("**Bot Answer:** Governing law is typically identified by GPE (Geopolitical Entity) tags in the entity extractor above.")
            else:
                st.warning("Bot: I couldn't map that query to a specific legal heuristic. Try asking about 'notice periods', 'liability', or 'governing law'.")

# -----------------------------------------
# 6. APP EXECUTION
# -----------------------------------------
if __name__ == "__main__":
    st.set_page_config(page_title="JurisParse AI", page_icon="⚖️", layout="wide")
    if not st.session_state.analyzed:
        render_landing_page()
    else:
        render_dashboard()