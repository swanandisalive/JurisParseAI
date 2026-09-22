import streamlit as st
import spacy

# 1. Page Configuration (Must be the first Streamlit command)
st.set_page_config(
    page_title="Natural Language Processing Lab",
    page_icon="📝",
    layout="wide"
)

# 2. Cached Resource for Loading the SpaCy Model
@st.cache_resource
def load_spacy_model():
    """Load the spaCy English language model.
    Using st.cache_resource ensures it's only loaded into memory once.
    """
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        # Fallback handling if the wheel wasn't pre-installed in the environment
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")

# Load model with a visual spinner
with st.spinner("Initializing NLP Model... Please wait."):
    nlp = load_spacy_model()

# 3. App Header and Description
st.title("📝 Natural Language Processing Lab")
st.markdown(
    "Welcome to your NLP toolkit! Enter any text below to extract "
    "linguistic features, inspect tokens, and explore named entities using spaCy."
)

# 4. User Input Section
st.subheader("Input Text")
default_text = (
    "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne "
    "in April 1976 to develop and sell personal computers in Cupertino, California."
)
user_text = st.text_area("Type or paste your text here:", default_text, height=150)

# 5. Analysis Logic
if st.button("Run Analysis", type="primary"):
    if user_text.strip():
        # Process the text with spaCy
        doc = nlp(user_text)
        
        # Display Metrics Overview
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Tokens", len(doc))
        with col2:
            st.metric("Sentences", len(list(doc.sents)))
        with col3:
            st.metric("Named Entities", len(doc.ents))
            
        st.divider()
        
        # Tabs for detailed breakdown
        tab1, tab2, tab3 = st.tabs(["🏷️ Named Entities", "🔍 Tokens & POS", "🌳 Dependencies"])
        
        with tab1:
            st.subheader("Named Entity Recognition (NER)")
            if doc.ents:
                entity_data = [
                    {"Entity Text": ent.text, "Label": ent.label_, "Description": spacy.explain(ent.label_)}
                    for ent in doc.ents
                ]
                st.dataframe(entity_data, use_container_width=True)
            else:
                st.info("No named entities detected in the text.")
                
        with tab2:
            st.subheader("Token Breakdown")
            token_data = [
                {
                    "Token": token.text,
                    "Lemma": token.lemma_,
                    "POS": token.pos_,
                    "Tag": token.tag_,
                    "Stopword": token.is_stop
                }
                for token in doc
            ]
            st.dataframe(token_data, use_container_width=True)
            
        with tab3:
            st.subheader("Syntactic Dependencies")
            dep_data = [
                {
                    "Token": token.text,
                    "Dependency (Dep)": token.dep_,
                    "Head Token": token.head.text,
                    "Head POS": token.head.pos_
                }
                for token in doc
            ]
            st.dataframe(dep_data, use_container_width=True)
            
    else:
        st.warning("Please provide some text to analyze.")

# Footer info
st.markdown("---")
st.caption("Powered by Streamlit, spaCy, and Python.")