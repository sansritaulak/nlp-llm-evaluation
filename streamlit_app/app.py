"""
Simple Streamlit demo for sentiment analysis
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Page config
st.set_page_config(
    page_title="Sentiment Analysis Demo",
    page_icon="🎬",
    layout="centered"
)

# Title
st.title("🎬 Movie Review Sentiment Analysis")
st.markdown("Analyze movie reviews with ML models")
st.markdown("---")

# Model loading functions
@st.cache_resource
def load_baseline():
    """Load baseline model"""
    try:
        from src.models.baseline import BaselineClassifier
        model = BaselineClassifier.load(Path("C:/week4-nlp-llms/outputs/models/baseline_logistic"))
        return model
    except Exception as e:
        st.error(f"Error loading baseline: {e}")
        return None

@st.cache_resource
def load_lora():
    """Load LoRA model"""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from peft import PeftModel
        
        base = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", 
            num_labels=2
        )
        model = PeftModel.from_pretrained(base, "outputs/models/lora_finetuned")
        tokenizer = AutoTokenizer.from_pretrained("outputs/models/lora_finetuned")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Error loading LoRA: {e}")
        return None, None, None

# Sidebar - Model selection
st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.radio(
    "Select Model:",
    ["Baseline (Fast - 88.12%)", "LoRA (Accurate - 91.34%)"]
)

# Load selected model
if "Baseline" in model_choice:
    with st.spinner("Loading Baseline model..."):
        model = load_baseline()
    model_type = "baseline"
else:
    with st.spinner("Loading LoRA model..."):
        lora_model, lora_tokenizer, device = load_lora()
    model_type = "lora"

# Main interface
st.subheader("Enter a Movie Review")

# Example reviews
examples = {
    "Select an example...": "",
    "Positive Example": "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout.",
    "Negative Example": "Terrible waste of time. The plot made no sense and the acting was wooden.",
    "Mixed Example": "Great acting but the plot was terrible and confusing."
}

selected_example = st.selectbox("Or try an example:", list(examples.keys()))

# Text input
if selected_example != "Select an example...":
    review_text = st.text_area(
        "Review text:",
        value=examples[selected_example],
        height=150
    )
else:
    review_text = st.text_area(
        "Review text:",
        placeholder="Type or paste a movie review here...",
        height=150
    )

# Analyze button
if st.button("🚀 Analyze Sentiment", type="primary"):
    if not review_text:
        st.warning("⚠️ Please enter a review!")
    else:
        with st.spinner("Analyzing..."):
            try:
                if model_type == "baseline":
                    # Baseline prediction
                    if model is None:
                        st.error("Model not loaded!")
                    else:
                        prediction = model.predict(pd.Series([review_text]))[0]
                        proba = model.predict_proba(pd.Series([review_text]))[0]
                        confidence = max(proba)
                else:
                    # LoRA prediction
                    if lora_model is None:
                        st.error("Model not loaded!")
                    else:
                        inputs = lora_tokenizer(
                            review_text,
                            return_tensors="pt",
                            truncation=True,
                            max_length=512
                        ).to(device)
                        
                        with torch.no_grad():
                            outputs = lora_model(**inputs)
                        
                        probs = torch.softmax(outputs.logits, dim=1)[0]
                        prediction = torch.argmax(probs).item()
                        confidence = probs[prediction].item()
                
                # Display result
                sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
                color = "green" if prediction == 1 else "red"
                
                st.markdown("### Result:")
                st.markdown(f"## :{color}[{sentiment}]")
                st.metric("Confidence", f"{confidence:.1%}")
                
                # Show probabilities
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                if model_type == "baseline":
                    col1.metric("Negative", f"{proba[0]:.1%}")
                    col2.metric("Positive", f"{proba[1]:.1%}")
                else:
                    col1.metric("Negative", f"{probs[0]:.1%}")
                    col2.metric("Positive", f"{probs[1]:.1%}")
                
            except Exception as e:
                st.error(f"Error during prediction: {e}")

# Info section
st.markdown("---")
st.markdown("### 📊 Model Information")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Baseline Model**")
    st.write("• Accuracy: 88.12%")
    st.write("• Speed: ~1ms")
    st.write("• Type: TF-IDF + LogReg")

with col2:
    st.markdown("**LoRA Model**")
    st.write("• Accuracy: 91.34%")
    st.write("• Speed: ~50ms")
    st.write("• Type: DistilBERT + LoRA")

# Footer
st.markdown("---")
st.caption("Week 4 NLP & LLMs Project | Built with Streamlit")