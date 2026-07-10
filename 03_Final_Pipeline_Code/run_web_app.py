import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time

# Import our modularized pipeline components
from module_vision.image_enhancer import ImageEnhancer
from module_vision.disease_classifier import DiseaseClassifier
from module_llm.trust_validator_rag import RAGValidator

# --- MOCK MediVLM ---
# Since MediVLM was built by another teammate on MIMIC-CXR, 
# you should replace this function with their actual inference code.
import subprocess
import os

def generate_draft_report(image_path):
    """
    Calls the MediVLM inference script built by your teammate.
    Requires the MediVLM repository to be cloned in the same directory.
    """
    # Assuming MediVLM repo is cloned and checkpoint exists
    inference_script = "MediVLM/scripts/inference.py"
    config_file = "MediVLM/configs/mimic_cxr_sample.yaml"
    checkpoint = "MediVLM/outputs/mimic_cxr_sample/best.ckpt"
    
    if not os.path.exists(inference_script) or not os.path.exists(config_file):
        time.sleep(2)
        return "Patient presents with mild respiratory distress. Signs of pleural effusion are visible. The cardiac silhouette is enlarged, suggesting cardiomegaly. Note: This is a fallback draft report because the MediVLM config/checkpoint is missing from the repository."
        
    try:
        # Run inference script and capture output
        result = subprocess.run(
            ["python", inference_script, "--config", config_file, "--checkpoint", checkpoint, "--image", image_path],
            capture_output=True, text=True, check=True
        )
        # Assuming the script prints the final report to stdout
        output = result.stdout.strip()
        # Clean up output if there are logs mixed in (depends on their inference script)
        report_lines = [line for line in output.split('\n') if not line.startswith('INFO')]
        return " ".join(report_lines)
    except subprocess.CalledProcessError as e:
        return f"Error running MediVLM inference: {e.stderr}"


# --- INITIALIZATION ---
st.set_page_config(page_title="ReXTrust Pipeline", layout="wide")

@st.cache_resource
def load_models():
    enhancer = ImageEnhancer(use_unimie=False) # Set True if you have the diffusion checkpoint
    classifier = DiseaseClassifier(model_path="best_hybrid_model.pth")
    rag = RAGValidator(use_gemini=True) # Will fall back to OpenAI if Gemini key missing
    return enhancer, classifier, rag

enhancer, classifier, rag = load_models()

# --- UI FRONTEND ---
st.title("ReXTrust: XAI-Driven Multi-Disease Diagnosis")
st.markdown("Upload a Chest X-ray to run the full diagnostic pipeline (Enhancement -> Vision Classification -> MediVLM -> RAG Validation).")

uploaded_file = st.file_uploader("Choose a Chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    st.write("### 1. Image Enhancement")
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(Image.fromarray(img_rgb), caption="Original Input Image", use_column_width=True)
        
    with st.spinner("Assessing quality and enhancing if necessary..."):
        # Save temp file for the enhancer which expects a path
        temp_path = "temp_input.jpg"
        cv2.imwrite(temp_path, img_bgr)
        
        enhanced_rgb, action_taken = enhancer.enhance(temp_path)
        
    with col2:
        st.image(enhanced_rgb, caption=f"Enhanced Image ({action_taken})", use_column_width=True)
        
    st.divider()
    
    st.write("### 2. Disease Detection & Visual Explainability (ScoreCAM)")
    with st.spinner("Running Hybrid CNN-Transformer..."):
        probs, heatmaps = classifier.predict_and_explain(enhanced_rgb)
        
    # Display probabilities and heatmaps
    num_diseases = len(probs)
    cols = st.columns(num_diseases)
    
    for i, (disease, prob) in enumerate(probs.items()):
        with cols[i]:
            st.metric(label=disease, value=f"{prob*100:.1f}%")
            st.image(Image.fromarray(heatmaps[disease]), caption=f"{disease} Saliency Map", use_column_width=True)
            
    st.divider()
    
    st.write("### 3. Trust-Aware Report Generation")
    col3, col4 = st.columns(2)
    
    with col3:
        st.write("#### MediVLM Draft Report (Unverified)")
        with st.spinner("Generating draft report with MediVLM..."):
            # Pass the saved temp image path to the subprocess
            draft_report = generate_draft_report(temp_path)
        st.info(draft_report)
        
    with col4:
        st.write("#### AMG-RAG Grounded Report (Final)")
        with st.spinner("Validating report against CNN probabilities..."):
            try:
                final_report = rag.validate_report(draft_report, probs)
                st.success(final_report)
            except Exception as e:
                st.error(f"RAG Validation failed (check API keys): {e}")

