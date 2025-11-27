"""Streamlit app for BIRADS classification"""
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from pathlib import Path

from src.explainability import ModelExplainer
from src.model import build_birads_model

st.set_page_config(page_title="BIRADS Classification", page_icon="🔬")
st.title("🔬 Mammography BIRADS Classification")

MODELS_DIR = Path("models")
WEIGHTS_PATH = MODELS_DIR / "birads_model.h5"

uploaded_file = st.file_uploader("Upload mammography image", type=['png', 'jpg'])
if uploaded_file:
    img = Image.open(uploaded_file).resize((256, 224))
    st.image(img)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, 0)
    
    if st.button("Classify"):
        # Ensure models directory exists
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # Build model and load weights if available
        model = build_birads_model()
        if WEIGHTS_PATH.exists():
            st.info(f"Loading trained weights from `{WEIGHTS_PATH}`")
            model.load_weights(str(WEIGHTS_PATH))
        else:
            st.warning(
                "⚠️ Trained weights not found. Using randomly initialized model "
                "(predictions will be unreliable). Train the model first."
            )

        pred = model.predict(img_array, verbose=0)
        birads_class = np.argmax(pred[0]) + 1
        confidence = pred[0][birads_class - 1]
        st.success(f"BIRADS Class: {birads_class} (Confidence: {confidence:.2%})")
        
        # Explainability section
        st.divider()
        st.subheader("🔍 Explainability")
        if st.button("Explain Prediction (SHAP)"):
            try:
                import matplotlib.pyplot as plt
                import shap
                
                with st.spinner("Computing SHAP values..."):
                    explainer = ModelExplainer(model, img_array[:1])
                    shap_values = explainer.explain_instance(img_array, plot=False, class_idx=birads_class-1)
                    
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    axes[0].imshow(img_array[0])
                    axes[0].set_title('Original Image')
                    axes[0].axis('off')
                    
                    shap_image = np.abs(shap_values[0])
                    if len(shap_image.shape) == 3:
                        shap_image = shap_image.sum(axis=2)
                    
                    axes[1].imshow(shap_image, cmap='hot')
                    axes[1].set_title('SHAP Values (Importance)')
                    axes[1].axis('off')
                    
                    axes[2].imshow(img_array[0])
                    axes[2].imshow(shap_image, cmap='hot', alpha=0.5)
                    axes[2].set_title('Overlay')
                    axes[2].axis('off')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    st.info("💡 Red/hot regions indicate areas that strongly influence the BIRADS classification.")
            except Exception as e:
                st.error(f"Error computing explanation: {str(e)}")

