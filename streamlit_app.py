"""
Streamlit Web Application for Skin Cancer Classification

This application provides a professional interface for uploading dermoscopic images
and obtaining predictions using a pre-trained deep learning model.
"""

import streamlit as st
from PIL import Image
import io
from inference import SkinCancerClassifier
import pandas as pd
import os


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'classifier' not in st.session_state:
        st.session_state.classifier = None
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None


def load_classifier():
    """Load the classification model with error handling."""
    if st.session_state.classifier is None:
        try:
            st.session_state.classifier = SkinCancerClassifier()
        except FileNotFoundError as e:
            st.error(f"Model Loading Error: {str(e)}")
            return False
    return True


def display_header():
    """Display application header and description."""
    st.set_page_config(
        page_title="Skin Cancer Classification",
        page_icon="",
        layout="centered",
        initial_sidebar_state="expanded"
    )
    
    st.title("Skin Cancer Classification System")
    st.markdown("""
    This application uses deep learning to classify dermoscopic images into one of 9 skin cancer types.
    Upload an image to receive a classification prediction with confidence scores.
    """)
    st.divider()


def display_info_section():
    """Display information about supported classes."""
    with st.expander("View Supported Classifications"):
        classes_info = {
            'actinic_keratosis': 'Pre-cancerous lesion induced by sun exposure',
            'basal_cell_carcinoma': 'Most common form of skin cancer',
            'dermatofibroma': 'Benign skin growth',
            'melanoma': 'Most serious type of skin cancer',
            'nevus': 'Common mole',
            'pigmented_benign_keratosis': 'Benign brown spot',
            'seborrheic_keratosis': 'Common harmless growth',
            'squamous_cell_carcinoma': 'Common skin cancer',
            'vascular_lesion': 'Blood vessel-related lesion'
        }
        
        for class_name, description in classes_info.items():
            st.markdown(f"**{class_name.replace('_', ' ').title()}**")
            st.write(description)
            st.divider()


def process_uploaded_image(uploaded_file):
    """
    Process uploaded image file and save temporarily.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        str: Path to saved image file
    """
    file_path = f"temp_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def display_prediction_results(prediction_result, image_path):
    """
    Display prediction results with visualizations.
    
    Args:
        prediction_result (dict): Dictionary containing prediction and confidence scores
        image_path (str): Path to the uploaded image
    """
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Uploaded Image")
        image = Image.open(image_path)
        st.image(image, use_column_width=True)
    
    with col2:
        st.subheader("Classification Result")
        predicted_class = prediction_result['predicted_class'].replace('_', ' ').title()
        confidence = prediction_result['confidence']
        
        st.metric(
            label="Predicted Class",
            value=predicted_class,
            delta=f"{confidence*100:.1f}% confidence"
        )
    
    st.divider()
    
    st.subheader("Confidence Distribution")
    scores = prediction_result['all_scores']
    scores_display = {
        k.replace('_', ' ').title(): v
        for k, v in scores.items()
    }
    
    st.bar_chart(scores_display)
    
    st.divider()
    st.subheader("Detailed Scores")
    
    scores_sorted = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    score_data = []
    for i, (class_name, score) in enumerate(scores_sorted, 1):
        score_data.append({
            'Rank': i,
            'Class': class_name.replace('_', ' ').title(),
            'Confidence': f"{score*100:.2f}%"
        })
    
    import pandas as pd
    df_scores = pd.DataFrame(score_data)
    st.dataframe(df_scores, use_container_width=True, hide_index=True)


def main():
    """Main application function."""
    initialize_session_state()
    display_header()
    display_info_section()
    
    st.markdown("## Image Classification")
    
    if not load_classifier():
        st.stop()
    
    uploaded_file = st.file_uploader(
        "Upload a dermoscopic image",
        type=['jpg', 'jpeg', 'png'],
        help="Supported formats: JPG, JPEG, PNG"
    )
    
    if uploaded_file is not None:
        try:
            with st.spinner("Processing image and generating prediction..."):
                image_path = process_uploaded_image(uploaded_file)
                prediction_result = st.session_state.classifier.predict(image_path)
                st.session_state.prediction_result = prediction_result
            
            display_prediction_results(prediction_result, image_path)
            
            if os.path.exists(image_path):
                os.remove(image_path)
        
        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")
    
    st.divider()
    st.markdown("""
    **Important Notice:** This tool is for informational purposes only
    and should not be used as a substitute for professional medical diagnosis.
    Please consult a dermatologist for accurate medical evaluation.
    """)


if __name__ == "__main__":
    main()
