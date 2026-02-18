"""
Skin Cancer Image Classification Module

This module provides functionality for loading the pre-trained model
and performing predictions on dermoscopic images.
"""

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img


class SkinCancerClassifier:
    """Handles model loading and image classification."""
    
    CLASS_LABELS = [
        'actinic_keratosis',
        'basal_cell_carcinoma',
        'dermatofibroma',
        'melanoma',
        'nevus',
        'pigmented_benign_keratosis',
        'seborrheic_keratosis',
        'squamous_cell_carcinoma',
        'vascular_lesion'
    ]
    
    TARGET_SIZE = (180, 180)
    MODEL_PATH = 'custom_skin_cancer_model_finalreview.h5'
    
    def __init__(self, model_path=None):
        """
        Initialize the classifier with a pre-trained model.
        
        Args:
            model_path (str): Path to the h5 model file. If None, uses default path.
        """
        self.model_path = model_path or self.MODEL_PATH
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the pre-trained Keras model."""
        try:
            self.model = load_model(self.model_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Model file not found at: {self.model_path}. "
                "Please ensure the model file exists in the project directory."
            )
    
    def predict(self, image_path):
        """
        Predict the skin cancer class for a given image.
        
        Args:
            image_path (str): Path to the image file.
            
        Returns:
            dict: Contains predicted class and confidence scores.
        """
        try:
            img = load_img(image_path, target_size=self.TARGET_SIZE)
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            predictions = self.model.predict(img_array, verbose=0)
            predicted_class_index = np.argmax(predictions)
            predicted_class_label = self.CLASS_LABELS[predicted_class_index]
            confidence = float(predictions[0][predicted_class_index])
            
            confidence_scores = {
                self.CLASS_LABELS[i]: float(predictions[0][i])
                for i in range(len(self.CLASS_LABELS))
            }
            
            return {
                'predicted_class': predicted_class_label,
                'confidence': confidence,
                'all_scores': confidence_scores
            }
        except Exception as e:
            raise Exception(f"Error during prediction: {str(e)}")
