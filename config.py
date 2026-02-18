"""
Project Configuration

Central configuration file for the Skin Cancer Classification project.
"""

import os
from pathlib import Path


class Config:
    """Project configuration settings."""
    
    # Project directories
    PROJECT_ROOT = Path(__file__).parent
    MODEL_DIR = PROJECT_ROOT
    DATA_DIR = Path(r'C:\Users\91789\Desktop\Skin cancer ISIC The International Skin Imaging Collaboration')
    
    # Model settings
    MODEL_NAME = 'custom_skin_cancer_model_finalreview.h5'
    MODEL_PATH = MODEL_DIR / MODEL_NAME
    
    # Image settings
    IMAGE_HEIGHT = 180
    IMAGE_WIDTH = 180
    IMAGE_CHANNELS = 3
    TARGET_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
    
    # Training settings
    BATCH_SIZE = 32
    EPOCHS = 25
    LEARNING_RATE = 0.0001
    VALIDATION_SPLIT = 0.2
    
    # Data paths
    TRAIN_DATA_PATH = DATA_DIR / 'train'
    TEST_DATA_PATH = DATA_DIR / 'test'
    
    # Class labels
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
    
    NUM_CLASSES = len(CLASS_LABELS)
    
    # Augmentation settings
    AUGMENTATION_CONFIG = {
        'rescale': 1.0 / 255,
        'shear_range': 0.2,
        'zoom_range': 0.2,
        'width_shift_range': 0.2,
        'height_shift_range': 0.2,
        'rotation_range': 40,
        'horizontal_flip': True,
        'vertical_flip': False
    }
    
    @classmethod
    def verify_data_paths(cls):
        """
        Verify that required data directories exist.
        
        Returns:
            dict: Status of each path
        """
        paths = {
            'train': cls.TRAIN_DATA_PATH.exists(),
            'test': cls.TEST_DATA_PATH.exists(),
            'model': cls.MODEL_PATH.exists()
        }
        return paths


if __name__ == "__main__":
    print("Project Configuration Check")
    print("=" * 50)
    print(f"Project Root: {Config.PROJECT_ROOT}")
    print(f"Model Path: {Config.MODEL_PATH}")
    print(f"Number of Classes: {Config.NUM_CLASSES}")
    print(f"Image Size: {Config.IMAGE_HEIGHT}x{Config.IMAGE_WIDTH}")
    print("\nData Path Status:")
    paths = Config.verify_data_paths()
    for path_name, exists in paths.items():
        status = "OK" if exists else "MISSING"
        print(f"  {path_name}: {status}")
