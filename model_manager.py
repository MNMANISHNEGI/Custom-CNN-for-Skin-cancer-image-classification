"""
Model Management Utilities

This module provides utilities for managing the trained model,
including loading, saving, and validation functions.
"""

import os
from pathlib import Path


class ModelManager:
    """Manages model paths and validation."""
    
    MODEL_FILENAME = 'custom_skin_cancer_model_finalreview.h5'
    
    @staticmethod
    def get_model_path():
        """
        Get the full path to the model file.
        
        Returns:
            str: Absolute path to the model file
        """
        current_dir = Path(__file__).parent
        model_path = current_dir / ModelManager.MODEL_FILENAME
        return str(model_path)
    
    @staticmethod
    def model_exists():
        """
        Check if the model file exists.
        
        Returns:
            bool: True if model file exists, False otherwise
        """
        return os.path.exists(ModelManager.get_model_path())
    
    @staticmethod
    def get_model_info():
        """
        Get information about the model file.
        
        Returns:
            dict: Contains size and exists status
        """
        path = ModelManager.get_model_path()
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            return {
                'exists': True,
                'path': path,
                'size_mb': round(size_mb, 2)
            }
        return {
            'exists': False,
            'path': path,
            'size_mb': 0
        }


if __name__ == "__main__":
    info = ModelManager.get_model_info()
    print("Model Status:")
    print(f"  Exists: {info['exists']}")
    print(f"  Path: {info['path']}")
    print(f"  Size: {info['size_mb']} MB")
