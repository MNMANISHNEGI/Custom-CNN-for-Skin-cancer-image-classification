# Project Documentation

## Directory Structure

```
pythonProject/
├── Core Application Files
│   ├── streamlit_app.py              - Main Streamlit web application
│   ├── inference.py                  - Model inference module
│   ├── model_manager.py              - Model management utilities
│   └── config.py                     - Project configuration
│
├── Training & Development
│   ├── masterclassification.py        - Primary training script with preprocessing
│   ├── practice9class.py              - CNN model architecture definition
│   ├── newpractice9.py                - Data augmentation and visualization
│   └── app.py                         - Flask backend (legacy)
│
├── Model & Data
│   ├── custom_skin_cancer_model_finalreview.h5  - Pre-trained model (50+ MB)
│   └── [Training data goes to C:\Users\...\Desktop\Skin cancer ISIC...]
│
├── Configuration
│   ├── .streamlit/
│   │   └── config.toml                - Streamlit application settings
│   ├── requirements.txt               - Python dependencies
│   ├── run.bat                        - Windows launcher script
│   └── config.py                      - Project configuration
│
└── Documentation
    ├── README.md                      - Project overview
    └── DEVELOPMENT.md                 - This file
```

## File Descriptions

### Application Layer

**streamlit_app.py**
- Main web application interface
- Handles image upload, prediction display, and visualization
- Entry point: `streamlit run streamlit_app.py`

**inference.py**
- `SkinCancerClassifier` class with model loading and prediction logic
- Handles image preprocessing and confidence calculations
- Provides structured prediction output

**model_manager.py**
- Utilities for model file management
- Validates model existence and retrieves model information
- Used during application initialization

**config.py**
- Centralized configuration for all parameters
- Image dimensions, batch sizes, class labels
- Data paths and training hyperparameters

### Training Layer

**masterclassification.py**
- Complete training pipeline
- Data loading from ISIC dataset
- Preprocessing, augmentation, and visualization
- Model training with metrics tracking

**practice9class.py**
- CNN model architecture definition
- Standalone training implementation
- Reference architecture with LeakyReLU and Dropout

**newpractice9.py**
- Data augmentation strategies
- Dataset analysis and visualization
- Class distribution analysis using Augmentor library

### Infrastructure

**run.bat**
- Windows batch script to launch the application
- Automatically checks and installs dependencies
- Starts Streamlit on localhost:8501

**.streamlit/config.toml**
- Streamlit framework configuration
- Theme, client settings, and logging configuration
- Professional color scheme (blue theme)

**requirements.txt**
- Python package dependencies with pinned versions
- Used for reproducible environment setup

## Getting Started

### Installation Steps

1. **Clone/Download the project**
```bash
cd pythonProject
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Verify setup**
```bash
python config.py
```

4. **Download dataset** (for training only)
- Download ISIC dataset from official source
- Extract to: `C:\Users\91789\Desktop\Skin cancer ISIC The International Skin Imaging Collaboration`

### Running the Application

**Option 1: Using batch script (Windows)**
```bash
run.bat
```

**Option 2: Using command line**
```bash
streamlit run streamlit_app.py
```

**Option 3: With custom configuration**
```bash
streamlit run streamlit_app.py --logger.level=debug
```

## Development Guidelines

### Code Organization

- **Separation of Concerns**: Core logic in `inference.py`, UI in `streamlit_app.py`, config in `config.py`
- **Reusability**: Training scripts are independent; inference module is standalone
- **Configuration**: All adjustable parameters in `config.py`

### Adding New Features

1. **New classification logic**: Add methods to `SkinCancerClassifier` in `inference.py`
2. **UI improvements**: Modify `streamlit_app.py` layout and interactions
3. **Training enhancements**: Update `masterclassification.py` with new data or architectures
4. **Configuration changes**: Update `config.py` with new parameters

### Testing

```bash
# Verify model loading
python -c "from inference import SkinCancerClassifier; clf = SkinCancerClassifier()"

# Check configuration
python config.py

# Check model status
python model_manager.py
```

## Performance Notes

- Model file: 50-100 MB (h5 format)
- Inference time: 1-3 seconds per image (GPU recommended)
- Streamlit startup: 5-10 seconds (first run longer)
- Supported image formats: JPG, JPEG, PNG
- Maximum upload size: 50 MB (configurable)

## Dependencies Overview

| Package | Version | Purpose |
|---------|---------|---------|
| TensorFlow | 2.13.0 | Deep learning framework |
| Keras | 2.13.0 | Neural network API |
| Streamlit | 1.28.1 | Web application framework |
| NumPy | 1.24.3 | Numerical computing |
| Pandas | 2.0.3 | Data manipulation |
| Pillow | 10.0.0 | Image processing |
| Matplotlib | 3.7.2 | Data visualization |
| Seaborn | 0.12.2 | Statistical visualization |
| Augmentor | 0.2.12 | Data augmentation |

## Troubleshooting

### Model not found
- Ensure `custom_skin_cancer_model_finalreview.h5` is in project root
- Run `python model_manager.py` to check model status

### Import errors
- Verify all dependencies: `pip install -r requirements.txt`
- Consider creating a virtual environment

### Slow predictions
- GPU acceleration recommended for TensorFlow
- Increase system RAM for faster preprocessing

### Data path issues
- Update paths in `config.py` if dataset location differs
- Ensure dataset directory structure matches ISIC format

## Professional Standards

- Code follows PEP 8 style guidelines
- Docstrings for all modules and classes
- Error handling with informative messages
- No emoji in production code (professional presentation)
- Comprehensive logging and status reporting

## Medical Disclaimer

This application is for educational and informational purposes only.
It should not be used for actual medical diagnosis or clinical decision-making.
Always consult qualified healthcare professionals for medical diagnosis.
