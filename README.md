# Skin Cancer Classification using CNN (ISIC Dataset)

This project uses a custom Convolutional Neural Network (CNN) built in TensorFlow/Keras to classify dermoscopic skin lesion images into 9 categories. It is trained on the ISIC Skin Cancer dataset with over 2357 images. The final model is capable of predicting the type of skin cancer from an input image with a professional web interface built using Streamlit.

---

## Dataset

- Source: [ISIC Archive - Skin Cancer Dataset](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic)
- Classes (9 types):
  - actinic_keratosis
  - basal_cell_carcinoma
  - dermatofibroma
  - melanoma
  - nevus
  - pigmented_benign_keratosis
  - seborrheic_keratosis
  - squamous_cell_carcinoma
  - vascular_lesion

---

## Project Workflow

1. Data Preprocessing
   - Normalized pixel values using `rescale=1./255`
   - Handled class imbalance with `Augmentor` to generate synthetic images
   - Split data into training, validation, and test sets

2. Data Augmentation
   - Applied rotation, width/height shift, zoom, shear, horizontal flip

3. Model Architecture
   - Custom CNN with 5 Convolutional Layers:
     - Conv2D + LeakyReLU + MaxPooling
     - Increasing filters: 32 → 128
     - Kernel sizes: 3×3 (shallow layers), 5×5 (deep layers)
   - Dense layers: 512 → 256 with Dropout(0.5)
   - Output layer: Dense(9, Softmax)

5. Model Training
   - Batch size: 32, Epochs: 25
   - Learning rate: 0.0001 with Adam optimizer
   - Model saved as `custom_skin_cancer_model_finalreview.h5`

6. Web Application Interface
   - Professional Streamlit web application
   - Real-time image upload and classification
   - Confidence visualization and detailed prediction breakdown
   - Responsive design for desktop and mobile browsers

---

## Libraries Used

- `tensorflow` - Deep learning framework
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `matplotlib`, `seaborn` - Data visualization
- `Augmentor` - Data augmentation
- `streamlit` - Web application framework
- `pillow` - Image processing

---

## Key Features

- Trained on real-world dermoscopic images from ISIC dataset
- Custom CNN architecture optimized for medical image classification
- LeakyReLU activation to prevent dying neurons in deeper layers
- Dropout regularization for overfitting control
- Professional web interface for easy accessibility
- Real-time confidence scores and prediction breakdown
- Responsive design for all devices

---

## Installation and Setup

### Requirements

- Python 3.7 or higher
- Windows, macOS, or Linux
- At least 2 GB RAM (4+ GB recommended)
- GPU optional but recommended for faster inference

### Step 1: Install Dependencies

```bash
cd C:\Users\91789\PycharmProjects\pythonProject
pip install -r requirements.txt
```

### Step 2: Run the Application

**Option A: Windows Batch File (Recommended)**
```bash
Double-click: run.bat
```

**Option B: Command Line**
```bash
streamlit run streamlit_app.py
```

The application will start at: `http://localhost:8502`

---

## How to Use the Application

### Uploading an Image

1. Open the application in your web browser
2. Click the "Upload a dermoscopic image" button
3. Select an image file (JPG, JPEG, or PNG format)
4. Maximum file size: 50 MB

### Viewing Results

The application displays four main sections:

1. **Uploaded Image** - Shows your selected dermoscopic image

2. **Classification Result** - Displays:
   - Predicted skin cancer type
   - Confidence percentage (0-100%)

3. **Confidence Distribution** - Bar chart showing model prediction confidence for all 9 classes

4. **Detailed Scores Table** - Complete list of all predictions with percentages, sorted by confidence

### Classification Categories

The model can identify:
- Actinic Keratosis
- Basal Cell Carcinoma
- Dermatofibroma
- Melanoma
- Nevus
- Pigmented Benign Keratosis
- Seborrheic Keratosis
- Squamous Cell Carcinoma
- Vascular Lesion

---

## Model Performance

| Parameter | Value |
|-----------|-------|
| Input Image Size | 180×180 pixels |
| Training Samples | 2357+ images |
| Validation Split | 20% |
| Optimizer | Adam (learning_rate=0.0001) |
| Loss Function | Categorical Crossentropy |
| Training Epochs | 25 |
| Batch Size | 32 |
| Inference Time | 1-3 seconds per image |

---

## Project Structure

```
pythonProject/
├── streamlit_app.py                 # Main web application
├── inference.py                     # Model inference module
├── config.py                        # Project configuration
├── model_manager.py                 # Model management utilities
├── masterclassification.py          # Training script
├── custom_skin_cancer_model_finalreview.h5  # Pre-trained model
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── .streamlit/                      # Streamlit configuration
```

---

## Training the Model

To retrain the model with your own data:

1. Download the ISIC dataset and place in the configured directory
2. Update paths in `config.py` if necessary
3. Run:
```bash
python masterclassification.py
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Application won't start | Check Python version (3.7+) and run `pip install -r requirements.txt` |
| "Model not found" error | Ensure `custom_skin_cancer_model_finalreview.h5` is in project root |
| Port 8502 already in use | Run `streamlit run streamlit_app.py --server.port 8503` |
| Slow predictions | Predictions typically take 1-3 seconds; consider GPU acceleration |
| Import errors | Try `pip install --upgrade -r requirements.txt` |

---

## Medical Disclaimer

**Important Notice:** This application is designed for educational and informational purposes only. It should NOT be used for:
- Clinical diagnosis or medical decision-making
- Treatment planning or recommendations
- Replacement of professional medical evaluation

Always consult a qualified dermatologist for accurate medical diagnosis and treatment decisions. The model predictions are based on statistical patterns and may not always be accurate.

---



---

## License

This project uses the ISIC dataset for educational purposes. Refer to ISIC's terms of use for proper attribution and usage rights.

---



## Author 
Manish Negi
negi94432@gmail.com




