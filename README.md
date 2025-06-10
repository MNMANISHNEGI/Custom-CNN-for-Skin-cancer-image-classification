# 🔬 Skin Cancer Classification using CNN (ISIC Dataset)

This project uses a custom Convolutional Neural Network (CNN) built in TensorFlow/Keras to classify dermoscopic skin lesion images into 9 categories. It is trained on the ISIC Skin Cancer dataset with over 2357  images. The final model is capable of predicting the type of skin cancer from an input image and includes a GUI for real-time image upload and prediction using Tkinter.

---

## 📁 Dataset

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

## 📊 Project Workflow

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

4. Compilation & Training
   - Loss Function: `categorical_crossentropy`
   - Optimizer: Adam (`learning_rate=0.0001`)
   - Metrics: Accuracy
   - Epochs: 25

5. Evaluation
   - Tracked accuracy and loss over epochs
   - Visualized using `matplotlib`

6. GUI Interface (Tkinter)
   - Allows uploading of skin lesion image
   - Predicts and displays disease class

---

## 🧪 Libraries Used

- `tensorflow`, `keras`
- `numpy`, `pandas`
- `matplotlib`, `seaborn`
- `Augmentor`
- `tkinter`, `PIL`

---

## 🧠 Key Features

- Trained on real-world dermoscopic images
- Custom CNN architecture optimized for small datasets
- LeakyReLU used to prevent dying neurons in deeper layers
- Dropout for overfitting control
- GUI app for real-time predictions
- Clear plots of accuracy, validation, and class distributions

---

