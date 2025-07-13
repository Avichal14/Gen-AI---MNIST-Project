# MNIST Digit Classification with TensorFlow

This project demonstrates a neural network built using TensorFlow and Keras to classify handwritten digits from the MNIST dataset. The MNIST dataset is a classic benchmark in the machine learning community consisting of grayscale images of handwritten digits (0–9).

---

## 🚀 Project Overview

The goal is to develop a digit classification model using a Convolutional Neural Network (CNN). The project walks through data preprocessing, model building, training, and evaluation.

---

## 🧠 Model Architecture

- **Input Layer**: 28x28 grayscale images
- **Normalization**: Rescale pixel values from [0, 255] to [0, 1]
- **Conv2D Layer**: 32 filters, 3x3 kernel, ReLU activation
- **MaxPooling2D**: Pool size 2x2
- **Conv2D Layer**: 64 filters, 3x3 kernel, ReLU activation
- **MaxPooling2D**
- **Flatten Layer**
- **Dense Layer**: 64 units, ReLU activation
- **Output Layer**: 10 units (for each digit), Softmax activation

---

## 🛠️ Libraries Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib (for plotting)

---

## 📊 Dataset

- **MNIST Dataset**: 60,000 training images and 10,000 testing images.
- Preloaded from `tensorflow.keras.datasets`.

---

## 🔄 Workflow

1. **Data Loading**: Load train/test images and labels.
2. **Data Normalization**: Scale image pixel values to range [0, 1].
3. **Reshaping**: Convert data into shape suitable for CNN input.
4. **Model Building**: Use `Sequential` model with Conv2D and Dense layers.
5. **Model Compilation**: Categorical Crossentropy loss and Adam optimizer.
6. **Training**: Fit the model on training data with validation split.
7. **Evaluation**: Assess model performance on the test set.
8. **Prediction & Visualization**: Display test results and predictions.

---

## 📈 Results

- Achieved **>98% accuracy** on the test set.
- Visualization of predictions on random test images included.

---

## 🖼️ Sample Predictions

Includes plotted test images with predicted vs actual labels to demonstrate performance.

---

## 💻 How to Run

```bash
# Clone this repository
git clone https://github.com/Avichal14/mnist-digit-classification.git
cd mnist-digit-classification

# Install required libraries
pip install tensorflow matplotlib numpy

# Run the notebook
jupyter notebook Gen_AI_mnist_Project.ipynb
