✅ Overview

This project implements a Convolutional Neural Network (CNN) using TensorFlow/Keras to classify handwritten digits (0–9) from the MNIST dataset.
The model is trained on 60,000 images and tested on 10,000 images, achieving high accuracy.

✅ Dataset
MNIST Dataset: A collection of 70,000 grayscale images (28×28 pixels) of handwritten digits (0–9).
Source: Available in tensorflow.keras.datasets.

✅ Project Features
Built using TensorFlow and Keras.

Preprocessing:
Normalized pixel values (0–1).
Reshaped images for CNN input.

CNN Architecture:
Conv2D → MaxPooling → Flatten → Dense layers.

Implemented Dropout for regularization.
Achieved 99.15% accuracy on test data.

Evaluated with:
Confusion Matrix
Classification Report

✅ Model Architecture
Conv2D (32 filters, 3x3, ReLU)
MaxPooling2D
Conv2D (64 filters, 3x3, ReLU)
MaxPooling2D
Flatten
Dense (128 neurons, ReLU)
Dropout (0.3)
Dense (10 neurons, Softmax)

✅ Results
Test Accuracy: 99.15%
Classification Report:
precision    recall    f1-score    support
0     0.99       1.00      0.99       980
1     0.99       1.00      0.99      1135
...
accuracy                          0.9915    10000

✅ Sample Predictions
True	Predicted
6	      6
2	      2
3      	3
7	      7
2	      2

✅ Technologies Used
Python
TensorFlow / Keras
NumPy
Matplotlib
