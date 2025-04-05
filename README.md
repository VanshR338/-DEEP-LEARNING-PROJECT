# -DEEP-LEARNING-PROJECT

NAME: Vansh Raina

INTERN ID: CT08WQF

DOMAIN: DATA SCIENCE DEVELOPMENT

DURATION: 4 WEEEKS

"MENTOR*: NEELA SANTOSH

# -Project Description
This project is a practical implementation of a Convolutional Neural Network (CNN) to automatically recognize and classify handwritten digits. The task falls under the category of image classification and serves as an excellent introduction to deep learning in computer vision.

The CNN model is designed to capture and learn spatial hierarchies in images using a series of convolutional operations followed by non-linear activation and pooling. These operations help the model identify relevant features such as curves, lines, and shapes that represent digits. After training, the model can accurately predict which digit (0–9) is present in an unseen image by outputting a probability distribution across the 10 possible classes.

The MNIST dataset’s simplicity makes it ideal for beginners, yet the task of achieving high accuracy challenges the developer to build an efficient and generalizable model. Despite its small size, the dataset represents a wide range of handwriting styles, ensuring the model learns to identify digits with high variability.

This project not only strengthens the understanding of CNNs and their architecture but also emphasizes best practices like data normalization, model evaluation, and the use of validation data.

# -Applications
Banking Sector:

Reading handwritten digits on cheques, forms, and receipts.

Postal Services:

Automatic ZIP code recognition from handwritten envelopes and packages.

Educational Technology:

Real-time handwriting recognition for digital notebooks or math practice apps.

Form Digitization:

Scanning and converting handwritten forms into digital records using OCR systems.

Mobile and Web Applications:

Integration in apps that require digit recognition through camera or touchscreen input.

Assistive Technologies:

Helping visually impaired users by recognizing and reading out handwritten content.
Dataset:

The project uses the MNIST dataset, a well-known benchmark in the field of machine learning and computer vision. It contains 70,000 grayscale images of handwritten digits (60,000 for training and 10,000 for testing), each image being 28x28 pixels.

Data Preprocessing:

Input images are normalized by scaling pixel values to the range [0, 1].

The images are reshaped to add a channel dimension, transforming the shape to (28, 28, 1), which is required for convolutional layers.

Model Architecture (CNN):

A sequential Convolutional Neural Network is implemented using TensorFlow and Keras.

The architecture includes:

Input Layer

Two Convolutional Layers with ReLU activation and MaxPooling layers for feature extraction and dimensionality reduction.

Flatten Layer to convert feature maps into a one-dimensional vector.

Dense Layer with 128 neurons to learn high-level patterns.

Output Layer with 10 neurons using softmax activation for classification.

Model Compilation & Training:

Compiled with Adam optimizer, using sparse categorical crossentropy as the loss function and accuracy as the evaluation metric.

Trained over multiple epochs to optimize performance on the training set and generalize well to unseen data.

Evaluation and Visualization:

The model is tested on a separate test set.

Accuracy and loss plots are used to visualize training performance.

Evaluation includes measuring overall classification accuracy.

# -OUTPUT

![Image](https://github.com/user-attachments/assets/8272edcb-a0f8-444e-b742-09848d9062e3)
