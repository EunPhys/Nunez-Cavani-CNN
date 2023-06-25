import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

# Training and test data 
train_dir = '/Users/eunanpro/Desktop/Untitled Folder/Data/nunez-cavani/train'
test_dir = '/Users/eunanpro/Desktop/Untitled Folder/Data/nunez-cavani/test'

# Initialize lists to store the training and testing data
train_data = []
train_labels = []
test_data = []
test_labels = []

# Resize of input images
target_size = (150, 150)

# Data augmentation using ImageDataGenerator
datagen = ImageDataGenerator(width_shift_range=0.2)

# Preprocessing steps including Haar cascade based head zoom ('haarcascade_frontalface_default.xml')
for class_name in os.listdir(train_dir):
    class_dir = os.path.join(train_dir, class_name)
    if not os.path.isdir(class_dir):
        continue  # Skip non-directory files
    class_label = 1 if class_name == 'nunez' else 0  # Assign label based on class name

    for file in os.listdir(class_dir):
        img_path = os.path.join(class_dir, file)
        img = cv2.imread(img_path)

        # Skip the image if it cannot be read or has an empty size
        if img is None or img.size == 0:
            print(f"Skipped image: {img_path}")
            continue

        # Perform face detection and crop around the head region
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            (x, y, w, h) = faces[0]  # Crop around the first detected face
            img = img[y:y + h, x:x + w]

        img = cv2.resize(img, target_size)
        img = img / 255.0  # Normalize the image pixels to the range [0, 1]

        # Perform additional preprocessing or data augmentation using ImageDataGenerator
        img = np.squeeze(img)  # Remove single-dimensional entries from the shape of the array
        img = img.astype(np.float32)  # Convert image data type to np.float32

        # Apply random data augmentation with randomly selected transformation parameters
        if np.random.rand() < 0.5:
            augmented_images = datagen.flow(np.expand_dims(img, axis=0), batch_size=1)
            for augmented_image in augmented_images:
                augmented_image = augmented_image[0]
                train_data.append(augmented_image)
                train_labels.append(class_label)
                break  # Break the loop to generate only one augmented image
        else:
            train_data.append(img)
            train_labels.append(class_label)

# Preprocess the TEST images
for class_name in os.listdir(test_dir):
    class_dir = os.path.join(test_dir, class_name)
    if not os.path.isdir(class_dir):
        continue

    class_label = 1 if class_name == 'nunez' else 0  # Assign label based on class name

    for file in os.listdir(class_dir):
        img_path = os.path.join(class_dir, file)
        img = cv2.imread(img_path)

        # Skip the image if it cannot be read or has an empty size
        if img is None or img.size == 0:
            print(f"Skipped image: {img_path}")
            continue

        # Perform face detection and crop around the head region
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            (x, y, w, h) = faces[0]  # Crop around the first detected face
            img = img[y:y + h, x:x + w]

        img = cv2.resize(img, target_size)
        img = img / 255.0  # Normalize the image pixels to the range [0, 1]

        img = np.squeeze(img)  # Remove single-dimensional entries from the shape of the array
        img = img.astype(np.float32)  # Convert image data type to np.float32

        test_data.append(img)
        test_labels.append(class_label)

# Convert the data and labels to NumPy arrays
train_data = np.array(train_data)
train_labels = np.array(train_labels)
test_data = np.array(test_data)
test_labels = np.array(test_labels)

# Print the shapes of the train and test data - just a check
print(f"Train Data Shape: {train_data.shape}")
print(f"Train Labels Shape: {train_labels.shape}")
print(f"Test Data Shape: {test_data.shape}")
print(f"Test Labels Shape: {test_labels.shape}")

# Build the model, based on tensorflow/keras
model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(512, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))  # Add dropout for regularization
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

model.summary()
# Set the number of epochs
num_epochs = 10

# Train the model
history = model.fit(train_data, train_labels, epochs=num_epochs, validation_split=0.2)

# Evaluate the model on the training set
train_loss, train_accuracy = model.evaluate(train_data, train_labels)

print(f"Training Loss: {train_loss:.4f}")
print(f"Training Accuracy: {train_accuracy:.4f}")

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_data, test_labels)

# Print the evaluation results
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Get the model predictions on the test data
test_predictions = model.predict(test_data)
test_predictions = (test_predictions > 0.5).astype(int)  # Convert probabilities to binary predictions

# Calculate evaluation metrics
true_positives = 0
true_negatives = 0
false_positives = 0
false_negatives = 0

for i in range(len(test_labels)):
    if test_predictions[i] == 1 and test_labels[i] == 1:
        true_positives += 1
    elif test_predictions[i] == 0 and test_labels[i] == 0:
        true_negatives += 1
    elif test_predictions[i] == 1 and test_labels[i] == 0:
        false_positives += 1
    elif test_predictions[i] == 0 and test_labels[i] == 1:
        false_negatives += 1

# Calculate evaluation metrics
accuracy = (true_positives + true_negatives) / len(test_labels)
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1_score = 2 * (precision * recall) / (precision + recall)

# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
