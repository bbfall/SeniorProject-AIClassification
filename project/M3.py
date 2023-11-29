import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import os
import pandas as pd

# Define your data directory
data_dir = 'data'

# Define subdirectories for clean and dirty windows
clean_dir = os.path.join(data_dir, 'clean')
dirty_dir = os.path.join(data_dir, 'dirty')

# List all the image files in the clean and dirty subdirectories
clean_image_files = [os.path.join(clean_dir, filename) for filename in os.listdir(clean_dir) if filename.endswith(('.jpg', '.png'))]
dirty_image_files = [os.path.join(dirty_dir, filename) for filename in os.listdir(dirty_dir) if filename.endswith(('.jpg', '.png'))]

# Create labels for clean and dirty images
clean_labels = [1] * len(clean_image_files)
dirty_labels = [0] * len(dirty_image_files)

# Combine the image files and labels
all_image_files = clean_image_files + dirty_image_files
all_labels = clean_labels + dirty_labels

# Split the data into training and testing sets
train_files, test_files, train_labels, test_labels = train_test_split(all_image_files, all_labels, test_size=0.2, random_state=42)

# Data preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Convert labels to strings
train_labels = [str(label) for label in train_labels]
test_labels = [str(label) for label in test_labels]

# Create data generators for training and testing
train_generator = train_datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({'filename': train_files, 'class': train_labels}),
    x_col="filename",
    y_col="class",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({'filename': test_files, 'class': test_labels}),
    x_col="filename",
    y_col="class",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Create a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.summary()

# Compile the model
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Train the model
model.fit(train_generator, epochs=10, validation_data=test_generator)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test accuracy: {test_accuracy}")

# Save the model for deployment
model.save('window_clean_classifier.h5')
