import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
import urllib.request

# Load the pre-trained ResNet50 model
base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Create a custom model for binary classification
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))  # Dropout for regularization
model.add(Dense(1, activation='sigmoid'))

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Function to classify window cleanliness
def classify_window_cleanliness(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = preprocess_input(img)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))

    # Make predictions
    prediction = model.predict(img)
    
    # Threshold the prediction (e.g., if > 0.5, it's clean, otherwise, it's not clean)
    if prediction > 0.5:
        return "Clean"
    else:
        return "Not Clean"

# Example usage
image_url = 'https://live.staticflickr.com/2874/8895879575_e8cbaabdf2_b.jpg'  # Replace with the URL of your window image
image_path, _ = urllib.request.urlretrieve(image_url)
cleanliness_label = classify_window_cleanliness(image_path)
print(f"The window is classified as: {cleanliness_label}")
