import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import urllib.request

# Load the pre-trained ResNet50 model
model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers for binary classification (clean or not clean)
x = model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)

custom_model = tf.keras.models.Model(inputs=model.input, outputs=predictions)

# Compile the model
custom_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Function to classify window cleanliness
def classify_window_cleanliness(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = preprocess_input(img)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))

    # Make predictions
    prediction = custom_model.predict(img)
    
    # Threshold the prediction (e.g., if > 0.5, it's clean, otherwise, it's not clean)
    if prediction > 0.5:
        print(prediction)
        return "Clean"
    else:
        print(prediction)
        return "Not Clean"
    
    

# Example usage
image_url = 'https://www.iwantthatdoor.com/wp-content/uploads/2022/04/black-frame-iron-window.jpg'  # Replace with the URL of your window image
image_path, _ = urllib.request.urlretrieve(image_url)
cleanliness_label = classify_window_cleanliness(image_path)
print(f"The window is classified as: {cleanliness_label}")