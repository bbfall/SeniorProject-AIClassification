from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
# Load the trained model

model_path = os.path.join('model', 'window_clean_classifier.h5')
image_path = os.path.join('data','clean','windows2.jpg')


model = load_model(model_path)  # Replace with the actual model file name

# Load an image for prediction
img = image.load_img(image_path, target_size=(224, 224))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = img / 255.0

# Make a prediction
prediction = model.predict(img)

if prediction[0][0] > 0.5:
    print("The window is clean.")
else:
    print("The window is dirty.")
