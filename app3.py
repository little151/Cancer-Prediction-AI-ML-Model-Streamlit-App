import streamlit as st
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.models import load_model

# Load the trained model
model_path = "trained_model.h5"  # Change this to the path of your trained model file
model = load_model(model_path)

# Define categories
categories = ['Bengin cases', 'Malignant cases', 'Normal cases']

# Function to preprocess the input image
def preprocess_image(image):
    # Resize the image to match the input size of the model
    resized_image = cv2.resize(image, (64, 64))  # Assuming the model was trained with 64x64 images
    # Convert to grayscale
    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    # Normalize the pixel values
    normalized_image = grayscale_image / 255.0
    # Expand dimensions to match the input shape of the model
    processed_image = np.expand_dims(normalized_image, axis=0)
    processed_image = np.expand_dims(processed_image, axis=3)
    return processed_image

# Function to make prediction
def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction

# Streamlit app
def main():
    st.title("Lung Cancer Prediction")
    st.write("Upload a lung image to predict whether it's Bengin, Malignant, or Normal.")

    # File upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Perform prediction when the button is clicked
        if st.button('Predict'):
            # Convert the uploaded image to OpenCV format
            img_array = np.array(image)
            # Perform prediction
            prediction = predict(img_array)
            # Get the predicted class
            predicted_class = categories[np.argmax(prediction)]
            # Display the prediction result
            st.success(f"The predicted class is: {predicted_class}")

if __name__ == '__main__':
    main()
