#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the pre-trained model
model = tf.keras.models.load_model('keras_model.h5')

# Load the labels
with open('labels.txt', 'r') as f:
    labels = f.read().splitlines()

# Function to preprocess the image
def preprocess_image(image):
    # Resize the image to the required input shape of the model
    image = image.resize((224, 224))
    # Convert the image to a numpy array
    image = np.array(image)
    # Normalize the image
    image = image / 255.0
    # Add an extra dimension to represent the batch size (required by the model)
    image = np.expand_dims(image, axis=0)
    return image

# Function to make predictions
def predict(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    # Get the index of the predicted class with the highest probability
    predicted_class_index = np.argmax(prediction)
    # Get the corresponding label
    predicted_label = labels[predicted_class_index]
    # Get the confidence score for the predicted class
    confidence = prediction[0][predicted_class_index]
    return predicted_label, confidence

# Streamlit app
def main():
    st.title("Garbage Image Classification")
    st.write("Upload an image and let the model predict the garbage category.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        predicted_label, confidence = predict(image)
        st.write(f"Predicted Label: {predicted_label}")
        st.write(f"Confidence: {confidence:.2f}")

if __name__ == '__main__':
    main()


# In[ ]:




