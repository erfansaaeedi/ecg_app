import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import time

# Load the trained model
def load_trained_model():
    model = tf.keras.models.load_model('my_ecg_model.h5')  # Replace with your model path
    return model

# Preprocess the image
def preprocess_image(image):
    # Convert PIL image to numpy array
    image = cv2.resize(image, (512, 512))  # Resize to (512, 512)
    image = image[85:500, 0:512]  # Crop to (415, 512)
    image = image / 255.0  # Normalize pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    return image


# Custom CSS


# Define categories
categories = ['Infarction', 'Non-Infarction']  # Replace with your actual categories

# Streamlit app
def main():
    st.set_page_config(
        page_title="ECG Disease Detection",
        page_icon="🫀",
        layout="centered"
    )

    # Title and description
    st.title("\U00002764 ECG Disease Detection")
    st.markdown("""
    Upload an ECG image, and our AI model will predict whether it is healthy or unhealthy.
    """)
    st.markdown("""
<style>
    .stApp {
        background-color: #f0f2f6;
    }
    h1 {
        color: #2c3e50;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

    # File uploader
    uploaded_file = st.file_uploader("Upload an ECG Image", type=["jpg", "jpeg", "png",'jfif'])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        image = np.array(image)
        st.image(image, caption="Uploaded ECG Image", use_column_width=True)

        # Preprocess the image
        with st.spinner("Processing the image... Please wait 🕒"):
            start_time = time.time()  # Start time
            processed_image = preprocess_image(image)
            model = load_trained_model()
            prediction = model.predict(processed_image)
            end_time = time.time()  # End time
        processing_time = end_time - start_time
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        st.write(prediction)
        predicted_class = categories[predicted_class_index]

        # Display the result
        st.subheader("Prediction:")
        if predicted_class == "Non-Infarction":
            st.success(f"The ECG is **{predicted_class}** ✅")
        else:
            st.error(f"The ECG is **{predicted_class}** ⚠️")

        # Display confidence scores
        st.subheader("Confidence Scores:")
        for i, category in enumerate(categories):
            st.write(f"{category}: {prediction[0][i] * 100:.2f}%")
        st.info(f"Processing Time: {processing_time:.2f} seconds ⏱️")

# Run the app
if __name__ == "__main__":
    main()