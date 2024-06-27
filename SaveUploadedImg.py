import streamlit as st
import os

# Function to save the uploaded image to a temporary file and return its path
def save_uploaded_file(uploaded_Img):
    if uploaded_Img is not None:
        # Create a temporary directory if it doesn't exist
        if not os.path.exists("temp"):
            os.makedirs("temp")
        # Save the uploaded file to a temporary location
        file_path = os.path.join("temp", uploaded_Img.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_Img.getbuffer())
        return file_path
    return None
