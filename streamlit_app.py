
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image

# Load the model
@st.cache_resource
def load_model():
    model_path = "trained_model.pth"  # Update with the correct model file path
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    return model

model = load_model()

# Function to predict using the model
def predict(image_path, model):
    # Add preprocessing logic (resize, normalize, etc.) here
    image = Image.open(image_path)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    # Convert image to tensor, e.g.:
    # input_tensor = preprocess(image).unsqueeze(0)
    # output = model(input_tensor)
    # Add prediction logic here
    return "Prediction Output"

# Streamlit UI
st.title("Brain Tumor Detection")
st.write("Upload an image to detect the presence of a brain tumor.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    prediction = predict(uploaded_file, model)
    st.write(f"Prediction: {prediction}")
