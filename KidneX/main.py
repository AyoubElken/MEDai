import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load your pre-trained Keras model
def load_keras_model(model_path):
    model = load_model(model_path)
    return model

# Preprocess input image
def preprocess_image(image_path):
    img = Image.open(image_path).resize((224, 224))
    img_array = np.array(img) / 255.0
    return img_array

# Make predictions
def predict(model, img_array):
    # Perform any necessary preprocessing here
    prediction = model.predict(np.expand_dims(img_array, axis=0))
    return prediction

def main():
    st.title("Patient's Kidneys scan results")
    
    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        
        # Load model
        model = load_keras_model("/Users/macbookair/Documents/PRJTS/Kidney_X/KIDNX_model.h5")
        
        # Preprocess and predict
        img_array = preprocess_image(uploaded_file)
        prediction = predict(model, img_array)
        
        # Display prediction
        st.write("Prediction:", prediction)

if __name__ == "__main__":
    main()




