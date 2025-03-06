#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 01:34:53 2024

@author: macbookair
"""

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
def predict(model, img_array, classes):
    # Perform any necessary preprocessing here
    prediction = model.predict(np.expand_dims(img_array, axis=0))
    pred_label = np.argmax(prediction)
    pred_prob = prediction[0][pred_label]
    pred_class = classes[pred_label]
    return pred_class, pred_prob

def main():
    st.title("Patient's Kidneys scan results")
    
    # Load class labels
    classes = ['Cyst', 'Normal', 'Stone', 'Tumor']
    
    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Get user's name
    user_name = st.text_input("Patient's name:")
    
    if uploaded_file is not None and user_name != "":        
        with st.spinner('Classifying...'):
            model = load_keras_model("/Users/macbookair/Documents/PRJTS/Kidney_X/KIDNX_model.h5")
        
        # Preprocess and predict
        img_array = preprocess_image(uploaded_file)
        pred_class, pred_prob = predict(model, img_array, classes)
        
        st.error('CAUTION : HUMAN EXPERT REVISION REQUIRED!') 
        
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        
        
        st.write("Prediction:", pred_class)
        st.write("Probability:", pred_prob)

        
        
        save_results(user_name, uploaded_file.name, pred_class, pred_prob)
      

def save_results(user_name, image_name, pred_class, pred_prob):
    with open("KDN_scan_results.txt", "a") as file:
        file.write(f"User: {user_name}, Image: {image_name}, Prediction: {pred_class}, Probability: {pred_prob}\n")
if __name__ == "__main__":
    main()
