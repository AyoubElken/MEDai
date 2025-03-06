#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 10:33:17 2024

@author: macbookair
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from PIL import Image
import numpy as np 

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.cnn_model = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
        nn.Tanh(),
        nn.AvgPool2d(kernel_size=2, stride=5),
        nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
        nn.Tanh(),
        nn.AvgPool2d(kernel_size=2, stride=5))

        self.fc_model = nn.Sequential(
        nn.Linear(in_features=256, out_features=120),
        nn.Tanh(),
        nn.Linear(in_features=120, out_features=84),
        nn.Tanh(),
        nn.Linear(in_features=84, out_features=1))

    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc_model(x)
        x = F.sigmoid(x)

        return x
def load_model(model_path):
    model = CNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")  # Ensure the image has 3 channels (RGB)
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    return img_array


def predict(model, img_array):
    tensor_img = torch.tensor(img_array.transpose(2, 0, 1)).unsqueeze(0).float()
    with torch.no_grad():
        prediction = model(tensor_img)
    return prediction


def main():
    st.title("Patient's Brain scan results")

    # Load class labels
    classes = ['Healthy', 'Tumor']


    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    user_name = st.text_input("Patient's name:")

    if uploaded_file is not None:
        model_path = "model_state_dictv1.pth"
        model = load_model(model_path)

        img_array = preprocess_image(uploaded_file)
        prediction = predict(model, img_array)

        # Display prediction results
        pred_class = classes[prediction.argmax()]
        pred_prob = prediction[0][prediction.argmax()].item()

        st.error('CAUTION : HUMAN EXPERT REVISION REQUIRED!') 
        
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        
        
        st.write("Prediction:", pred_class)
        st.write("Probability:", pred_prob)

        
        
        save_results(user_name, uploaded_file.name, pred_class, pred_prob)
      

def save_results(user_name, image_name, pred_class, pred_prob):
    with open("B_scan_results.txt", "a") as file:
        file.write(f"User: {user_name}, Image: {image_name}, Prediction: {pred_class}, Probability: {pred_prob}\n")
if __name__ == "__main__":
    main()

