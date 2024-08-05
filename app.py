import streamlit as st
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array, load_img # type: ignore
import numpy as np
import matplotlib.pyplot as plt

model = load_model('D:\Api\save_model.keras')
gender_mapping = {0: 'Male', 1: 'Female'}

def get_image_features(image_path):
    image = load_img(image_path, color_mode='grayscale', target_size=(128, 128))
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

st.title("Age and Gender Prediction")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:

    image = load_img(uploaded_file, color_mode='rgb')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    features = get_image_features(uploaded_file)
    pred = model.predict(features)
    gender = gender_mapping[round(pred[0][0][0])]
    age = round(pred[1][0][0])


    st.write(f'**Predicted Age:** {age}')
    st.write(f'**Predicted Gender:** {gender}')
