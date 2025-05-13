import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load Keras model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")
    return model

# Preprocess uploaded image
def preprocess_image(image, target_size):
    image = image.resize(target_size)
    image = np.array(image).astype(np.float32) / 255.0  # normalize to [0,1]
    image = np.expand_dims(image, axis=0)
    return image

# Predict function
def predict(model, image):
    prediction = model.predict(image)
    return prediction

# Class labels (update if needed)
class_names = [
    'Afghan', 'African Wild Dog', 'Airedale', 'American Hairless', 'American Spaniel',
    'Basenji', 'Basset', 'Beagle', 'Bearded Collie', 'Bermaise',
    'Bichon Frise', 'Blenheim', 'Bloodhound', 'Bluetick', 'Border Collie',
    'Borzoi', 'Boston Terrier', 'Boxer', 'Bull Mastiff', 'Bull Terrier',
    'Bulldog', 'Cairn', 'Chihuahua', 'Chinese Crested', 'Chow',
    'Clumber', 'Cockapoo', 'Cocker', 'Collie', 'Corgi',
    'Coyote', 'Dalmation', 'Dhole', 'Dingo', 'Doberman',
    'Elk Hound', 'French Bulldog', 'German Sheperd', 'Golden Retriever', 'Great Dane',
    'Great Perenees', 'Greyhound', 'Groenendael', 'Irish Spaniel', 'Irish Wolfhound',
    'Japanese Spaniel', 'Komondor', 'Labradoodle', 'Labrador', 'Lhasa',
    'Malinois', 'Maltese', 'Mex Hairless', 'Newfoundland', 'Pekinese',
    'Pit Bull', 'Pomeranian', 'Poodle', 'Pug', 'Rhodesian',
    'Rottweiler', 'Saint Bernard', 'Schnauzer', 'Scotch Terrier', 'Shar_Pei',
    'Shiba Inu', 'Shih-Tzu', 'Siberian Husky', 'Vizsla', 'Yorkie'
]

# Streamlit UI
st.title("🐶 Dog Breed Detection")
st.write("Upload an image of a dog and I'll guess the breed!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image.', use_container_width=True)

    model = load_model()
    input_shape = model.input_shape  # e.g., (None, 224, 224, 3)
    target_size = (input_shape[1], input_shape[2])

    processed_image = preprocess_image(image, target_size)
    prediction = predict(model, processed_image)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.success(f"Predicted Breed: **{predicted_class}** ({confidence*100:.2f}% confidence)")
