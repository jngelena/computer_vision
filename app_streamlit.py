import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

# Load models
pet_model = load_model('Animal_Human_model.h5')
emotion_model = load_model('Animal_emotion_model.h5')
human_model = load_model('Human_emotion_model.h5')

# Preprocessing for pet models (RGB images)
def preprocess_pet_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array) 

# Preprocessing for human model (Grayscale images)
def preprocess_human_image(img_path):
    img = Image.open(img_path).convert('L')
    img = img.resize((48, 48))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array[:, :, :, np.newaxis]
    return preprocess_input(img_array)

def main():
    st.title("Pet or Human Emotion Detection")

    image_path = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
    if st.button("Submit") and image_path is not None:
        # Use different preprocessing based on model type
        pet_img_array = preprocess_pet_image(image_path)
        human_img_array = preprocess_human_image(image_path)

        # Pet or human prediction
        pet_prediction = pet_model.predict(pet_img_array)
        is_pet = pet_prediction[0][0] < 0.5

        if is_pet:
            # Pet emotion prediction
            emotion_prediction = emotion_model.predict(pet_img_array)
            emotion_class_index = np.argmax(emotion_prediction)
            class_labels = ['Angry', 'Happy', 'Other', 'Sad']
            emotion_label = class_labels[emotion_class_index]

            st.image(Image.open(image_path), caption=f"The pet is {emotion_label}", use_column_width=True)
        else:
            # Human emotion prediction
            human_emotion_prediction = human_model.predict(human_img_array)
            human_emotion_class_index = np.argmax(human_emotion_prediction)
            class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
            human_emotion_label = class_labels[human_emotion_class_index]

            st.image(Image.open(image_path), caption=f"The human is {human_emotion_label}", use_column_width=True)

if __name__ == "__main__":
    main()
