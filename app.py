from flask import Flask, render_template, request, redirect
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.efficientnet import preprocess_input
import io

app = Flask(__name__)

# Load models
pet_model = load_model('Animal_Human_model.h5')
emotion_model = load_model('Animal_emotion_model.h5')
human_model = load_model('Human_emotion_model.h5')

# Preprocessing for pet models (RGB images)
def preprocess_pet_image(img):
    img = img.resize((224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# Preprocessing for human model (Grayscale images)
def preprocess_human_image(img):
    img = img.convert('L')
    img = img.resize((48, 48))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array[:, :, :, np.newaxis]
    return preprocess_input(img_array)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            img = Image.open(io.BytesIO(file.read()))

            pet_img_array = preprocess_pet_image(img.copy())
            human_img_array = preprocess_human_image(img.copy())

            # Pet or human prediction
            pet_prediction = pet_model.predict(pet_img_array)
            is_pet = pet_prediction[0][0] < 0.5

            if is_pet:
                # Pet emotion prediction
                emotion_prediction = emotion_model.predict(pet_img_array)
                emotion_class_index = np.argmax(emotion_prediction)
                class_labels = ['Angry', 'Happy', 'Other', 'Sad']
                emotion_label = class_labels[emotion_class_index]
                message = f"The pet is {emotion_label}"
            else:
                # Human emotion prediction
                human_emotion_prediction = human_model.predict(human_img_array)
                human_emotion_class_index = np.argmax(human_emotion_prediction)
                class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
                human_emotion_label = class_labels[human_emotion_class_index]
                message = f"The human is {human_emotion_label}"

            img.seek(0)
            return render_template('index.html', message=message, image=file.stream.read())
    return render_template('index.html', message=None, image=None)

if __name__ == '__main__':
    app.run(debug=True)
