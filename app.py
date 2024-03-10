from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
from translate import Translator
# Check if the 'temp' directory exists, if not, create it
if not os.path.exists('temp'):
    os.makedirs('temp')
app = Flask(__name__)

# Load the model
model = load_model("best_model.h5")

# Define image size for model input
img_width, img_height = 256, 256

# Define class labels
Class = ['1', '2', 'address', 'afternoon', 'bad', 'drink', 'family', 'good']

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(img_width, img_height))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize pixel values
    return img

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    if 'image' not in request.files:
        return render_template('index.html', prediction=None)
    
    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', prediction=None)

    # Save uploaded image to a temporary folder
    img_path = 'temp/temp.jpg'
    file.save(img_path)

    # Preprocess image
    img = preprocess_image(img_path)

    # Perform prediction
    prediction = model.predict(img)
    predicted_class_index = np.argmax(prediction)
    predicted_class = Class[predicted_class_index]

    # Remove temporary image file
    # os.remove(img_path)
# Translate predicted text to Kannada
    translated_text = translate_text(predicted_class)
    return render_template('index.html', prediction=translated_text, uploaded_image=img_path)

def translate_text(text):
    # Translate text to Kannada
    translator = Translator(to_lang="kn")
    translated_text = translator.translate(text)

    return translated_text
if __name__ == '__main__':
    app.run(debug=True)
