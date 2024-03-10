from flask import Flask, render_template, request, send_from_directory
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
from translate import Translator

app = Flask(__name__)

# Load the model
model = load_model("best_model.h5")

# Define image size for model input
img_width, img_height = 256, 256

# Define class labels
Class = ['1', '2', 'address', 'afternoon', 'bad', 'drink', 'family', 'good']

# Define upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Check if the 'uploads' directory exists, if not, create it
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

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

    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(upload_path)

    # Preprocess image
    img = preprocess_image(upload_path)

    # Perform prediction
    prediction = model.predict(img)
    predicted_class_index = np.argmax(prediction)
    predicted_class = Class[predicted_class_index]

    # Translate predicted text to Kannada
    translated_text = translate_text(predicted_class)
    return render_template('index.html', prediction=translated_text, uploaded_image=file.filename)

def translate_text(text):
    # Translate text to Kannada
    translator = Translator(to_lang="kn")
    translated_text = translator.translate(text)
    return translated_text

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
