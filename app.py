from flask import Flask, render_template, request, send_from_directory
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import io
from translate import Translator
import base64
from PIL import Image

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

def translate_text(text):
    # Translate text to Kannada
    translator = Translator(to_lang="kn")
    translated_text = translator.translate(text)
    return translated_text

@app.route('/')
def home():
    return render_template('index.html')

# File upload functionality
@app.route('/file')
def file():
    return render_template('file.html')

@app.route('/classify', methods=['POST'])
def classify():
    if 'image' not in request.files:
        return render_template('file.html', prediction=None)
    
    file = request.files['image']
    if file.filename == '':
        return render_template('file.html', prediction=None)

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
    return render_template('result.html', prediction=translated_text, uploaded_image=file.filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Webcam functionality

def preprocess_image_webcam(img):
    img = img.resize((img_width, img_height))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize pixel values
    return img


@app.route('/webcam')
def webcam():
    return render_template('webcam.html')
@app.route('/classifywebcam', methods=['POST'])
def capture_photo():
    img_data = request.form['image']
    img_data_decoded = base64.b64decode(img_data.split(',')[1])
    img = Image.open(io.BytesIO(img_data_decoded))

    # Preprocess the loaded image
    img_numpy = preprocess_image_webcam(img)
    img_numpy = img_numpy.squeeze(axis=0)

    # Perform prediction
    prediction = model.predict(img_numpy[np.newaxis, ...])
    predicted_class_index = np.argmax(prediction)
    predicted_class = Class[predicted_class_index]

    # Translate predicted text to Kannada
    translated_text = translate_text(predicted_class)

    # Convert the image to base64 string
    img_pil = Image.fromarray((img_numpy * 255).astype('uint8'), 'RGB')
    img_buffer = io.BytesIO()
    img_pil.save(img_buffer, format='JPEG')
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    return render_template('webresult.html', prediction=translated_text, uploaded_image=img_base64)

if __name__ == '__main__':
    app.run(debug=True)