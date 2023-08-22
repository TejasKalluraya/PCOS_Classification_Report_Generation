import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from classification_model import build_bayesian_cnn_model, load_and_preprocess_image, get_image_probabilities, plot_image_probabilities

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define the image width and height
img_width, img_height = 256, 256

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')
def load_and_preprocess_image(image_path, img_height, img_width):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (img_height, img_width))
    img_normalized = img_resized / 255.0
    img_array = np.expand_dims(img_normalized, axis=0)
    return img_array
def get_image_probabilities(model, img_array):
    predictions = model.predict(img_array)
    probs = predictions.mean().numpy()[0]
    return probs

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img_width, img_height = 256, 256
        image_array = load_and_preprocess_image(filepath,img_height,img_width)
        report = build_bayesian_cnn_model(filepath,img_width)
        image_array = load_and_preprocess_image(filepath, img_height, img_width)
        model_bayes = build_bayesian_cnn_model(img_height, img_width)
        probabilities = get_image_probabilities(model_bayes, image_array)
        plot_image_probabilities(probabilities)
        return render_template('report.html', report=report, image_filename=filename, probabilities=probabilities)
    
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(f"static/uploads/{filename}", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
