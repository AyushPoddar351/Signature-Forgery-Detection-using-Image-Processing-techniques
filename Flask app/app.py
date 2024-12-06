from flask import Flask, render_template, request, redirect, url_for
import os
import threading
import time
import json
import forgery
import thresholding
import cv2
import base64
import shutil

app = Flask(__name__)
UPLOAD_FOLDER = 'Flask app/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

original_dataset_path = None
processing_status = {'status': None}

# Registering a custom filter for base64 encoding
@app.template_filter('b64encode')
def b64encode_filter(data):
    return base64.b64encode(data).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/display-dataset', methods=['POST'])
def display_dataset():
    global original_dataset_path
    original_dataset_path = request.form['dataset_path']
    if not os.path.exists(original_dataset_path):
        return "Path does not exist.", 400
    
    sample_images = get_sample_images(original_dataset_path)
    return render_template('display.html', images=sample_images)


@app.route('/preprocess', methods=['POST'])
def preprocess_dataset():
    if not original_dataset_path:
        return "Original dataset path not found.", 400
    
    processing_status['status'] = "Processing"

    def run_preprocessing():
        thresholding.threshold(original_dataset_path)
        processing_status['status'] = "Finished"

    threading.Thread(target=run_preprocessing).start()
    return redirect(url_for('preprocessing'))


@app.route('/preprocessing')
def preprocessing():
    if processing_status['status'] == "Finished":
        return redirect(url_for('upload_image'))
    return render_template('preprocessing.html')

@app.route('/upload-image')
def upload_image():
    return render_template('upload.html')

# Function to delete scores.json and uploads folder
def cleanup():
    # Delete the scores.json file
    scores_file = 'Flask app/scores.json'
    if os.path.exists(scores_file):
        os.remove(scores_file)

    # Delete the uploads folder and its contents
    if os.path.exists(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)

@app.route('/forgery-check', methods=['POST'])
def forgery_check():
    if not original_dataset_path:
        return "Original dataset path not found.", 400
    
    if 'file' not in request.files:
        return "No file uploaded.", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file.", 400
    
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    result = forgery.forgery(filepath, original_dataset_path)

    sample_images = get_sample_images(original_dataset_path)
    uploaded_image = cv2.imread(filepath)
    _, uploaded_buffer = cv2.imencode('.jpg', uploaded_image)

    cleanup()

    return render_template('result.html', 
                           images=sample_images, 
                           uploaded_image=uploaded_buffer.tobytes(), 
                           result=result)


# Utility function to get sample images
def get_sample_images(path, limit=5):
    images = []
    for filename in os.listdir(path)[:limit]:
        img_path = os.path.join(path, filename)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            _, buffer = cv2.imencode('.jpg', img)
            images.append(buffer.tobytes())
    return images


if __name__ == '__main__':
    app.run(debug=True)
