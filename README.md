# Forgery Detection Flask App

## Overview
This is a Flask-based web application designed for forgery detection using image processing techniques. The app preprocesses a dataset of images, compares uploaded images with the dataset, and determines if the uploaded image is forged or genuine. It includes features like dataset preprocessing, image upload, real-time feedback, and cleanup of resources after completion.

## Features
1. **Dataset Display**: Visualize a few sample images from the dataset.
2. **Preprocessing**: Preprocess the dataset to prepare it for forgery detection.
3. **Upload Image**: Upload an image for forgery detection.
4. **Dynamic Feedback**: View preprocessing progress and uploaded images in real-time.
5. **Forgery Detection**: Compare the uploaded image with the dataset and receive detailed results.
6. **Resource Cleanup**: Automatically deletes temporary files (`scores.json`, uploads) after displaying results.
7. **Exit and Restart Options**: Includes options to start over or exit the application after viewing results.

## Prerequisites
- Python 3.8 or higher
- Required Python libraries:
  - Flask
  - OpenCV
  - NumPy
  - threading
  - shutil
  - json
  - base64

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name/forgery-detection-flask-app.git
   ```
2. Navigate to the project directory:
   ```bash
   cd forgery-detection-flask-app
   ```
3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## File Structure
```
forgery-detection-flask-app/
│
├── static/
│   └── styles.css               # Global CSS for consistent styling
│
├── templates/
│   ├── index.html               # Upload dataset path
│   ├── display.html             # Display sample images from the dataset
│   ├── preprocessing.html       # Preprocessing status page
│   ├── upload.html              # Image upload page
│   └── result.html              # Result display page
│
├── uploads/                     # Temporary folder for uploaded images (created dynamically)
├── app.py                       # Flask application logic
├── forgery.py                   # Forgery detection logic
├── thresholding.py              # Dataset preprocessing logic
└── README.md                    # Project documentation
```

---

## Usage

### 1. Start the Application
Run the Flask app:
```bash
python app.py
```
Open your browser and go to [http://127.0.0.1:5000](http://127.0.0.1:5000).

---

### 2. Steps to Use the App
1. **Upload Dataset Path**:
   - Enter the path of the original dataset folder.
2. **View Dataset**:
   - View a few sample images from the dataset.
3. **Preprocess Dataset**:
   - Click "Preprocess Dataset" to prepare the dataset for forgery detection.
   - A progress page will display preprocessing status.
4. **Upload Image**:
   - Upload an image for forgery detection.
   - The uploaded image is displayed on the same page.
5. **Forgery Detection Result**:
   - View a detailed comparison result, including metrics like MSE, SSIM, etc.
   - Options:
     - **Start Again**: Restart the app from the beginning.
     - **Exit**: Close the app or navigate to a thank-you page.

---

## Cleanup
After displaying the result:
- `scores.json` and all files in the `uploads/` folder are automatically deleted.

---

## Notes
- Ensure the dataset folder contains valid images.
- The app supports real-time feedback using JavaScript and AJAX.

---

This README provides a complete overview of the app and should make it easy for others to use and contribute to your project. Let me know if you need any modifications!
