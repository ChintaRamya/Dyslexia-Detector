import os
import glob
import cv2
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, jsonify, Response
from werkzeug.utils import secure_filename
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
REFERENCE_FOLDER = 'reference_images'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for tracking
final_accuracy = None
frame_counter = 0
MAX_FRAMES = 100
eyeball_data = []

def reset_tracking():
    global final_accuracy, frame_counter, eyeball_data
    final_accuracy = None
    frame_counter = 0
    eyeball_data = []

# Load reference eye movement vectors
ref_vectors = []
csv_files = glob.glob('data/*.csv')
for file in csv_files:
    try:
        df = pd.read_csv(file)
        if {'LX', 'LY', 'RX', 'RY'}.issubset(df.columns):
            ref_vectors.append(df[['LX', 'LY', 'RX', 'RY']].values)
    except Exception as e:
        print(f"Error reading {file}: {e}")
ref_vectors = np.vstack(ref_vectors) if ref_vectors else np.empty((0, 4))

# Load VGG16 model
vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
feature_extractor = Model(inputs=vgg.input, outputs=vgg.output)

def extract_features(img):
    img = cv2.resize(img, (224, 224))
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return feature_extractor.predict(img).flatten()

# Load reference image features
reference_features = {}
for fname in os.listdir(REFERENCE_FOLDER):
    fpath = os.path.join(REFERENCE_FOLDER, fname)
    img = cv2.imread(fpath)
    if img is not None:
        reference_features[fname] = extract_features(img)

# Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
cap = cv2.VideoCapture(0)

# Routes
@app.route('/')
def home():
    reset_tracking()
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    reset_tracking()
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username and password:
            return redirect(url_for('dashboard', username=username))
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    reset_tracking()
    username = request.args.get('username', 'User')
    return render_template('dashboard.html', username=username)

@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    reset_tracking()
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            return "No file uploaded"
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img = cv2.imread(filepath)
        if img is None:
            return "Invalid image"

        uploaded_features = extract_features(img)
        best_match, best_distance = None, float('inf')

        for name, ref_feat in reference_features.items():
            dist = np.linalg.norm(uploaded_features - ref_feat)
            if dist < best_distance:
                best_distance = dist
                best_match = name

        similarity = max(0, 1 - best_distance / 2000) * 100
        return redirect(url_for('result_page', match=best_match, score=round(similarity, 2)))

    return render_template('upload.html')

@app.route('/result')
def result_page():
    match = request.args.get('match')
    score = request.args.get('score')
    return render_template('result.html', match=match, score=score)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_accuracy')
def get_accuracy():
    global final_accuracy
    return jsonify({'accuracy': float(f"{final_accuracy:.2f}") if final_accuracy is not None else 0.00})

def generate_frames():
    global final_accuracy, frame_counter, eyeball_data
    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            eyes_detected = []

            for (ex, ey, ew, eh) in eyes[:2]:
                cx = x + ex + ew // 2
                cy = y + ey + eh // 2
                eyes_detected.append((cx, cy))
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

            if len(eyes_detected) == 2 and frame_counter < MAX_FRAMES:
                left, right = sorted(eyes_detected)
                eyeball_data.append([left[0], left[1], right[0], right[1]])
                frame_counter += 1

        if frame_counter >= MAX_FRAMES and final_accuracy is None and len(ref_vectors) > 0:
            input_vector = np.mean(eyeball_data, axis=0)
            distances = np.linalg.norm(ref_vectors - input_vector, axis=1)
            min_d, max_d = np.min(distances), np.max(distances)
            acc = 100 - ((min_d / max_d) * 100) if max_d else 0
            final_accuracy = round(max(0, min(acc, 100)), 2)

        if final_accuracy:
            cv2.putText(frame, f'Accuracy: {final_accuracy:.2f}%', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

if __name__ == '__main__':
    app.run(debug=True)
