from flask import Flask, render_template, request, redirect, send_from_directory, url_for
import numpy as np
import json
import uuid
import tensorflow as tf
import os
import urllib.request

app = Flask(__name__)

# Start with NO brain loaded so Render turns on the website instantly!
model = None 

def load_ai_brain():
    global model
    if model is None:
        print("Loading AI Brain for the first time...")
        model_path = "models/plant_disease_recog_model_pwp.keras"
        os.makedirs("models", exist_ok=True)
        if not os.path.exists(model_path):
            print("Downloading massive 203MB model...")
            url = "https://github.com/mahesh123babu/plant-disease-detection-/releases/download/v1.0/plant_disease_recog_model_pwp.1.keras"
            urllib.request.urlretrieve(url, model_path)
        model = tf.keras.models.load_model(model_path)
        print("Brain loaded successfully!")

# Your original labels
label = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Background_without_leaves', 'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy', 'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

@app.route('/')
def home():
    return render_template('home.html')

def extract_features(image):
    img = tf.keras.utils.load_img(image, target_size=(160,160))
    feature = tf.keras.utils.img_to_array(img)
    feature = np.array([feature])
    return feature

def model_predict(image):
    load_ai_brain() 
    img = extract_features(image)
    prediction = model.predict(img)
    prediction_label = label[prediction.argmax()]
    return prediction_label

@app.route('/upload', methods=['POST', 'GET'])
def uploadimage():
    if request.method == "POST":
        image = request.files['img']
        temp_name = f"uploadimages/{uuid.uuid4().hex}_{image.filename}"
        os.makedirs("uploadimages", exist_ok=True)
        image.save(temp_name)
        prediction = model_predict(temp_name)
        return render_template('home.html', result=True, imagepath=temp_name, prediction=prediction)
    return redirect('/')

@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    return send_from_directory('uploadimages', filename)

if __name__ == "__main__":
    app.run(debug=True)
