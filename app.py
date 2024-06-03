from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Charger le modèle SVM et le transformateur PCA
svm_model = joblib.load('best_svm_model88.joblib')
pca = joblib.load('pca_transformer.joblib')

# Définir les classes prédéfinies
classes = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# Function to process the image and make predictions
def classify_image(image_bytes):
    img_size = (200, 200)  # Taille des images
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize(img_size)
    img = np.array(img)
    img = img / 255.0  # Normalize the image
    img_flatten = img.flatten().reshape(1, -1)
    img_pca = pca.transform(img_flatten)
    prediction = svm_model.predict(img_pca)
    return prediction[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        img_bytes = file.read()
        prediction = classify_image(img_bytes)
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
