from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import cv2

app = Flask(__name__)

# Load the model
model = load_model('D:/Skin_Cancer_Detection_Main/model.keras')

# Define lesion types dictionary
lesion_type_dict = {
    0: 'Melanocytic nevi',
    1: 'Melanoma',
    2: 'Benign keratosis-like lesions',
    3: 'Basal cell carcinoma',
    4: 'Actinic keratoses',
    5: 'Vascular lesions',
    6: 'Dermatofibroma'
}

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # Read image file from request
        file = request.files["file"]
        if not file:
            return "No file uploaded", 400

        # Process image
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        image = cv2.resize(image, (100, 100))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ensure image is in RGB format
        image = image / 255.0  # Normalize image
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        
        # Make predictions
        predictions = model.predict(image)
        confidence = np.max(predictions) * 100  # Get the highest confidence percentage
        predicted_class = np.argmax(predictions)
        lesion_type = lesion_type_dict[predicted_class]

        # Pass prediction results to template
        return render_template("index.html", lesion_type=lesion_type, confidence=round(confidence, 2))

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
