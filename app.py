from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

model = load_model("model/cattle_model.h5")
classes = ['Gir', 'Sahiwal', 'Murrah', 'Mehsana']

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(224,224))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    breed = classes[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100

    return jsonify({
        "breed": breed,
        "confidence": f"{confidence:.2f}%"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])