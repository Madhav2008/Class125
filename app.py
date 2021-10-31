from flask import Flask, app, jsonify, request
from MVC import get_prediction

app = Flask(__name__)

@app.route("/predict-digit", methods = ["POST"])
def predict_data():
    image = request.files.get("digit")
    prediction = get_prediction(image)
    return jsonify({
        "prediction" : prediction
    }), 200

if __name__ == "__main__":
    app.run(debug = True)