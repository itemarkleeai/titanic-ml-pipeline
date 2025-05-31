from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("rf_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = [
        data.get("Pclass", 3),
        data.get("Sex", 0),
        data.get("Age", 30),
        data.get("SibSp", 0),
        data.get("Parch", 0),
        data.get("Fare", 32),
        data.get("FamilySize", 1)
    ]
    pred = model.predict([features])
    return jsonify({"survived": int(pred[0])})

if __name__ == "__main__":
    app.run(debug=True)
