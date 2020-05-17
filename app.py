from flask import Flask, request
import pandas as pd 
import numpy as numpy
import pickle

# Starting point of flask application
app = Flask(__name__)

with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Providing decorator
# Will be get method by default --> if no method was provided.
@app.route("/")
def welcome():
    return "Hello World!"

# Running through get method
# passing values through get : http://127.0.0.1:8000/predict?varience=2&skewness=2&curtosis=2&entropy=2
@app.route("/predict") 
def predict_note_authentication():
    varience = request.args.get("varience")
    skewness = request.args.get("skewness")
    curtosis = request.args.get("curtosis")
    entropy = request.args.get("entropy")

    prediction = model.predict([[varience, skewness, curtosis, entropy]])

    return "Predicted value is : " + str(prediction)

@app.route("/predict_file", methods=["POST"])
def predict_note_authentication_file():
    df_test = pd.read_csv(request.files.get("file"))
    y_pred = model.predict(df_test)
    return "Predictions are : " + str(list(y_pred))


if __name__=="__main__":
    app.run(port=8000, debug=True)
