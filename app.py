from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from statistics import mode
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load and preprocess training data
DATA_PATH = "data/Training (1).csv"
data = pd.read_csv(DATA_PATH).dropna(axis=1)

encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Train models
final_svm_model = SVC()
final_nb_model = GaussianNB()
final_rf_model = RandomForestClassifier(random_state=18)
final_svm_model.fit(X, y)
final_nb_model.fit(X, y)
final_rf_model.fit(X, y)

# Create symptom index dictionary
symptoms = X.columns.values
symptom_index = { " ".join([i.capitalize() for i in value.split("_")]): index for index, value in enumerate(symptoms) }

data_dict = {
    "symptom_index": symptom_index,
    "predictions_classes": encoder.classes_
}

# Define prediction function
def make_prediction(symptoms):  
    symptoms = symptoms.split(",")
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        index = data_dict["symptom_index"].get(symptom, None)
        if index is not None:
            input_data[index] = 1
    input_data = np.array(input_data).reshape(1, -1)
    
    rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
    nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
    svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]
    
    final_prediction = mode([rf_prediction, nb_prediction, svm_prediction])
    predictions = {
        "rf_model_prediction": rf_prediction,
        "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": svm_prediction,
        "final_prediction": final_prediction
    }
    return predictions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not all(key in data for key in ['name', 'age', 'gender', 'symptoms']):
        return jsonify({"error": "Missing required fields"}), 400
    
    name = data['name']
    age = data['age']
    gender = data['gender']
    symptoms = data['symptoms']
    
    prediction = make_prediction(symptoms)
    
    response = {
        "name": name,
        "age": age,
        "gender": gender,
        "symptoms": symptoms,
        "prediction": prediction['final_prediction']
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
