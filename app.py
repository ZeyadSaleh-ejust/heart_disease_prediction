from flask import Flask, request, jsonify, render_template
import joblib
from utils import preprocess_input


app = Flask(__name__) # this creates the main object of your web app
try:
    model = joblib.load("models/rf_model.pkl")

except:
    print("Error loading models.")


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    try:

        data = request.get_json() # reads the JSONdata and convert it into dict
        processed_data = preprocess_input(data)

        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0][1]

        return jsonify({'prediction': int(prediction),
                         'probability': float(probability),
                         'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'})



if __name__ == '__main__':
    app.run(debug=True)
