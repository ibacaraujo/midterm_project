import pickle

from flask import Flask
from flask import request
from flask import jsonify

model_file = 'model.pkl'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)


app = Flask('income')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    income = y_pred >= 0.5

    result = {
        'income_probability': float(y_pred),
        'income': bool(income)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8885)
