import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [features]
    prediction = model.predict(final_features)

    output = np.round(prediction[0],2)[0]

    return render_template('index.html', prediction_text='Predicted Salary is {} dollars'.format(output))


if __name__ == "__main__":
    try:
        app.run(port=5000, debug=True)
    except:
        print("Server is exited unexpectedly. Please contact server admin.")

