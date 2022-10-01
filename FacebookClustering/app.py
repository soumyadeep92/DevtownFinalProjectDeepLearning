from flask import (
    Flask,
    request,
    render_template,
    url_for
)
import pickle
import numpy as np
from scipy.spatial import distance

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def get_input_values():
    val = request.form['my_form']


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'GET':
        return 'Error in Page.........No field should be left blank...'

    if request.method == 'POST':
        input_val = request.form

        if input_val != None:
            # collecting values
            val1 = float((float(input_val['feature_01'])-2)/(114-2))
            val2 = float((float(input_val['feature_02'])-87)/(2286228-87))
            vals = [val1,val2]
            #print(input_val.keys())
            #for key, value in input_val.items():
                #vals.append(float(value))
            print(vals)

        # Calculate Euclidean distances to freezed centroids
        with open('model.pkl', 'rb') as file:
            freezed_centroids = pickle.load(file)

        assigned_clusters = []
        l = []  # list of distances

        for i, this_segment in enumerate(freezed_centroids):
            dist = distance.euclidean(vals, this_segment)
            l.append(dist)
            index_min = np.argmin(l)
            assigned_clusters.append(index_min)

        return render_template(
            'predict.html', result_value=f'Out of 3 groups, you belong to the Cluster {index_min+1}'
            )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)