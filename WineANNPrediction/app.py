import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from keras.layers import Dropout

app = Flask(__name__)
model=Sequential()
model.add(tf.keras.layers.Input(shape = 9,))
model.add(tf.keras.layers.Dense(32,activation='relu'))
model.add(tf.keras.layers.Dense(64,activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(64,activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(6,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics= ['accuracy'])
model.load_weights('model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [features]
    prediction = model.predict(final_features)

    output = np.argmax(prediction)
    dict_wine={0:'Cabernet Sauvignon', 1:'Merlot', 2:'Shiraz', 3:'Pinot Noir', 4:'Malbec', 5:'Zinfandel'}
    wine_type=dict_wine[output]
    return render_template('index.html', prediction_text='Wine of best quality is of type {} ::-- {}'.format((output+1),wine_type))

#2.34,4.25,2.34,0.23,0.64,1.25,0.24,1.023,2.014
if __name__ == "__main__":
    try:
        app.run(port=5000, debug=True)
    except:
        print("Server is exited unexpectedly. Please contact server admin.")

