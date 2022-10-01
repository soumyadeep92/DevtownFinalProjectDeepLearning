# import files
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import nltk
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input,Embedding,LSTM,Dense,GlobalMaxPooling1D,Flatten
from tensorflow.keras.models import Model
import random
app = Flask(__name__)

with open('content.json',encoding='utf-8') as content:
  data1=json.load(content)
tags=[]
inputs=[]
responses={}
for intent in data1['intents']:
  responses[intent['tag']]=intent['responses']
  for lines in intent['input']:
    inputs.append(lines)
    tags.append(intent['tag'])
data=pd.DataFrame({"inputs":inputs,"tags":tags})
data['inputs_word_count']=data.inputs.str.split().map(lambda x:len(x))
data['inputs_length']=data.inputs.map(lambda x:len(x))
import string
data['punctuation_count']=data['inputs'].map(lambda x:len([c for c in str(x) if c in string.punctuation]))
data['inputs']=data['inputs'].apply(lambda wrd:[ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
data['inputs']=data['inputs'].apply(lambda wrd:''.join(wrd))
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer=Tokenizer(num_words=2000)
tokenizer.fit_on_texts(data['inputs'])
train=tokenizer.texts_to_sequences(data['inputs'])
from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train=pad_sequences(train)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y_train=le.fit_transform(data['tags'])
input_shape=x_train.shape[1]
output_shape=len(y_train)
vocabulary=len(tokenizer.word_index)
output_length=le.classes_.shape[0]
i=Input(shape=(input_shape,))
x=Embedding(vocabulary+1,10)(i)
x=LSTM(10,return_sequences=True)(x)
x=Flatten()(x)
x=Dense(output_length,activation="softmax")(x)
model=Model(i,x)
model.compile(loss="sparse_categorical_crossentropy",optimizer='adam',metrics=['accuracy'])
model.load_weights('model.h5')



@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    texts_p=[]
    prediction_input=[letters.lower() for letters in userText if letters not in string.punctuation]
    prediction_input=''.join(prediction_input)
    texts_p.append(prediction_input)
    prediction_input=tokenizer.texts_to_sequences(texts_p)
    prediction_input=np.array(prediction_input).reshape(-1)
    prediction_input=pad_sequences([prediction_input],input_shape)
    output=model.predict(prediction_input)
    output=output.argmax()
    response_tag=le.inverse_transform([output])[0]
    return str(random.choice(responses[response_tag]))


if __name__ == "__main__":
    app.run()
