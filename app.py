from flask import Flask, render_template, request;
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json
import numpy as np


# from keras.models import model_from_json
import cv2

app  = Flask(__name__);

(X_Train, Y_Train) , (X_Test, Y_Test) = keras.datasets.mnist.load_data();

X_Train = X_Train/255
X_Test = X_Test/255
X_Train_Flatten = X_Train.reshape(len(X_Train), 28*28)
X_Test_Flatten = X_Test.reshape(len(X_Test), 28*28)
 
MODEL_ARCHITECTURE = './Model/model.json'   ###
MODEL_WEIGHTS = './Model/model_weights.h5'  ###

#Load Model
json_file = open(MODEL_ARCHITECTURE)
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='rmsprop')
y_predicted = loaded_model.predict(X_Test_Flatten)
prediction = np.argmax(y_predicted[0])

print("Model Loaded")

# Get weights into the model
loaded_model.load_weights(MODEL_WEIGHTS)
print("Weights Loaded")

@app.route('/')
def index():
    return render_template('index.html', data="X-Ray");

@app.route('/prediction', methods=["POST", "GET"])
def prediction():
    if request.method == "POST":
        xrayImage = request.files['xrayImage']
        if xrayImage:            
            xrayImage.save('predict.jpg')
            img = cv2.imread('predict.jpg')
            img = cv2.resize(img, (128,128))
            img = img / 255.0
            img = img.reshape(1, 128,128,3)
            print("hello")
            # prediction = model.predict(img)
            y_predicted = loaded_model.predict(X_Test_Flatten)
            pred = np.argmax(y_predicted[1])
            return render_template('prediction.html', prediction = pred);
    return render_template('prediction.html', prediction = 0);

if __name__ == "__main__":
    app.run(debug=True);