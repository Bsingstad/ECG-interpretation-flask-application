from flask import Flask, url_for, request, render_template, jsonify, flash
import tensorflow as tf
from tensorflow import keras
from werkzeug.utils import secure_filename
import os
from scipy.io import loadmat
import numpy as np

app = Flask(__name__)

def load_challenge_data(filename):
    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)
    #new_file = filename.replace('.mat','.hea')
    #input_header_file = os.path.join(new_file)
    #with open(input_header_file,'r') as f:
    #   header_data=f.readlines()
    return data

def FCN():
    inputlayer = keras.layers.Input(shape=(5000,12)) 

    conv1 = keras.layers.Conv1D(filters=128, kernel_size=8,input_shape=(5000,12), padding='same')(inputlayer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation(activation='relu')(conv1)

    conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)

    conv3 = keras.layers.Conv1D(128, kernel_size=3,padding='same')(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)

    gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)


    outputlayer = keras.layers.Dense(27, activation='sigmoid')(gap_layer)

    model = keras.Model(inputs=inputlayer, outputs=outputlayer)
  


    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=[tf.keras.metrics.BinaryAccuracy(
        name='accuracy', dtype=None, threshold=0.5)])


    return model

def pred_to_labels(score):
    threshold = np.array([0.12585957, 0.09031925, 0.09345833, 0.17864081, 0.11545804,
       0.27795241, 0.1596176 , 0.11184793, 0.16626318, 0.24791257,
       0.1930114 , 0.07277747, 0.05153947, 0.06665818, 0.09982059,
       0.00390505, 0.14655532, 0.19118162, 0.17891057, 0.11025203,
       0.15657453, 0.11539103, 0.1691824 , 0.17392144, 0.17765048,
       0.10066959, 0.08176011])
    binary_prediction = score > threshold
    binary_prediction = binary_prediction * 1
    binary_prediction = np.asarray(binary_prediction)
    
    labels=['pacing rhythm',
         'prolonged qt interval',
         'atrial fibrillation',
         'atrial flutter',
         'left bundle branch block',
         'qwave abnormal',
         't wave abnormal',
         'prolonged pr interval',
         'ventricular premature beats',
         'low qrs voltages',
         '1st degree av block',
         'premature atrial contraction',
         'left axis deviation',
         'sinus bradycardia',
         'bradycardia',
         'sinus rhythm',
         'sinus tachycardia',
         'premature ventricular contractions',
         'sinus arrhythmia',
         'left anterior fascicular block',
         'right axis deviation',
         'right bundle branch block',
         't wave inversion',
         'supraventricular premature beats',
         'nonspecific intraventricular conduction disorder',
         'incomplete right bundle branch block',
         'complete right bundle branch block']
    labels = np.asarray(labels)
    diagnosis=labels[np.where(binary_prediction)[0]]
    result= ''
    for element in diagnosis:
        result += str(element)
        result += ", " 
    return result           

model = FCN()
model.load_weights("assets/fcn_model.h5")



@app.route("/")
def index():
    return render_template("index.html")


@app.route('/', methods=['POST'])
def submit_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save('uploaded_files/'+ secure_filename(f.filename))
        # do prediction
        data = load_challenge_data('uploaded_files/'+ secure_filename(f.filename))
        padded_signal = keras.preprocessing.sequence.pad_sequences(data, maxlen=5000, truncating='post',padding="post")
        reshaped_signal = padded_signal.reshape(1,5000,12)
        prediction = model.predict(reshaped_signal)[0]
        result_string = pred_to_labels(prediction)
        return result_string

@app.route('/test', methods=['GET'])
def test():
    return 'Pinging Model Application!!'

if __name__ == '__main__':
#    app.run(host="127.0.0.1", port=8080, debug=True)
    app.run()