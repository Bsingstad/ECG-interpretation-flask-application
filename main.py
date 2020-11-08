

from flask import Flask, url_for, request, render_template, jsonify, flash
import tensorflow as tf
from tensorflow import keras
from werkzeug.utils import secure_filename
import os
from scipy.io import loadmat
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

app = Flask(__name__)
app_dash = dash.Dash(__name__, server=app, url_base_pathname='/pathname/')

global df
df = pd.DataFrame({
    "I": np.ones(5000),
    "II": np.ones(5000),
    "III": np.ones(5000),
    "aVL": np.ones(5000),
    "aVR": np.ones(5000),
    "aVF": np.ones(5000),
    "V1": np.ones(5000),
    "V2": np.ones(5000),
    "V3": np.ones(5000),
    "V4": np.ones(5000),
    "V5": np.ones(5000),
    "V6": np.ones(5000),
    "samples": np.arange(5000)})

fig = px.line(df, x="samples", y="II")

app_dash.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),
    dcc.Dropdown(
        id='dropdown',
        options=[{'label': i, 'value': i} for i in df.columns.unique()[0:-1]],
        value='II'
    ),
    html.Div(id='output'),

    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])
             
@app_dash.callback(Output('example-graph', 'figure'),
              [Input('dropdown', 'value')])
def update_output_1(value):
    # Safely reassign the filter to a new variable
    #df_new = df[value]
    fig = px.line(df, x="samples", y=value)

    fig.update_layout(transition_duration=500)
    return fig




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
        
        df['I'] = reshaped_signal[0].T[0]
        df['II'] = reshaped_signal[0].T[1]
        df['III'] = reshaped_signal[0].T[2]
        df['aVL'] = reshaped_signal[0].T[3]
        df['aVF'] = reshaped_signal[0].T[4]
        df['aVR'] = reshaped_signal[0].T[5]
        df['V1'] = reshaped_signal[0].T[6]
        df['V2'] = reshaped_signal[0].T[7]
        df['V3'] = reshaped_signal[0].T[8]
        df['V4'] = reshaped_signal[0].T[9]
        df['V5'] = reshaped_signal[0].T[10]
        df['V6'] = reshaped_signal[0].T[11]
        result_string = pred_to_labels(prediction)
        result_string = result_string[:-2]
        return render_template('index.html', prediction_text='The predicted diagnoses are: {}'.format(result_string))
        

@app.route('/test', methods=['GET'])
def test():
    return 'Pinging Model Application!!'

@app.route('/dash') 
def render_dashboard():
    return app_dash.index()

@app.route('/vmd_timestamp')
def vmd_timestamp():
return render_template('vmd_timestamp.html')



if __name__ == '__main__':
    #app.run(host="127.0.0.1", port=8080, debug=True)
    app.run()