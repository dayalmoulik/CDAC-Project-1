# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import functools
import os
import shutil
import tensorflow_text as text
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

import streamlit as st
import pandas as pd
import numpy as np
text = [['fool sanghi'],["rahuls priority clear defeat communal force bjp defeat dictator modi save india destruction help indian regain pride rahuls priority clear win 2024 bring back people rule"]]

def labels(text,y_lab):
    label = {0:'Anti-National' , 1:'Critisism' , 2:'Hate'}
    temp = {}
    for i,j in zip(text,y_lab):
        temp["".join(i)] = label[j]
    return temp
#@st.cache(allow_output_mutation=True)
def load_model():
    model_url = 'smallBert/'
    model = tf.keras.models.load_model(model_url)
    return model

def upload_predict(model):
    
    y_out = tf.nn.softmax(model.predict(tf.constant(text)))
    y_lab = np.argmax(y_out.numpy(),axis = 1)
    return y_lab
model = load_model()
y_lab = upload_predict(model)

st.write(labels(text,y_lab))




