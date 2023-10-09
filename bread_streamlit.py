#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import joblib

import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd

from tensorflow.keras.layers import Dense, Flatten, Input, Dropout
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.utils import plot_model
from keras import optimizers

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics

from tensorflow.keras.regularizers import l1, l2, l1_l2

st.header('Artificial Intelligence Prediction of Bread Qualities (Texture)')

uploaded_file = st.file_uploader("Choose an excel file", type="xlsx")

if uploaded_file:
    df1 = pd.read_excel(uploaded_file)

    st.dataframe(df1)
               
    df_train = pd.read_excel("KFRI-108-copy.xlsx", header = 0)
    df_train.dropna(inplace=True)
    df_train1 = df_train.drop(columns=["code",'addition', "Hardness1", "Specific volume", "Clean"])
    scaler = MinMaxScaler()
    df_train1_normal = scaler.fit_transform(df_train1.values)
    
    df1.dropna(inplace=True)
    new_sample1 = df1.drop(columns=["code",'addition', "Hardness1", "Specific volume", "Clean"])
    
    model1 = load_model('bread_hardness.h5')
    
    new_normal = scaler.transform(new_sample1.values)    
    new_x_data = new_normal[:, 0:-1]
    
    result = model1.predict(new_x_data)
    
    new_normal[0][30] = result[0]
    pred = scaler.inverse_transform(new_normal)
    
    st.write('예측되는 빵의 텍스처 (hardness): ', round(pred[0][30], 2), ' N')


# In[ ]:




