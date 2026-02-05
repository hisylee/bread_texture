#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model


st.header('Artificial Intelligence Prediction of Bread Qualities (Texture)')
st.write('made by Korea Food Research Institute and Sejong Univ.')
st.write(' ')
st.write('(밀가루 Mixolab 특성을 토대로 다층 퍼셉트론 모델로 구축되었으며, 예측 정확도는 R2 > 0.8 입니다)')

@st.cache_resource
def load_model_and_scaler():
    model = load_model("bread_hardness.h5")

    df_train = pd.read_excel("KFRI-108-copy.xlsx")
    df_train.dropna(inplace=True)

    features = df_train.drop(
        columns=["code", "addition", "Hardness1", "Specific volume", "Clean"]
    )

    scaler = MinMaxScaler()
    scaler.fit(features.values)

    return model, scaler

model, scaler = load_model_and_scaler()

uploaded_file = st.file_uploader("Choose an excel file", type="xlsx")

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.dataframe(df)

    df.dropna(inplace=True)
    X_new = df.drop(
        columns=["code", "addition", "Hardness1", "Specific volume", "Clean"]
    )

    X_scaled = scaler.transform(X_new.values)
    X_input = X_scaled[:, :-1]

    y_pred = model.predict(X_input)

    #X_scaled[0, -1] = y_pred[0]
    #X_inverse = scaler.inverse_transform(X_scaled)

    st.success(f"예측 Hardness: {y_pred} N")


#    st.success(f"예측 Hardness: {X_inverse[0, -1]:.2f} N")
    

    


# In[2]:


#pip list --format=freeze > requirements.txt


# In[ ]:




