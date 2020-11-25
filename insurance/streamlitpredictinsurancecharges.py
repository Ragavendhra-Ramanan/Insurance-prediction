# -*- coding: utf-8 -*-
import pickle
import numpy as np
import pandas as pd

# load the model from disk
loaded_model = pickle.load(open('Life_insurance_Model.pkl', 'rb'))
scaler = pickle.load(open('scaler_value.pkl', 'rb'))
import streamlit as st
st.title("Predict yearly Insurance charges")
st.image('dataScience.png',use_column_width=True)
"""
Data Science Team

* Anade Davis -Data Science Manager
* Ragavendhra Ramanan -Project Lead
* Ivy Hsu -Data Scientist
* Raques McGill -Data Scientist
* Hafizah ab Rahim -Data Scientist
* Brandon Oppong-Antwi -Data Engineer
"""

# Creating the Titles and Image
st.title("Calculating the insurance charges based on a person's attributes")

# Building dropdown for features sex and smoker.
df = pd.DataFrame({'sex': ['Male','Female'],'smoker': ['Yes', 'No']}) 
    
# Building dropdown for region feature.
df1 = pd.DataFrame({'region' : ['southeast' ,'northwest' ,'southwest' ,'northeast']})
    
# Take the users input
sex = st.selectbox("Select Sex", df['sex'].unique())
smoker = st.selectbox("Are you a smoker", df['smoker'].unique())
region = st.selectbox("Which region do you belong to?", df1['region'].unique())
age = st.slider("What is your age?", 18, 100)
height=st.slider("Your height(in inches)",58,77)
weight=st.slider("Your weight(in pounds)",80,450)
children = st.slider("Number of children", 0, 10)

# converting text input to numeric to get back predictions from backend model.
if sex == 'Male':
    gender = 1
else:
    gender = 0
    
if smoker == 'Yes':
    smoke = 1
else:
    smoke = 0
    
#if region == 'southeast':
#    reg = 2
#elif region == 'northwest':
#    reg = 3
#elif region == 'southwest':
#    reg = 1
#else:
#    reg = 0
    
if region == 'northeast':
    reg=[1,0,0,0]
elif region == 'northwest':
    reg=[0,1,0,0]
elif region == 'southeast':
    reg=[0,0,1,0]
else:
    reg=[0,0,0,1]

    
# store the inputs
weight1=weight*0.454
height1=height*0.0254
bmi= weight1/(height1*height1)
#features = [gender, smoke, reg, age, bmi, children]
features=[age,bmi,children,gender,smoke]+reg

# convert user inputs into an array for the model

int_features = [int(x) for x in features]
final_feature = [np.array(int_features)]
final_features=scaler.transform(final_feature)
print(final_features)
# Final prediction
if st.button('Predict'):           # when the submit button is pressed
    prediction =  loaded_model.predict(final_features)
    st.balloons()
    st.success(f'Your insurance charges per year would be: ${round(prediction[0],2)}')
    
