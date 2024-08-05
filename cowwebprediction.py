# -*- coding: utf-8 -*-
"""
Created on Thu May  4 09:42:23 2023

@author: user
"""

import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('C:/Users/ASHWATHI/Desktop/catle/cow.sav', 'rb'))

def disease_prediction(input_data):
    
    
    input_data=(6,148,72,35,0,33.6,0.627)
    input_data_as_numpy_array=np.asarray(input_data)
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
    prediction=loaded_model.predict(input_data_reshaped)
    print(prediction)
    if(prediction[0]==1):
        return 'the cow is healthy'
    else:
      return 'the cow is lumpy'
      
def main():
          
          
          st.title('Cow Health Prediction')
          
          

          Pregnancies=st.text_input("Pregnancies")
          Glucose=st.text_input("Glucose level")
          BloodPressure=st.text_input("BP Value")
          SkinThickness=st.text_input("Skinthickness")      
          Insulin=st.text_input("Insulin level")  
          BMI=st.text_input("BMI")  
          DiabetesPedigreeFunction=st.text_input("DPF") 
          
          diagnosis = ''

          if st.button('Cow Health Test Result'):
              diagnosis = disease_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction])
             
          st.success(diagnosis) 

if __name__=='__main__':
    main()
        
          
          
      
    