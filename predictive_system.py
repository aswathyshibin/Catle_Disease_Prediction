# -*- coding: utf-8 -*-
"""
Created on Thu May  4 09:38:35 2023

@author: user
"""


import numpy as np
import pickle

# loading the saved model
loaded_model = pickle.load(open('C:/Users/ASHWATHI/Desktop/catle/cow.sav', 'rb'))


input_data=(6,148,72,35,0,33.6,0.627)
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction=loaded_model.predict(input_data_reshaped)
print(prediction)
if(prediction[0]==1):
  print('the cow is healthy')
else:
  print('the cow is lumpy')