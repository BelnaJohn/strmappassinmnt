# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 23:28:15 2020

@author: USER
"""
import streamlit as st
import pandas as pd
import numpy as np
pip install scikit-learn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#import plotly.express as px
#from plotly.subplots import make_subplots
#import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
pickle_in = open('titanicreg.pkl', 'rb')
classifier = pickle.load(pickle_in)
Passengerid= st.number_input("enter your passengerid")
Pclass= st.number_input("enter your passengerclass")
Age= st.number_input("enter your age")
SibSp= st.number_input("enter your number of sibling and spouce travelled with you")
Parch= st.number_input("if you travelled with your children or parents enter their number, if you travelled with both enter the joined count")
Fare= st.number_input("enter your ticket fare")
male= st.number_input("enter your sex,1-male,0-female")
Q= st.number_input("if you started your journey from queenstown enter 1 else enter 0")
S= st.number_input("whether you started your journey from southhampton enter 1 else enter 0")
submit = st.button('Predicts')
if submit:
    prediction = classifier.predict([[Passengerid,Pclass,Age,SibSp,Parch,Fare,male,Q,S]])
    if(prediction==1):
        st.write("Congrats you are survived")
    else:
        st.write("Sorry you seem not to be survived")

