# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 16:02:24 2018

@author: Justin
"""
from wavedata import WaveData
from wave_predictors import NaiveFFT, ZeroX, PersistFFT1, PersistFFT2, PSHC
import pandas as pd
from data_retrieval_tools import get_names_data, unshelve_lean
from time import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

A=pd.read_csv("4_15_dense_data.csv")
#A=pd.read_csv("4_21_dense_data_midiver.csv")
#y=A.loc[:, 'pitch':'instrument_source']
#y=A.loc[:, 'pitch']
y=A.loc[:, 'instrument_family']
x_with_dgm=A.loc[:, 'NaiveFFT':'peak_49_persistence']
x_dgm_2=x_with_dgm.drop(columns=["fft_mean", "max_fft_amp", "freq_at_max_fft_amp"])
x_dgm_2=x_dgm_2.loc[:, 'NaiveFFT':'peak_3_persistence']
x=A.loc[:, 'NaiveFFT':'PSHC']
"""
If you change the csv at all, change the above lines.
X should be the input for our classifier, Y the output.
"""
x_train, x_test, y_train, y_test=train_test_split(x,y)

lr=LogisticRegression()
lr.fit(x_train, y_train)
print(lr.score(x_test,y_test), 'score for LR')

rf=RandomForestClassifier()
rf.fit(x_train,y_train)
print(rf.score(x_test,y_test), 'score for RF')


x_train, x_test, y_train, y_test=train_test_split(x_with_dgm,y)
rf=RandomForestClassifier()
rf.fit(x_train,y_train)
print(rf.score(x_test,y_test), 'score for RF w dgm')

x_train, x_test, y_train, y_test=train_test_split(x_dgm_2,y)
rf=RandomForestClassifier()
rf.fit(x_train,y_train)
print(rf.score(x_test,y_test), 'score for RF w mindgm')