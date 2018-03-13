# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 16:07:06 2018

@author: Justin
"""
from import_test import examples_import, init_df
from iotest import wav_import
from random import randint
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
import numpy as np

adata,anames = examples_import()

def get_example():
    n=randint(0,len(anames))
    example_name=anames[n]
    example_data=wav_import(example_name, precise=True)
    ex_pitch=adata[example_name]['pitch']
    ex_fam=adata[example_name]['instrument_family_str']
    return example_data, ex_pitch, ex_fam

data, pitch, fam = get_example()
N=500

data_win=data[1][500:1000]
con_test=fftconvolve(data_win,data_win[::-1])
con_window=con_test[len(con_test)//2:]

plt.plot(list(range(len(con_test))), con_test)
plt.show()
plt.plot(list(range(len(con_window))), con_window)
plt.show()

print(fft_fund(data, output='midi', rnd=True), 'fft guess')
print(autocorr_fund(data, output='midi', rnd=True), 'autocorr guess')
print(pitch, 'pitch')