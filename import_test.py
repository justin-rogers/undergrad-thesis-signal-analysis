# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 20:06:48 2018

@author: Justin
"""
import iotest
import os
import shelve
import matplotlib.pyplot as plt
from scipy.io import wavfile
import json
import pandas as pd

def examples_import():
    jdata=open('./nsynth-test/examples.json').read()
    audtest=json.loads(jdata)
    audionames=list(os.listdir('./nsynth-test/audio'))
    audionames_noext=[x[:-4] for x in audionames]
    return audtest, audionames_noext

""" This works and is fine, but the shelf method now seems unnecessary.
shelfFile=shelve.open('sound_metadata_test')
for name in audionames:
    shelfFile[name] = audtest[name]
shelfFile.close()
"""

def dicts_to_df():
    adata, anames=examples_import()
    adf=pd.DataFrame.from_dict(adata, orient='index') #constructs dataframe, rows labels filenames, columns qualities
    return adf

adata, anames=examples_import()
def accuracy_test():
    fft_hits=0
    zc_hits=0
    total=0
    for filename in anames:
        pitch_target=adata[filename]['pitch'] #int target
        signal=iotest.wav_import(filename)
        fft_guess, zc_guess=iotest.fft_fund(signal), iotest.zerox_fund(signal) #hz guess
        fft_guess, zc_guess=iotest.hz_to_midi(fft_guess, rnd=True), iotest.hz_to_midi(zc_guess, rnd=True) #midi guess
        fft_hits+=(fft_guess==pitch_target) #increment based on accuracy
        zc_hits+=(zc_guess==pitch_target)
        total+=1
    print('fft_hits: {}, zc_hits: {}, total: {}'.format(fft_hits, zc_hits, total))
    print('FFT accuracy: {}'.format(fft_hits/total))
    print('ZC accuracy: {}'.format(zc_hits/total))

    #FFT accuracy: 0.543212890625
    #ZC accuracy: 0.04296875