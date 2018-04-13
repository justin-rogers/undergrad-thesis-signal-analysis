# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 20:06:48 2018

@author: Justin
"""
import os
import shelve
import json
import pandas as pd
from time import time

def examples_import():
    jdata=open('./nsynth-test/examples.json').read()
    audtest=json.loads(jdata)
    audionames=list(os.listdir('./nsynth-test/audio'))
    audionames_noext=[x[:-4] for x in audionames] #removes .wav extension
    return audtest, audionames_noext

def init_df():
    adata, anames=examples_import()
    adf=pd.DataFrame.from_dict(adata, orient='index') #constructs dataframe, rows labels filenames, columns qualities
    return adf

#The following assumes that data has been shelved with wave_shelver.py.
def get_names_data():
    shelfFile=shelve.open('names_data')
    try:
        names=shelfFile['names']
        data=shelfFile['data']
    except:
        print('shelve your names with wave_shelver.py')
        raise
    shelfFile.close()
    return names,data

def unshelve_wave(filename):
    shelfFile=shelve.open('WaveDataShelf')
    W=shelfFile[filename]
    shelfFile.close()
    return W

def unshelve_lean(filename):
    shelfFile=shelve.open('WaveLeanShelf')
    W=shelfFile[filename]
    shelfFile.close()
    return W

def shelve_names():
    shelfFile=shelve.open('names_data')
    anames,adata=examples_import()
    shelfFile['names']=anames
    shelfFile['data']=adata
    shelfFile.close()

def shelve_waves():
    from wavedata import WaveData
    start_time=time()
    names=get_names_data()[0]
    shelfFile=shelve.open('WaveDataShelf')
    for name in names:
        shelfFile[name]=WaveData(name)
    shelfFile.close()
    end_time=time()
    print('waves shelved in {} secs'.format(end_time-start_time))

def shelve_lean_waves():
    from wavedata import LeanWave
    start_time=time()
    names=get_names_data()[0]
    shelfFile=shelve.open('WaveLeanShelf')
    for name in names:
        shelfFile[name]=LeanWave(name)
    shelfFile.close()
    end_time=time()
    print('waves shelved in {} secs'.format(end_time-start_time))

def shelve_lean_waves_as_list():
    from wavedata import LeanWave
    start_time=time()
    names=get_names_data()[0]
    full_set=[]
    for name in names:
        full_set.append(LeanWave(name))
    shelfFile=shelve.open('WaveLeanShelf')
    shelfFile['full_dataset']=full_set
    shelfFile.close()
    end_time=time()
    print('waves shelved in {} secs'.format(end_time-start_time))