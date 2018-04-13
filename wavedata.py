# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 21:42:25 2018

@author: Justin
"""
from data_retrieval_tools import get_names_data, unshelve_wave
from functional_bricabrac import persist0, calc_midi

from scipy.io import wavfile
from random import randint
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from operator import itemgetter

style.use('fivethirtyeight')
adata, anames=get_names_data()
#we want every class instance to have access to this data

class WaveData:
    """
    A WaveData object consists of all the things we'd like to know about a .wav file.
    In particular, it consists of the data (as a float64 array) and the metadata (e.g., pitch, instrument).
    It also contains the name of the file, frequency bins for graphing, and a copy of the FFT.
    
    You may initialize a WaveData object by passing a name (e.g., WaveData('reed_acoustic_018-086-127')),
    or by passing no parameters-- this chooses a random name in your audio directory. (WaveData()).
    
    A WaveData object W may easily be graphed: W.graph_td() or W.graph_fft().
    """
    
    def __init__(self, name=''):
        if len(name)==0: #pick a random example if given no filename
            n=randint(1,len(anames))
            name=anames[n]
        self.name=name
        #import the data
        self.path="./nsynth-test/audio/"+str(name)+".wav"
        A=wavfile.read(self.path)
        self.sr, data=A[0], A[1].astype('float32') #sample rate, sound data
        self.metadata=adata[name] #JSON data of pitch, etc.
        self.fft=np.absolute(np.fft.rfft(data)).astype('float32') #modulus of rfft for amplitude data
        self.dgm, self.midi=persist0(self.fft, high_pass=True)
        freq=(np.argmax(self.fft()[107:])+107)/4
        self.fft_component=[max(self.fft[107:]), freq] #Amplitude, frequency of biggest part.
    
    def get_new_dgm_and_midi(self, sig):
        #0<sig<1
        #returns a coarser persistence diagram, e.g. throwing away components
        #below 10% of max freq instead of the default of 5%
        if sig==.05:
            return self.dgm, self.midi
        a=sorted(self.dgm, key=itemgetter(1))
        max_amp=a[-1][1] #biggest amplitude in the diagram

        new_dgm=[[a,b] for a,b in self.dgm if b>sig*max_amp]
        new_midi=calc_midi(new_dgm)
        return new_dgm, new_midi
        
    def graph_td(self, save=False, title=None): #graphs time domain
        if title==None:
            title=self.name
        data=self.get_td()
        
        plt.figure(figsize=(12,12))
        plt.plot(list(range(len(data))),data, linewidth=1.0)
        plt.xlabel('Time (samples)')
        plt.ylabel('Amplitude')
        plt.title('Time domain of {}'.format(title))
        plt.tight_layout()
        if save:
            plt.savefig(str(title)+'_td.png')
        plt.show()
        return
        
    def graph_fft(self, save=False, title=None, a=1, b=None):
        data=self.get_fft()
        freq=np.fft.rfftfreq(2*len(data)-1, 1/self.sr)
        if title==None:
            title=self.name

        
        plt.figure(figsize=(12,12))
        plt.plot(freq[a:b], data[a:b], linewidth=1.0)
        plt.xlabel('Hz')
        plt.ylabel('Amplitude')
        plt.title('FFT of {}'.format(title))
        plt.tight_layout()
        if save:
            plt.savefig(str(title)+'_fft.png')
        plt.show()
        
    def get_td(self): #returns data if necessary
        A=wavfile.read(self.path)
        return A[1].astype('float32')
    
    def get_fft(self):
        return self.fft

class LeanWave(WaveData):
    #This class is intended to be a sparser repn of WaveData.
    #It does not include the actual FFT-- just the max frequency and persistence info.
    def __init__(self, name=''):
        WaveData.__init__(self,name)
        self.fft=None #so we avoid accidentally trying to use the fft of a LeanWave. too much data.
        if len(self.dgm)>50:
            self.dgm=sorted(sorted(self.dgm, key=itemgetter(1))[-50:]) #50 biggest parts
            self.midi=calc_midi(self.dgm)
    
    def get_fft(self): #returns a fft if necessary
        data=self.get_td()
        return np.absolute(np.fft.rfft(data)).astype('float32')