# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 12:43:26 2018

@author: Justin
"""
from data_retrieval_tools import get_names_data, unshelve_lean
from functional_bricabrac import shc_dft_predict, best_guess, midi_to_hz, hz_to_midi

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from collections import Counter
from scipy.signal import fftconvolve
from time import time
from scipy.io import wavfile

style.use('fivethirtyeight')
adata, anames=get_names_data()


try:
    if len(full_dataset)!=len(anames):
        full_dataset=unshelve_lean('full_dataset')
except NameError:
    full_dataset=unshelve_lean('full_dataset')

#we want every class instance to have access to this data
#but if the data was already loaded this session, don't bother with it again.




class BasePredictor:
    """
    Idea: Abstract the base functionality of all my predictors
    (e.g., fft_fund, zerox_fund, etc.) to this class BasePredictor.
    
    Then, for each prediction algorithm, create a subclass that overwrites the
    prediction method for this class.
    
    Desired functionality:
        * Guess an individual WaveData pitch
        * Iteratively guess all pitches in a given collection
        * Graph error bins after doing a large collection of guesses
    """
    predname=None
    def predict(self, wavedata): #throws error if used-- should be overwritten
        #predict should take input a WaveData instance and output an integer (MIDI guess)
        assert 0, 'BasePredictor class should not be used directly, or subclass predict method improperly defined'
    
    def predict_all(self, wavedataset=full_dataset, output_bins=False,
                        graph_error=True, save=False, **kwargs):
        #This returns the error bins for a given list of WaveData objects.
        #(by default the whole directory)
        #If output_bins=False, it will not return the dictionaries.
        total_err_count=0
        total_pitch_cnt, total_fam_cnt=Counter(), Counter()
        rel_pitch_err, abs_pitch_err, fam_err=Counter(), Counter(), Counter()
        
        for wavedata in wavedataset:
            pitch_guess=self.predict(wavedata, **kwargs)
            true_pitch, family=wavedata.metadata['pitch'], wavedata.metadata['instrument_family_str']
            if pitch_guess!=true_pitch: #incorrect guess
                #increment error counters
                pitch_discrepancy=pitch_guess-true_pitch
                total_err_count+=1
                rel_pitch_err[pitch_discrepancy]+=1
                abs_pitch_err[true_pitch]+=1
                fam_err[family]+=1
            total_pitch_cnt[true_pitch]+=1
            total_fam_cnt[family]+=1
        
        accuracy=(len(full_dataset)-total_err_count)/len(full_dataset)
        print('Accuracy for {}: {:.2%}'.format(self.predname, accuracy))
        
        #makes a new dictionary-- percent inaccuracy on each note, instead of number of misclassifications
        pitch_acc_percent =  Counter({note: count/total_pitch_cnt[note] for note, count in abs_pitch_err.items()})
        #similar
        fam_acc_percent =  Counter({family: count/total_fam_cnt[family] for family, count in fam_err.items()})
        
        if graph_error:
            self.graph_rel_pitch(rel_pitch_err, save=save)
            self.graph_abs_pitch(pitch_acc_percent, save=save)
            self.graph_fams(fam_acc_percent, save=save)
        
        if output_bins:
            return (total_pitch_cnt, total_fam_cnt, rel_pitch_err, fam_err,
                    abs_pitch_err, pitch_acc_percent, fam_acc_percent)

    
    def graph_rel_pitch(self, rel_pitch_err, save=False):
        #nice_keys=[x for x in rel_pitch_err.keys() if 1<x<41]
        #nice_keys=list(range(-41,42))
        nice_keys=list(range(-86,86))
        nice_vals=[rel_pitch_err[x] for x in nice_keys]
        
        plt.figure(figsize=(12,12))    
        plt.bar(nice_keys, nice_vals, 1.0, color=[86/255,180/255,233/255])
        plt.xticks([-24,-12, 1, 7, 12, 19, 24, 27])
        plt.ylabel('Error count')
        plt.xlabel('Margin of error')
        plt.title(self.predname + ' relative error')
        plt.tight_layout()
        
        if save:
            plt.savefig('graphs/'+self.predname+'_relative_pitch_error.png')
        plt.show()
    
    def graph_abs_pitch(self, pitch_acc_percent, save=False):
        nice_keys=list(range(21,109))
        nice_vals=[pitch_acc_percent[x] for x in nice_keys]
        
        plt.figure(figsize=(12,12))    
        plt.bar(nice_keys, nice_vals, color=[86/255,180/255,233/255])
        plt.xlabel('MIDI pitch')
        plt.ylabel('Percent missed')
        plt.yticks(np.arange(0,1.1,0.1))
        plt.title(self.predname + ' absolute error')
        plt.tight_layout()
        
        if save:
            plt.savefig('graphs/'+self.predname+'_absolute_pitch_error.png')
        plt.show()
        
    def graph_fams(self, fam_acc_percent, save=False):
        nice_keys=["bass", "brass", "flute", "guitar", "keyboard",
                   "mallet", "organ", "reed", "string", "vocal"]
        nice_vals=[fam_acc_percent[x] for x in nice_keys]
    
        plt.figure(figsize=(12,12))
        plt.bar(range(0,len(nice_keys)), nice_vals, tick_label=nice_keys,
                color=[86/255,180/255,233/255])
        plt.yticks(np.arange(0,1.1,0.1))
        plt.xlabel('Instrument families')
        plt.ylabel('Percent missed')
        plt.title(self.predname + ' family classification error')
        plt.tight_layout()

        if save:
            plt.savefig('graphs/'+self.predname+'_instrument_family_error.png')
        plt.show()

class NaiveFFT(BasePredictor):
    """
    This guesses a signal's fundamental by looking at the single frequency
    with largest amplitude.
    """
    def __init__(self):
        self.predname='NaiveFFT' 
    def predict(self, wave, output='midi'):
        fund_guess=wave.fft_component[1]
        if output=='midi':
            ans=hz_to_midi(fund_guess)
            if ans>108: #dataset constrained between 21 and 108 
                return 108
            if ans<21:
                return 21
            return ans
        if output=='hz':
            return fund_guess

class ZeroX(BasePredictor):
    """
    This provides an approximate guess of a signal's fundamental by counting
    rising zero crossings. Ineffective but computationally cheap.
    """
    def __init__(self):
        self.predname='ZeroX'
    def predict(self, wave, output='midi'):        
        A_data=wave.get_td()
        positive = A_data > 0
        Z=np.where(np.bitwise_and(np.logical_not(positive[1:]), positive[:-1]))[0]
        fund_guess = len(Z)/4
        if output=='hz':
            return fund_guess
        if output=='midi':
            ans=hz_to_midi(fund_guess)
            if ans>108:            
                return 108
            if ans<21:
                return 21
            return ans


class AutoCorr1(BasePredictor):
    """
    Attempts to find frequency by autocorrelation in the time domain.
    This technique makes two copies of the signal and shifts one forward.
    
    When the signals line up very well, hopefully we've shifted by the fundamental.
    So we convert the time shift to a frequency, and that's our guess.
    """
    def __init__(self):
        self.predname='AutoCorr1'
    def predict(self, wave, output='midi', wl=500, wu=1000):        
        A_data=wave.get_td()
        A_window=A_data[wl:wu]
        corr = fftconvolve(A_window, A_window[::-1])
        corr = corr[len(corr)//2:]
        
        diff_approx=np.where([np.diff(corr)>0])
        if len(diff_approx)>0:
            try:
                start=diff_approx[1][1]
            except IndexError:
                print(wave.name)
                start=1
        else:
            start=1
            
        idx=start+np.argmax(corr[start:])+1
        fund_guess=wave.sr/idx
        if output=='hz':
            return fund_guess
        if output=='midi':
            ans=hz_to_midi(fund_guess)
            if ans>108:            
                return 108
            if ans<21:
                return 21
            return ans

class PersistFFT1(BasePredictor):
    """
    Takes the persistence diagram (read: peak information) of the FFT,
    along with a significance threshhold. We take the left-most significant peak
    as our guess.
    
    80.xx% depending on sig parameter, optimally around .15
    """
    def __init__(self, signif=.05):
        self.predname='PersistFFT1'
        self.sig=signif
    def predict(self, wave, output='midi'):
        sig=self.sig
        if sig==.05:
            midi=wave.midi
            fund_guess=midi[0]
        else:
            midi=wave.get_new_dgm_and_midi(sig)[1]
            fund_guess=midi[0]
        if output=='midi':
            return fund_guess
        if output=='hz':
            return midi_to_hz(fund_guess)

class PersistFFT2(BasePredictor):
    """
    Takes the persistence diagram and a significance threshhold.
    
    A list of candidate guesses is generated, and we check them all to see
    which predict the persistence diagram best. Then we take the best guess.
    """
    #87.52% for sig=.05, similar for other sigs
    def __init__(self, signif=.05):
        self.predname='PersistFFT2'
        self.sig=signif
    def predict(self, wave, output='midi'):
        #Finds fundamental by choosing the best candidate component from several.
        if self.sig==.05:
            dgm, midi=wave.dgm, wave.midi
        else:
            dgm, midi=wave.get_new_dgm_and_midi(self.sig)
        fund_guess=best_guess(dgm, midi)
        if output=='midi':
            return fund_guess
        if output=='hz':
            return midi_to_hz(fund_guess)

class PSC(BasePredictor): #persistent spectral correlation
    """
    Something like correlation, but using the DFT instead of the direct signal.
    
    Exploits the fact that we expect frequency peaks to be distributed at integral
    multiples of the fundamental frequency.
    """
    #82.57% w/ signif=.15
    def __init__(self, signif=.05):
        self.predname='PSC'
        self.sig=signif
    def predict(self, wave, output='midi'):
        spec=wave.get_fft()        
        dgm,M=wave.get_new_dgm_and_midi(self.sig)
        fund_guess=shc_dft_predict(spec,M)
        
        if output=='midi':
            return fund_guess
        if output=='hz':
            return midi_to_hz(fund_guess)

if __name__=="__main__":

    """
    classifiers=[NaiveFFT(), ZeroX(), AutoCorr1(), PersistFFT1(.15),
                 PersistFFT2(.15), PSC(.15)]
    for pred in classifiers:
        pred.predict_all(save=True)
    """
    
    for param in np.linspace(.05,.40,36):
        p=PersistFFT2(signif=param)
        p.predict_all(graph_error=False)
    
    """
    start=time()
    psc=PSC()
    mid=time()
    print('psc constructed in {} secs'.format(mid-start))
    psc.predict_all()
    print('finished in {} secs'.format(time()-mid))
    """

    """
    testlist=[PersistFFT2(signif=.15)]
    for n in np.logspace(-10,10,21):
        testlist.append(PersistFFT2(signif=.15, error=n))
    for pred in testlist:
        print('testing with ep={}'.format(pred.ep))
        pred.predict_all(graph_error=False)
    """    

    """
    for param in np.linspace(.10,.25,16):
        p2=PersistFFT2(param)
        p1=PersistFFT1(param)
        ps=PSC(param)
        print(param)
        p1.predict_all(graph_error=False)
        p2.predict_all(graph_error=False)
        ps.predict_all(graph_error=False)
    """