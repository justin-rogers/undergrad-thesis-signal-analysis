# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 20:03:47 2018

@author: Justin
"""
from wavedata import WaveData
from wave_predictors import NaiveFFT, ZeroX, PersistFFT1, PersistFFT2, PSHC
import pandas as pd
from data_retrieval_tools import get_names_data, unshelve_lean
from time import time
from functional_bricabrac import hz_to_midi

def wave_to_series(w, midiver=False):
    """
    Turn a wave object into a pd.Series object, stackable to form a DataFrame.
    """
    meta=[w.metadata['pitch'], w.metadata['instrument_family'], w.metadata['instrument_source']]
    targets=pd.Series(meta, index=['pitch', 'instrument_family', 'instrument_source'])
    
    misc_attributes=pd.Series([w.fft_mean, w.fft_component[0], w.fft_component[1]], 
                              index=['fft_mean', 'max_fft_amp', 'freq_at_max_fft_amp'])
    
    def get_predicted_vals(w):
        labels=['NaiveFFT', 'ZeroX', 'PersistFFT1', 'PersistFFT2', 'PSHC']
        data=[x.predict(w) for x in [NaiveFFT(), ZeroX(), PersistFFT1(), PersistFFT2(), PSHC()]]
        return pd.Series(data, index=labels)
    
    def get_dgm_series(w):
        def pad50(dgm):
            amt_to_pad=(50-len(w.dgm))
            return dgm+[[0,0]]*amt_to_pad
        new_dgm=pad50(w.dgm) #pad dgm with 0s to make it the maximum length (50)
        dgm_data=[]
        dgm_labels=[]
        for i in range(50):
            dgm_data.append(new_dgm[i][0]) #append peak i freq
            dgm_labels.append('peak_{}_freq'.format(i)) #label for peak i
            dgm_data.append(new_dgm[i][1]) #append persistence of peak i
            dgm_labels.append('peak_{}_persistence'.format(i))
        return pd.Series(dgm_data, index=dgm_labels)
    
    def get_dgm_series_as_midi(w):
        def pad50(dgm):
            amt_to_pad=(50-len(w.dgm))
            return dgm+[[0,0]]*amt_to_pad
        new_dgm=pad50(w.dgm) #pad dgm with 0s to make it the maximum length (50)
        dgm_data=[]
        dgm_labels=[]
        for i in range(50):
            dgm_data.append(hz_to_midi(new_dgm[i][0]), rnd=False) #append peak i freq as midi
            dgm_labels.append('peak_{}_freq'.format(i)) #label for peak i
            dgm_data.append(new_dgm[i][1]) #append persistence of peak i
            dgm_labels.append('peak_{}_persistence'.format(i))
        return pd.Series(dgm_data, index=dgm_labels)
    
    pred_srs=get_predicted_vals(w)
    if midiver==True:
        dgm_srs=get_dgm_series_as_midi(w)
    else:
        dgm_srs=get_dgm_series(w)
    final_srs=pd.concat([targets,pred_srs,misc_attributes,dgm_srs])
    return final_srs.rename(w.name)

def get_dense_df():
    adata, anames = get_names_data()
    full_dataset=unshelve_lean('full_dataset')
    full_dataset=[x for x in full_dataset if 21<=x.metadata['pitch']<=108]
    serieses=[]
    
    time1=time()
    for w in full_dataset:
        serieses.append(wave_to_series(w))
    time2=time()
    print('total time to append stuff: {} secs'.format(time2-time1))
    ans=pd.concat(serieses, axis=1).transpose()
    print('concat time: {} secs'.format(time()-time2))
    ans.to_csv('4_15_dense_data.csv')
    print('csv done')
    return ans

def get_dense_df_midi_ver():
    adata, anames = get_names_data()
    full_dataset=unshelve_lean('full_dataset')
    full_dataset=[x for x in full_dataset if 21<=x.metadata['pitch']<=108]
    serieses=[]
    
    time1=time()
    for w in full_dataset:
        serieses.append(wave_to_series(w))
    time2=time()
    print('total time to append stuff: {} secs'.format(time2-time1))
    ans=pd.concat(serieses, axis=1).transpose()
    print('concat time: {} secs'.format(time()-time2))
    ans.to_csv('4_21_dense_data_midiver.csv')
    print('csv done')
    return ans

pretty_data=get_dense_df_midi_ver()
