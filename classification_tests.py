# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 00:13:45 2018

@author: Justin
"""
import numpy as np
import iotest
from import_test import examples_import
from import_test import init_df
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')

adata, anames=examples_import()

def bin_data(): #counts how many times a given instrument/pitch occurs
    A=init_df() #data as pandas df
    instrument_counter=Counter()
    pitch_counter=Counter()
    
    for x in set(A['instrument_family_str'].tolist()):
        n=len(A['instrument_family_str'][A['instrument_family_str']==x])
        instrument_counter[x]=n
        
    for x in set(A['pitch'].tolist()):
        n=len(A['pitch'][A['pitch']==x])
        pitch_counter[x]=n
    return instrument_counter, pitch_counter

def pitch_accuracy_test(predictor=iotest.fft_fund, pred_name='', error_chk=False, as_freq=True):
    #feed a predictor, label, & y/n error binning
    #output: dictionaries of where errors occurred in each of 3 categories.
    #(relative pitch error, absolute pitch error, instrument family error)
    hits, misses, trials, total_err = 0,0,0,0
    err_bins=Counter()
    pitch_err=Counter()
    family_err=Counter()
    
    for filename in anames:
        pitch_target = adata[filename]['pitch'] #int target
        signal = iotest.wav_import(filename)
        note_guess = predictor(signal, output='midi', rnd=True)
        
        hits+=(note_guess==pitch_target) #increment based on accuracy
        misses+=(note_guess!=pitch_target)
        res=abs(note_guess-pitch_target) #residual for this case
        total_err+=res
        trials+=1
        
        if error_chk and res>0:
            err_bins[res]+=1 #increase the counter for the residual: i.e., margin by which we misclassified            
            pitch_err[pitch_target]+=1            
            instrument=adata[filename]['instrument_family_str']
            family_err[instrument]+=1

    acc=hits/trials
    avg_err=total_err/misses
    print('Accuracy for predictor {}: {}'.format(pred_name, acc))
    print('Average error: {}. Hits: {}. Misses: {}. Trials: {}.'.format(avg_err, hits, misses, trials))
    
    if error_chk: #note: change to freq
        if as_freq==False:
            print('Top 5 errors: (margin of misclassification) {}'.format(err_bins.most_common()[:5]))
            print('Top 5 errors: (notes) {}'.format(pitch_err.most_common()[:5]))
            print('Top 5 errors: (instruments) {}'.format(family_err.most_common()[:5]))
            return err_bins, pitch_err, family_err
        else:
            all_fams, all_pitch = bin_data()
            #makes a new dictionary-- percent inaccuracy on each note, instead of number of misclassifications
            pitch_acc_percent =  {note: count/all_pitch[note] for note, count in pitch_err.items()}
            #similar
            fam_acc_percent =  {family: count/all_fams[family] for family, count in family_err.items()}
            return err_bins, pitch_acc_percent, fam_acc_percent
    
def error_bin_graph(err_bins, lb=0, ub=41, title='', filename=''): #input: a dictionary of error frequency. saves a graph.
    nice_keys=[x for x in list(err_bins.keys()) if lb<x<ub]
    nice_vals=[err_bins[x] for x in nice_keys]

    plt.figure(figsize=(12,12))    
    plt.bar(nice_keys, nice_vals, 1.0, color=[86/255,180/255,233/255])
    plt.xticks([1, 7, 12, 19, 24, 29, 36])
    plt.ylabel('Error count')
    plt.xlabel('Margin of error')
    plt.title(title)
    plt.tight_layout()
    
    plt.savefig(filename+'_errorbins.png')
    plt.show()
    
def pitch_error_graph(pitch_err, lb=0, ub=127, title='', filename=''):
    nice_keys=[x for x in list(pitch_err.keys()) if lb<x<ub]
    nice_vals=[pitch_err[x] for x in nice_keys]

    plt.figure(figsize=(12,12))    
    plt.bar(nice_keys, nice_vals, color=[86/255,180/255,233/255])
    plt.xlabel('MIDI pitch')
    plt.ylabel('Percent missed')
    plt.yticks(np.arange(0,1.1,0.1))
    plt.title(title)
    plt.tight_layout()
    
    plt.savefig(filename+'_pitcherror.png')
    plt.show()
    
def inst_error_graph(err, title='', filename=''):
    nice_keys=list(err.keys())
    nice_vals=list(err.values())

    plt.figure(figsize=(12,12))
    plt.bar(range(0,len(nice_keys)), nice_vals, tick_label=nice_keys,
            color=[86/255,180/255,233/255])
    plt.yticks(np.arange(0,1.1,0.1))
    plt.xlabel('Instrument families')
    plt.ylabel('Percent missed')
    plt.title(title)
    plt.tight_layout()
    
    plt.savefig(filename+'_famerror.png')
    plt.show()

def graph_err_bins(*args): #give (predictor, name, filename) tuples
    if not args:
        args=((iotest.fft_fund, 'Lone Max Frequency: fft_fund(), 0 fix', 'fft_fund_0fix'), (iotest.zerox_fund, 'Zero Crossing Estimator: zerox_fund(), 0 fix', 'zerox_fund_0fix'))
    #e.g. graph_err_bins((iotest.fft_fund, 'Lone Max Frequency: fft_fund()', 'fft_fund'), (iotest.zerox_fund, 'Zero Crossing Estimator: zerox_fund()', 'zerox_fund'))
    #or, more generally: input can be zip(predictors, labels, filenames)
    #another ex: graph_err_bins((iotest.fft_fund_bmh, 'Max frequency: fft_fund_bmh (BMH window)', 'fft_fund_bmh'))
    #another ex: graph_err_bins((iotest.autocorr_fund, 'autocorr_fund', 'autocorr_fund'))
    for arg in args:
        pred, title, filename=arg
        err_bins, pitch_err, family_err=pitch_accuracy_test(predictor=pred, pred_name=title, error_chk=1)
        error_bin_graph(err_bins, title=title, filename=filename)
        pitch_error_graph(pitch_err, title=title, filename=filename)
        inst_error_graph(family_err, title=title, filename=filename)