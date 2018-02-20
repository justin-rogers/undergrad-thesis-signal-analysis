# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 20:06:48 2018

@author: Justin
"""
import iotest
from nsynth_examples import examples_import

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