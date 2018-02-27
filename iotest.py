# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 16:22:01 2018

@author: Justin
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import functools
from math import log2 as lg

@functools.lru_cache(maxsize=128) #memoization decorator
def midi_to_hz(n): #converts midi numbers 0-127 to hz, assuming 69=A4=440hz
    dist=n-69
    return 440*2**(dist/12)

def hz_to_midi(n, rnd=False): #inverts above process: returns a midi note number
    if n==0:
        return 0
    try:
        ans=12*(lg(n)-lg(440))+69
        if rnd: #optionally gives an integer value
            ans=round(ans)
        return ans
    except ValueError:
        assert 0, 'valueerror from hz_to_midi on input n: {}'.format(n)

def wav_import(name): #input: e.g., "bass_electronic_018-022-100"
    name=str(name)+".wav"
    base_path="C:\\Users\\Justin\\AppData\\Local\\Programs\\Python\\Python36-32\\spyder 36\\aud\\nsynth-test\\audio\\"
    A=wavfile.read(base_path+name)
    return A #A[0] is sample rate, A[1] is transformed array

def fft_graph(A, window=-1): #feed this an input from wavfile.read. window optional
    if window==-1:
        window=len(A[1])
    """
    rfft is fft for real-valued inputs. rfftfreq gives the sample frequencies.
    the unit of rfftfreq depends on the sample spacing (A[0])
    'For instance, if the sample spacing is in seconds, then the frequency unit is cycles/second.'
    in our instance, the unit is 16000.
    """
    freq=np.fft.rfftfreq(window, 1/A[0])
    spec=np.fft.rfft(A[1][:window]).real #.real quiets a warning about losing imaginary part
    plt.plot(freq, spec) #normalize spectrum by removing the constant freq at spec[0]
    plt.show()
    return (freq, spec)

def data_graph(data, name, window=-1): #just plots on an xy axis
    if len(data)==2:
        data=data[1] #if given a 2-tuple of (type,data), takes the data
    if window==-1:
        window=len(data)
    plt.scatter(list(range(window)),data[:window], marker='.')
    plt.savefig(str(name)+'.png')
    plt.show()

def fft_fund(A, output='hz', rnd=False): #input: A[0] is SR, A[1] is data
    #find the fundamental through fft, no interpolation
    sample_rate, A_data = A #unpack inputs
    clip_time = len(A_data)/sample_rate #e.g., 64000/16000=4 seconds of audio
    spec = np.fft.rfft(A_data).real
    fund_guess = np.argmax(spec)/clip_time
    if output=='hz':
        return fund_guess
    elif output.lower()=='midi':
        freq_midi = hz_to_midi(fund_guess, rnd=rnd)
        return freq_midi
    else:
        assert 0, 'fft_fund output var error: {}'.format(output)
        

def zerox_fund(A, output='hz', rnd=False): #input: A[0] is SR, A[1] is data.
    #approximate frequency by counting rising zero crossings.
    sample_rate, A_data=A
    clip_time = len(A_data)/sample_rate
    positive = A_data > 0
    Z=np.where(np.bitwise_and(np.logical_not(positive[1:]), positive[:-1]))[0]
    fund_guess = len(Z)/clip_time
    if output=='hz':
        return fund_guess
    elif output.lower()=='midi':
        freq_midi = hz_to_midi(fund_guess, rnd=rnd)
        return freq_midi
    else:
        assert 0, 'fft_fund output var error: {}'.format(output)    

def test():
    w=8000
    A=wav_import("bass_electronic_018-022-100")
    data_graph(A, 'A_td', w)
    f, spec=fft_graph(A)
    print(np.argmax(spec)/4)
    print(fft_fund(A), zerox_fund(A), 'fft_fund, zerox_fund')
    #A should be ~~ B0 29.135. but this is a kick drum, so it's harder.
    
    B=wav_import("keyboard_electronic_069-040-075")
    data_graph(B, 'B_td', w)
    f, spec=fft_graph(B,16000)
    print(np.argmax(spec))
    print(fft_fund(B), zerox_fund(B), 'fft_fund, zerox_fund')
    #B should be E2: 82.4 hz
    
    #C should be the same thing-- but cleaner, it's computer-generated and pure.
    C=wavfile.read("C:\\Users\\Justin\\AppData\\Local\\Programs\\Python\\Python36-32\\spyder 36\\aud\\82.4_E2_test.wav")
    data_graph(C, 'C_td', w)
    f, spec=fft_graph(C, 16000)
    print(np.argmax(spec))
    print(fft_fund(C), zerox_fund(C), 'fft_fund, zerox_fund')
    #If we have n seconds of audio, we will have to normalize by multiplying our frequencies by 1/n.
    #E.g., with 4 seconds of audio, we get output: 329 instead of 82.25.
    
    print(midi_to_hz(40))

if __name__=="__main__": #avoids running all this code if i import stuff
    test()