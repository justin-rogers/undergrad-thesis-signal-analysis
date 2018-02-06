# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 22:26:19 2018

@author: Justin
"""

import os
import shelve
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from subprocess import check_output
import json

def examples_import():
    jdata=open('./nsynth-test/examples.json').read()
    audtest=json.loads(jdata)
    audionames=list(os.listdir('./nsynth-test/audio'))
    return audtest, audionames

""" This works and is fine, but the shelf method now seems unnecessary.
shelfFile=shelve.open('sound_metadata_test')
for name in audionames:
    shelfFile[name] = audtest[name]
shelfFile.close()
"""