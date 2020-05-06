Some work in F0 estimation, using persistent homology and various techniques from signal processing.

This work is part of my senior project from a few years ago. It's hacky "academic" code, far from production quality. I don't have any intent to continue this project, and there's many things that I would do differently now. 


functional_bricabrac.py - miscellaneous functions (including persistence calculations) used by various predictors
wave_predictors.py - predictor objects designed to iterate through the dataset
wavedata.py - two classes, WaveData and LeanWave. The latter is strictly preferred-- it is equivalent for most usage but much more lightweight.
data_retrieval_tools.py - used for various i/o functions
data_wrangling_playground.py - used to store the .csv with all the data.
ML_tests.py - deeply messy file where I ran some test predictions.


Dependencies: Requires pandas, numpy, sklearn, scipy, matplotlib. 

First time setup:

If you have properly cloned the directory, and all dependent libraries are installed, you're good to go. If you would like to run some test predictions, import the relevant predictor from wave_predictors (e.g., from wave_predictors import PersistFFT2). Initialize a prediction object MyPredictor=PersistFFT2(), and invoke its predict_all method (MyPredictor.predict_all()).

If you would like to play around with random data, import LeanWave from wavedata, initialize an object W=LeanWave(), and it will create a random LeanWave object. (This consists of all the relevant data that can be stored sparsely.) You can have a predictor check a single object: MyPredictor.predict(W).

If you would like to play around with the csv file, there should be two options saved in the directory-- one with persistent points listed as MIDI, one with them listed as Hz.

If you would like to be able to graph wave objects or directly access their FFTs, you need to connect it to some saved data-- not included in this repo. To do this, ensure that your working directory contains the nsynth-test folder. Or you may just change the relevant two lines of examples_import in data_retrieval_tools.py to point to the correct location.
