# Baconian-semiotics
Some work in F0 estimation, using persistent homology and various techniques from signal processing.

Dependencies: Requires pandas, numpy, sklearn, scipy, matplotlib. All of these are included in the default Anaconda Python 3 library.

First time quick setup:

If you have properly cloned the directory, and all dependent libraries are installed, you're good to go. If you would like to run some test predictions, import the relevant predictor from wave_predictors (e.g., from wave_predictors import PersistFFT2). Initialize a prediction object MyPredictor=PersistFFT2(), and invoke its predict_all method (MyPredictor.predict_all()).

If you would like to play around with random data, import LeanWave from wavedata, initialize an object W=LeanWave(), and it will create a random LeanWave object. (This consists of all the relevant data that can be stored sparsely.) You can have a predictor check a single object: MyPredictor.predict(W).

If you would like to play around with the csv file, there should be two options saved in the directory-- one with persistent points listed as MIDI, one with them listed as Hz.


functional_bricabrac.py - miscellaneous functions (including persistence calculations) used by various predictors
wave_predictors.py - predictor objects designed to iterate through the dataset
wavedata.py - two classes, WaveData and LeanWave. The latter is strictly preferred-- it is equivalent for most usage but much more lightweight.
data_retrieval_tools.py - used for various i/o functions
data_wrangling_playground.py - used to store the .csv with all the data.
ML_tests.py - deeply messy file where I ran some test predictions.

This is sufficient for most purposes, but expect some errors to occur.


If you would like to be able to graph wave objects or directly access their FFTs, you need to connect it to some saved data-- not included in my repo. To do this:

Ensure that your working directory has the nsynth-test folder inside it. Alternatively, change the relevant two lines of examples_import in data_retrieval_tools.py to the correct directories.
