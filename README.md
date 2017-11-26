# tiny-audio-classifier
Source code for Tiny Audio Classifier, a neural network based classifier of any audio signal.
Train it on your own labeled data and use it to predict new data, both offline and in real-time. Feature extraction is based on mel-frequency cepstrum coefficients (MFCC).

Further development is in progress, with a lot of stuff needing improvement:
- implementation of model cross-validation
- better training and prediction interface, with more input file formats, customizable file naming, more sound windowing size and neural network topology options
- broader feature extraction options, with LPC, MPEG-7 and spectrogram extraction implemented.
