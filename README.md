# tiny-audio-classifier
MSVS solution with libraries and executables for Tiny Audio Classifier, a C++ neural network based classifier of any audio signal.
Train it on your own labeled data and use it to predict new data, both offline and in real-time. Feature extraction is based on mel-frequency cepstrum coefficients (MFCC). 
Tiny Audio Classifier works on all platforms, having all 3rd-party libraries included as header-only and MIT or BSD licensed.

Further development is in progress, with a lot of stuff needing improvement:
- implementation of model cross-validation
- better training and prediction interface, with more input file formats, customizable file naming, more sound windowing size and neural network topology options
- broader feature extraction options, with LPC, MPEG-7 and spectrogram extraction implemented.

If you are using Tiny Audio Classifier
