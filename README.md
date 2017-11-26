# tiny-audio-classifier
Source code (soon with libraries and executables) for Tiny Audio Classifier, a C++ neural network based classifier of any audio signal.
Train it on your own labeled data and use it to predict new data, both offline and in real-time. Feature extraction is based on mel-frequency cepstrum coefficients (MFCC). 
Tiny Audio Classifier works on all platforms, having all 3rd-party libraries included as header-only and MIT or BSD licensed.

Further development is in progress, with a lot of stuff needing improvement:
- implementation of model cross-validation
- better training and prediction interface, with more input file formats, customizable file naming, more sound windowing size and neural network topology options
- broader feature extraction options, with LPC, MPEG-7 and spectrogram extraction implemented.

Make sure to include the "incl" folder to your include path if you are building from source code.

If you want a quick test of the prediction power, run the test executables and choose test 2. The pretrained model for speech recognition (word "queue") with test wave files can be find inside the recs folder.
