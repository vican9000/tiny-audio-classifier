#ifndef MFCC_CALC_H_
#define MFCC_CALC_H_

float melScaleInverse(float f);

float melScale(float f);

void buildFilterBank(float* filterBank);

float filter(float* f, int m, float f_k);

float computeMFFC(float* inputSpectrum, int spectrumSize, int coeff);

#endif