#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cmath>
#include <assert.h>
#include <vector>

#define F_MIN   0
#define F_MAX   8000

#define PI 3.14159

// Number of mel filters
#define M_NUMCH 26

float melScaleInverse(float f) {
    return 700 * (pow(10, f / 2595.0) - 1);
}

float melScale(float f) {
    return 2595 * log10(1 + (f / 700.0));
}

void buildFilterBank(float* filterBank) {
    int i = 0;
		float numBinFFT = 513;
		float samplingFreq = 16000;
    
		float melDelta = (melScale(F_MAX) - melScale(F_MIN)) / (M_NUMCH);
		float melFFTStep = (numBinFFT - 1) / (samplingFreq / 2);
    for(i = 0; i < M_NUMCH; i++) {
        filterBank[i] = melScaleInverse(melDelta * i);
				filterBank[i] = std::floor(filterBank[i] * melFFTStep);
    }
}

float filter(float* f , int m , float f_k) {
    
    if(f_k < f[m - 1] || f_k >= f[m + 1]) {
        return 0;
    } else if(f[m - 1] <= f_k && f_k < f[m]) {
        return (f_k - f[m - 1]) / (f[m] - f[m - 1]);
    } else if(f[m] <= f_k && f_k < f[m + 1]) {
        return (f_k - f[m + 1]) / (f[m] - f[m + 1]);
    } else {
        return 0;
    }
}

float computeMFFC(float* inputSpectrum, int inputSpectrumSize, int coeff) {
		int i, k;
		static float filterBank[M_NUMCH];

		static int hasInitialized = 0;

		if (!hasInitialized) {
			hasInitialized = 1;
			buildFilterBank(filterBank);
		}
    
		// Take mel-filtered cepstal energy (log(E)) and do DFT for MFCCs
		float c_mf[M_NUMCH] = { 0 };
		float c_mc;
        
        for(i = 1; i < M_NUMCH; i++) {
            c_mf[i] = 0;
            for(k = 0; k < inputSpectrumSize; k++) {
                c_mf[i] += inputSpectrum[k] * inputSpectrum[k] * filter(filterBank, i, k);
            }   
        }
            c_mc = 0;
            
            for(i = 0; i < M_NUMCH; i++) {
                if(c_mf[i] != 0) c_mc += log(c_mf[i]) * cos(coeff * (PI / M_NUMCH) * (i - 0.5));
            }       


    return c_mc;
}
