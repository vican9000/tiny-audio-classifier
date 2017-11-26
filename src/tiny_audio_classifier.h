/* Tiny Audio Classifier */
/* Copyright (c) 2017 Ivan Vican */

#ifndef CLASSIFIER_H_
#define CLASSIFIER_H_

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <math.h>
#include <numeric>
#include <vector>
#include "incl\audio_fft\AudioFFT.h"
#include "incl\mfcc\mfcc_calc.h"
#include "incl\tiny_dnn.h"
using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;

class TinyAudioClassifier
{

public:

	float detectorProbability;

	/* Sets the directory where audio clips (for training) are stored */
	int SetAudioSourceDirectory(std::string sourceFolderName);

	/* Sets the output directory where the model is stored in binary format */
	int SetModelOutputDirectory(std::string modelFolderName);

	/* Trains and creates a model based on the audio training corpus */
	int TrainAndCreateModel();

	/* Load model into memory. */
	int LoadSoundClassificationModel(std::string modelName);

	/* Function for processing audio in real-time */
	int ProcessAudio(int16_t* audioSampleBlock, int numberOfSamples, float& detectorProbability);
	int ProcessAudio(float* audioSampleBlock, int numberOfSamples, float& detectorProbability);

	std::vector<float> readAudioFile(const char* filename);

private:

	const uint16_t mainWindowSize = 8192;
	const uint16_t mainWindowHopSize = 4096;
	const uint16_t inputWindowSize = 1024;
	const uint16_t hopSize = 512;
	const uint16_t inputWIndowsInMain = mainWindowSize / hopSize - 1;
	const uint16_t numOfMFCCs = 20;
	const uint16_t featureVecSize = 3 * (3 * (numOfMFCCs - 1)) + inputWIndowsInMain;
	const float preemphasisCoef = 0.95f;

	network<sequential> neural_network;

	std::vector<float> meanVec;
	std::vector<float> stdVec;

	typedef struct {
		std::vector<float> arrayMFCC;
		std::vector<float> deltaMFCC;
		std::vector<float> deltaDeltaMFCC;
		std::vector<float> deltaAuxMFCC;
		std::vector<std::vector<float>> allMFCC;
	} windowMFCC;

	std::string audioSourceDirectory = "C:/source";
	std::string modelOutputDirectory = "C:/recs";

	std::vector<int> readLabelFile(const char* filename);
	std::vector<float> prepareHanning(int hannSize);
	std::vector<float> signalPreEmphHanning(std::vector<float>
		&audioSampleBlock, int numberOfSamples, std::vector<float> &hanning);

	std::vector<float> calcDeltaMFCC(std::vector<float> &inputMFCC, int MFCC_length, int deltaLength);

	float arrayMean(std::vector<float> &inputVec, int arraySize);
	float arrayStd(std::vector<float> &inputVec, int arraySize);
	float arrayKurtosis(std::vector<float> &inputVec, int arraySize);
	void updateAllMFCC(windowMFCC* MFCCs, int numOfMFCCs, int windowsNum);

	std::vector<float> makeFeatureVector(const windowMFCC* MFCCs, int numOfMFCCs, int numberOfWindows, int featureVecSize);
	vec_t processMainWindow(std::vector<float> &audioSampleBlock);
	void trainCreateDNN(std::vector<vec_t> featureMatrix, std::vector<label_t> labelVec);
};

#endif // CLASSIFIER_H_