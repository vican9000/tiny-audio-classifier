/* Tiny Audio Classifier */
/* Copyright (c) 2017 Ivan Vican */

#include "tiny_audio_classifier.h"

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

#define PI 3.14159265359


std::vector<int> TinyAudioClassifier::readLabelFile(const char* filename) {

	FILE *fileptr;
	uint8_t *buffer;
	int filelen;

	std::vector<int> tempVec;

	fileptr = fopen(filename, "rb");
	if (fileptr == NULL) {
		exit(1);
	}

	fseek(fileptr, 0, SEEK_END);
	filelen = ftell(fileptr);
	rewind(fileptr);

	buffer = (uint8_t *)malloc((filelen + 1) * sizeof(uint8_t));
	fread(buffer, filelen, 1, fileptr);
	fclose(fileptr);

	tempVec.assign(buffer, buffer + filelen);

	free(buffer);

	return tempVec;
}


std::vector<float> TinyAudioClassifier::readAudioFile(const char* filename) {

	// WAVE file header format
	struct HEADER {
		unsigned char riff[4];						// RIFF string
		unsigned int overall_size;				// overall size of file in bytes
		unsigned char wave[4];						// WAVE string
		unsigned char fmt_chunk_marker[4];			// fmt string with trailing null char
		unsigned int length_of_fmt;					// length of the format data
		unsigned int format_type;					// format type. 1-PCM, 3- IEEE float, 6 - 8bit A law, 7 - 8bit mu law
		unsigned int channels;						// no.of channels
		unsigned int sample_rate;					// sampling rate (blocks per second)
		unsigned int byterate;						// SampleRate * NumChannels * BitsPerSample/8
		unsigned int block_align;					// NumChannels * BitsPerSample/8
		unsigned int bits_per_sample;				// bits per sample, 8- 8bits, 16- 16 bits etc
		unsigned char data_chunk_header[4];		// DATA string or FLLR string
		unsigned int data_size;						// NumSamples * NumChannels * BitsPerSample/8 - size of the next chunk that will be read
	};

	std::vector<float> audioArray;

	// All header specifics are read although not used at the moment
	unsigned char buffer4[4];
	unsigned char buffer2[2];

	FILE *ptr;
	struct HEADER header;

	if (filename == NULL) {
		exit(1);
	}

	// open file
	ptr = fopen(filename, "rb");
	if (ptr == NULL) {
		exit(1);
	}

	int read = 0;

	// read header parts

	read = fread(header.riff, sizeof(header.riff), 1, ptr);

	read = fread(buffer4, sizeof(buffer4), 1, ptr);

	// convert little endian to big endian 4 byte int
	header.overall_size = buffer4[0] |
		(buffer4[1] << 8) |
		(buffer4[2] << 16) |
		(buffer4[3] << 24);

	read = fread(header.wave, sizeof(header.wave), 1, ptr);

	read = fread(header.fmt_chunk_marker, sizeof(header.fmt_chunk_marker), 1, ptr);

	read = fread(buffer4, sizeof(buffer4), 1, ptr);

	// convert little endian to big endian 4 byte integer
	header.length_of_fmt = buffer4[0] |
		(buffer4[1] << 8) |
		(buffer4[2] << 16) |
		(buffer4[3] << 24);

	read = fread(buffer2, sizeof(buffer2), 1, ptr);

	header.format_type = buffer2[0] | (buffer2[1] << 8);

	read = fread(buffer2, sizeof(buffer2), 1, ptr);

	header.channels = buffer2[0] | (buffer2[1] << 8);

	read = fread(buffer4, sizeof(buffer4), 1, ptr);

	header.sample_rate = buffer4[0] |
		(buffer4[1] << 8) |
		(buffer4[2] << 16) |
		(buffer4[3] << 24);

	read = fread(buffer4, sizeof(buffer4), 1, ptr);

	header.byterate = buffer4[0] |
		(buffer4[1] << 8) |
		(buffer4[2] << 16) |
		(buffer4[3] << 24);

	read = fread(buffer2, sizeof(buffer2), 1, ptr);

	header.block_align = buffer2[0] |
		(buffer2[1] << 8);

	read = fread(buffer2, sizeof(buffer2), 1, ptr);

	header.bits_per_sample = buffer2[0] |
		(buffer2[1] << 8);

	read = fread(header.data_chunk_header, sizeof(header.data_chunk_header), 1, ptr);

	read = fread(buffer4, sizeof(buffer4), 1, ptr);

	header.data_size = buffer4[0] |
		(buffer4[1] << 8) |
		(buffer4[2] << 16) |
		(buffer4[3] << 24);

	// calculate no.of samples
	int num_samples = (8 * header.data_size) / (header.channels * header.bits_per_sample);

	// calculate duration of file
	float duration_in_seconds = (float)header.overall_size / header.byterate;

	// read each sample from data chunk if PCM
	if (header.format_type == 1) {

		header.channels = 1;
		char data_buffer[2] = { 0 };

		printf("Reading file..\n");

		for (int i = 1; i <= num_samples; i++) {
			read = fread(data_buffer, sizeof(data_buffer), 1, ptr);

			if (read == 1) {

				int data_in_channel = 0;

				// convert data from little endian to big endian based on bytes in each channel sample
				data_in_channel = (data_buffer[0] & 0xff) |
					(data_buffer[1] << 8);

				audioArray.push_back((float)data_in_channel / (float)pow(2, 15));

			}
			else {
				printf("Error reading file. %d bytes\n", read);
				break;
			}
		}
	}

	// read each sample from data chunk if float
	if (header.format_type == 3) {

		header.channels = 1;
		char data_buffer[4];

		printf("Reading file..\n");

		for (int i = 1; i <= num_samples; i++) {
			read = fread(data_buffer, sizeof(data_buffer), 1, ptr);

			if (read == 1) {

				uint32_t data_in_channel = 0;
				float f;

				// convert float data from little endian to big endian based on bytes in each channel sample
				data_in_channel = ((data_buffer[1] << 24) |
					((data_buffer[0] & 0xff) << 16) |
					((data_buffer[3] & 0xff) << 8) |
					(data_buffer[2] & 0xff)); //parsing fine on available examples but should be tested more

				memcpy(&f, &data_in_channel, 4);
				audioArray.push_back(f);

			}
			else {
				printf("Error reading file. %d bytes\n", read);
				break;
			}
		}
	}

	printf("Closing file..\n");
	fclose(ptr);

	return(audioArray);

}

// Prepare Hanning window
std::vector<float> TinyAudioClassifier::prepareHanning(int hannSize) {

	std::vector<float> tempHanning;

	for (int i = 0; i < hannSize; i++)
		tempHanning.push_back(0.5f - 0.5f * (float)cos(2.0f * PI * i / (hannSize - 1)));

	return tempHanning;
}

// Apply Hanning window and pre-emphasis for MFCC extraction
std::vector<float> TinyAudioClassifier::signalPreEmphHanning(std::vector<float> &audioSampleBlock, int numberOfSamples, std::vector<float> &hanning) {

	std::vector<float> emphasizedSampleBlock(numberOfSamples, 0);
	emphasizedSampleBlock[0] = hanning[0] * audioSampleBlock[0];
	for (int i = 1; i < numberOfSamples; i++)
		emphasizedSampleBlock[i] = hanning[i] * (audioSampleBlock[i] - preemphasisCoef * audioSampleBlock[i - 1]);

	return emphasizedSampleBlock;
}

// Calculate deltas (and delta-deltas) of MFCCs with moving average filters
std::vector<float> TinyAudioClassifier::calcDeltaMFCC(std::vector<float> &inputMFCC, int MFCC_length, int deltaLength) {

	std::vector<int> filter(2 * deltaLength + 1, 0);
	std::vector<float> tempMFCC(MFCC_length + 2 * deltaLength + 1, 0);
	std::vector<float> outputMFCC(MFCC_length, 0);

	for (int k = -deltaLength; k <= deltaLength; k++) {
		filter[k + deltaLength] = k;
	}

	for (int i = 0; i <= MFCC_length + 2 * deltaLength; i++) {
		if (i < deltaLength) {
			tempMFCC[i] = inputMFCC[0];
		}
		else if (i >= MFCC_length + deltaLength) {
			tempMFCC[i] = inputMFCC[MFCC_length - 1];
		}
		else {
			tempMFCC[i] = inputMFCC[i - deltaLength];
		}
	}

	for (int arrayIndex = 0; arrayIndex < MFCC_length; arrayIndex++) {
		for (int filterIndex = 0; filterIndex < 2 * deltaLength + 1; filterIndex++) {
			outputMFCC[arrayIndex] += tempMFCC[arrayIndex + filterIndex] * filter[filterIndex];
		}
	}

	return outputMFCC;
}

float TinyAudioClassifier::arrayMean(std::vector<float> &inputVec, int arraySize) {

	float meanSum = 0;
	for (int i = 0; i < arraySize; i++)
		meanSum += inputVec[i];

	return meanSum / arraySize;
}

float TinyAudioClassifier::arrayStd(std::vector<float> &inputVec, int arraySize) {

	float mean = arrayMean(inputVec, arraySize);
	float variance = 0;

	for (int i = 0; i < arraySize; i++) {
		variance += (inputVec[i] - mean) * (inputVec[i] - mean);
	}

	variance /= arraySize;
	return sqrt(variance);
}

float TinyAudioClassifier::arrayKurtosis(std::vector<float> &inputVec, int arraySize) {

	std::vector<float> devQuadArray(arraySize, 0);

	float mean = arrayMean(inputVec, arraySize);
	float stdDev = arrayStd(inputVec, arraySize);

	for (int i = 0; i < arraySize; i++) {
		devQuadArray[i] = pow(inputVec[i] - mean, 4);
	}

	float meanDevQuadArray = arrayMean(devQuadArray, arraySize);

	return meanDevQuadArray / pow(stdDev, 4);
}

// Add subwindows MFCCs row to AllMFCC matrix
void TinyAudioClassifier::updateAllMFCC(windowMFCC* MFCCs, int numOfMFCCs, int windowsNum) {

	std::vector<float> tempVec(3 * numOfMFCCs, 0);

	for (int i = 0; i < numOfMFCCs; i++) {
		tempVec[i] = MFCCs->arrayMFCC[i];
		tempVec[i + numOfMFCCs] = MFCCs->deltaMFCC[i];
		tempVec[i + 2 * numOfMFCCs] = MFCCs->deltaDeltaMFCC[i];
	}

	MFCCs->allMFCC.push_back(tempVec);

}

// Calculate feature vector: means, stddevs, kurtosis statistics through distinct MFCCs in subwindows + first MFCCs (cepstral energies)
std::vector<float> TinyAudioClassifier::makeFeatureVector(const windowMFCC* MFCCs, int numOfMFCCs, int numberOfWindows, int featureVecSize) {

	/*std::vector<float> tempVecMFCC(numberOfWindows, 0);
	std::vector<float> tempFeatureVec;
	tempFeatureVec.reserve(3 * numOfMFCCs * inputWIndowsInMain);

	for (int MFCCPos = 0; MFCCPos < 3 * numOfMFCCs; MFCCPos++) {
		for (int winNum = 0; winNum < numberOfWindows; winNum++) {
			tempFeatureVec.push_back(MFCCs->allMFCC[winNum][MFCCPos]);
		}
	}*/

	std::vector<float> tempVecMFCC(numberOfWindows, 0);
	std::vector<float> tempFeatureVec;
	tempFeatureVec.reserve(featureVecSize);

	// Mean
	for (int MFCCPos = 1; MFCCPos < 3 * numOfMFCCs; MFCCPos++) {
		for (int winNum = 0; winNum < numberOfWindows; winNum++) {
			tempVecMFCC[winNum] = MFCCs->allMFCC[winNum][MFCCPos];
		}
		if (MFCCPos % numOfMFCCs) {
			tempFeatureVec.push_back(arrayMean(tempVecMFCC, numberOfWindows));
		}
	}

	// Standard Deviation
	for (int MFCCPos = 1; MFCCPos < 3 * numOfMFCCs; MFCCPos++) {
		for (int winNum = 0; winNum < numberOfWindows; winNum++) {
			tempVecMFCC[winNum] = MFCCs->allMFCC[winNum][MFCCPos];
		}
		if (MFCCPos % numOfMFCCs) {
			tempFeatureVec.push_back(arrayStd(tempVecMFCC, numberOfWindows));
		}
	}

	// Kurtosis
	for (int MFCCPos = 1; MFCCPos < 3 * numOfMFCCs; MFCCPos++) {
		for (int winNum = 0; winNum < numberOfWindows; winNum++) {
			tempVecMFCC[winNum] = MFCCs->allMFCC[winNum][MFCCPos];
		}
		if (MFCCPos % numOfMFCCs) {
			tempFeatureVec.push_back(arrayKurtosis(tempVecMFCC, numberOfWindows));
		}
	}

	// First MFCCs (Energy-based)
	for (int winNum = 0; winNum < numberOfWindows; winNum++) {
		tempFeatureVec.push_back(MFCCs->allMFCC[winNum][0]);
	}

	return tempFeatureVec;

}

vec_t TinyAudioClassifier::processMainWindow(std::vector<float> &audioSampleBlock) {

	static int has_initialized = 0;
	static windowMFCC MFCCs;
	std::vector<float> featureVector(featureVecSize);

	static std::vector<float> hanningWin(inputWindowSize);

	// Allocate FFT vectors
	std::vector<float> inputWindowFFTOut(audiofft::AudioFFT::ComplexSize(inputWindowSize));
	std::vector<float> inputWindowFFTRe(audiofft::AudioFFT::ComplexSize(inputWindowSize));
	std::vector<float> inputWindowFFTIm(audiofft::AudioFFT::ComplexSize(inputWindowSize));

	audiofft::AudioFFT fft;
	fft.init(inputWindowSize);

	// Initialize hanning, reserve space for vectors
	if (!has_initialized) {
		has_initialized = 1;

		hanningWin = prepareHanning(inputWindowSize);

		MFCCs.arrayMFCC.reserve(numOfMFCCs);
		MFCCs.deltaMFCC.reserve(numOfMFCCs);;
		MFCCs.deltaDeltaMFCC.reserve(numOfMFCCs);;
		MFCCs.deltaAuxMFCC.reserve(numOfMFCCs);;

	}

	MFCCs.allMFCC.clear();

	// Move through subwindows: take around 0.5 sec of audio and hop for around 0.25 sec
	for (int pos = 0; pos <= mainWindowSize - inputWindowSize; pos += hopSize) {

		MFCCs.arrayMFCC.clear();

		// Take audio subwindow
		std::vector<float> inputWindow(audioSampleBlock.begin() + pos, audioSampleBlock.begin() + pos + inputWindowSize);

		// "Hanning"-ize and preemphasize
		inputWindow = signalPreEmphHanning(inputWindow, inputWindowSize, hanningWin);

		// Calculate FFT
		fft.fft(inputWindow.data(), inputWindowFFTRe.data(), inputWindowFFTIm.data());

		// Get FFT magnitude
		for (uint32_t i = 0; i < inputWindowFFTRe.size(); i++) {
			inputWindowFFTOut[i] = sqrt(inputWindowFFTRe[i] * inputWindowFFTRe[i] + inputWindowFFTIm[i] * inputWindowFFTIm[i]);
		};

		// Get first n MFCCs (13 or 20 are normally used)
		for (int coeff = 0; coeff < numOfMFCCs; coeff++) {
			MFCCs.arrayMFCC.push_back(computeMFFC(&inputWindowFFTOut[0], inputWindowFFTOut.size(), coeff));
		}

		// Get MFCC deltas and delta-deltas
		MFCCs.deltaMFCC = calcDeltaMFCC(MFCCs.arrayMFCC, numOfMFCCs, 4);
		MFCCs.deltaAuxMFCC = calcDeltaMFCC(MFCCs.arrayMFCC, numOfMFCCs, 2);
		MFCCs.deltaDeltaMFCC = calcDeltaMFCC(MFCCs.deltaAuxMFCC, numOfMFCCs, 2);

		// Add row to AllMFCC
		updateAllMFCC(&MFCCs, numOfMFCCs, inputWIndowsInMain);

	}

	// Calculate statistics and make feature vector
	featureVector = makeFeatureVector(&MFCCs, numOfMFCCs, inputWIndowsInMain, featureVecSize);

	vec_t featureVectorDNN;

	for (int i = 0; i < featureVecSize; i++) {
		featureVectorDNN.push_back(featureVector[i]);
	}

	return featureVectorDNN;
}

// Train and create deep neural network (in this case, a multi-layer perceptron with 2 hidden layers)
void TinyAudioClassifier::trainCreateDNN(std::vector<vec_t> featureMatrix, std::vector<label_t> labelVec) {

	// Number of neurons in two hidden layers
	const size_t num_hidden_units_1 = 400;
	const size_t num_hidden_units_2 = 50;

	// Make MLP classifier with tanh activation 
	//auto nn = make_mlp<tanh_layer>({ featureVecSize, num_hidden_units_1, num_hidden_units_2, 2 });



	size_t input_dim = featureVecSize;
	size_t hidden_units_1 = 400;
	size_t hidden_units_2 = 50;
	size_t output_dim = 2;

	fully_connected_layer f1(input_dim, hidden_units_1);
	tanh_layer th1(hidden_units_1);
	dropout_layer dropout1(hidden_units_1, 0.5);
	fully_connected_layer f2(hidden_units_1, hidden_units_2);
	tanh_layer th2(hidden_units_2);
	dropout_layer dropout2(hidden_units_2, 0.5);
	fully_connected_layer f3(hidden_units_2, output_dim);
	tanh_layer th3(output_dim);
	using network = network<sequential>;
	network nn;
	nn << f1 << th1 << dropout1 << f2 << th2 << dropout2 << f3 << th3;



	// Define optimizer (adam, adagrad and many more are available)
	gradient_descent optimizer;

	// Allocate train and test matrices 
	std::vector<label_t> train_labels, test_labels;
	std::vector<vec_t> train_images, test_images;

	// Split the set: 85% for train and 15% for test. For large sets this can be done, in smaller ones cross-validation should be used
	float dataset_size = (float)featureMatrix.size();
	int train_size = (int)std::floor(dataset_size * 85 / 100);

	train_images.assign(&featureMatrix[0], &featureMatrix[train_size]);
	train_labels.assign(&labelVec[0], &labelVec[train_size]);

	test_images.assign(&featureMatrix[train_size + 1], &featureMatrix[(int)dataset_size - 1]);
	test_labels.assign(&labelVec[train_size + 1], &labelVec[(int)dataset_size - 1]);

	optimizer.alpha = 0.001f;

	progress_display disp(featureMatrix.size());
	timer t;

	int epoch_count = 1;

	// Create callback through epochs
	auto on_enumerate_epoch = [&]() {
		std::cout << t.elapsed() << "s elapsed." << std::endl;

		tiny_dnn::result res = nn.test(test_images, test_labels);
		tiny_dnn::result res2 = nn.test(train_images, train_labels);

		std::cout << optimizer.alpha << ", Test dataset precision: " << res.num_success << "/"
			<< res.num_total << " Train dataset precision: " << res2.num_success << "/" << res2.num_total
			<< " Epoch num: " << epoch_count << std::endl;

		epoch_count++;
		
		// Decay optimizer learning rate
		optimizer.alpha *= 0.9f;
		optimizer.alpha = std::max((tiny_dnn::float_t)0.00001, optimizer.alpha);

		disp.restart(featureMatrix.size());
		t.restart();
	};

	auto on_enumerate_data = [&]() { ++disp; };

	// Train on chosen set, with batch size of 20 and 25 epochs 
	nn.train<mse>(optimizer, train_images, train_labels, 20, 5, on_enumerate_data,
		on_enumerate_epoch);

	tiny_dnn::result res = nn.test(test_images, test_labels);
	tiny_dnn::result res2 = nn.test(train_images, train_labels);

	std::cout << optimizer.alpha << ", Test dataset precision: " << res.num_success << "/"
		<< res.num_total << " Train dataset precision: " << res2.num_success << "/" << res2.num_total
		<< " Epoch num: " << epoch_count << std::endl;

	// Save model or prompt again if the location is not correct
	try {
		nn.save(modelOutputDirectory + "/model");
	}
	catch (const tiny_dnn::nn_error) {
		std::cerr << "Model save directory is non-existent or unavailable! \n";
		printf("Set model save directory: ");
		std::getline(std::cin, modelOutputDirectory);
		nn.save(modelOutputDirectory + "/model");
	}

};


int TinyAudioClassifier::SetModelOutputDirectory(std::string modelFolderName) {

	modelOutputDirectory = modelFolderName;

	return 0;
}


int TinyAudioClassifier::SetAudioSourceDirectory(std::string sourceFolderName) {

	audioSourceDirectory = sourceFolderName;

	return 0;
}

int TinyAudioClassifier::LoadSoundClassificationModel(std::string modelPath) {

	neural_network.load(modelPath);

	meanVec.resize(featureVecSize);
	stdVec.resize(featureVecSize);

	std::string meanVecPath = modelPath + "_mean.bin";
	std::ifstream file1(meanVecPath.c_str(), std::ios::in | std::ifstream::binary);
	file1.read(reinterpret_cast<char*>(meanVec.data()), sizeof(float) * featureVecSize);
	file1.close();

	std::string stdVecPath = modelPath + "_std.bin";
	std::ifstream file2(stdVecPath.c_str(), std::ios::in | std::ifstream::binary);
	file2.read(reinterpret_cast<char*>(stdVec.data()), sizeof(float) * featureVecSize);
	file2.close();

	return 0;
}

// Train and create model made from the recordings in source path
int TinyAudioClassifier::TrainAndCreateModel() {

	int audioSampleNum = 1;
	std::vector<float> audioArray;
	std::vector<int> labelArray;

	vec_t featureVec;
	std::vector<label_t> labelVec;
	std::vector<vec_t> featureMatrix;

	// Iterate through recordings and label files 
	while (1) {

		FILE* file;

		// Get recordings and labels names
		std::string fileNameWav = audioSourceDirectory + "/Test" + std::to_string(audioSampleNum) + "_mixture.wav";
		std::string fileNameLabel = audioSourceDirectory + "/Test" + std::to_string(audioSampleNum) + "_label.lab";
		file = fopen(fileNameWav.c_str(), "r");

		if (file == NULL)
			break;

		fclose(file); // clearing for readAudioFile

									// Get recordings and labels data
		audioArray = readAudioFile(fileNameWav.c_str());
		labelArray = readLabelFile(fileNameLabel.c_str());

		printf("Reading file %s\n", fileNameWav.c_str());

		static int kk = 0;

		// Crop the recordings for balanced classification
		int centerBoundary = 0, lowerBoundary = 0, upperBoundary = 0;
		for (uint32_t i = 0; i < labelArray.size(); i++) {
			if (!centerBoundary && labelArray[i])
				centerBoundary = i;
			if (labelArray[i])
				upperBoundary = i;
		}

		upperBoundary = centerBoundary + (int)std::floor(0.95 * (upperBoundary - centerBoundary));
		lowerBoundary = (int)std::max(0, 2 * centerBoundary - upperBoundary);

		// If target sound is not present in file, include all
		if (upperBoundary == centerBoundary) {
			upperBoundary = 0.95 * labelArray.size();
			lowerBoundary = 0;
		}

		// Process audio and label files, get dataset
		for (int wavPosition = lowerBoundary; wavPosition < upperBoundary - mainWindowSize; wavPosition += mainWindowHopSize) {
			std::vector<float> audioSubArray(&audioArray[wavPosition], &audioArray[wavPosition + mainWindowSize]);
			std::vector<int> labelSubArray(&labelArray[wavPosition], &labelArray[wavPosition + mainWindowSize]);

			featureVec = processMainWindow(audioSubArray);

			int windowLabel = std::accumulate(labelSubArray.begin(), labelSubArray.end(), 0) > mainWindowSize / 2;
			labelVec.push_back(windowLabel);

			featureMatrix.push_back(featureVec);

		}

		audioSampleNum++;

	}

	// Feature scaling
	std::vector<float> tempColumnVec;
	meanVec.clear();
	stdVec.clear();
	std::vector<float> stdVec;
	for (int column = 0; column < featureVecSize; column++) {
		tempColumnVec.clear();
		for (int row = 0; row < featureMatrix.size(); row++) {
			tempColumnVec.push_back(featureMatrix[row][column]);
		}
		float columnMean = arrayMean(tempColumnVec, tempColumnVec.size());
		float columnStdDev = arrayStd(tempColumnVec, tempColumnVec.size());
		meanVec.push_back(columnMean);
		stdVec.push_back(columnStdDev);
		if (!columnStdDev)
			columnStdDev = 0.001;
		for (int row = 0; row < featureMatrix.size(); row++) {
			featureMatrix[row][column] = (featureMatrix[row][column] - columnMean) / columnStdDev;
		}
	}

	std::ofstream file1(modelOutputDirectory + "/model_mean.bin", std::ios::binary);
	file1.write(reinterpret_cast<char*>(meanVec.data()), sizeof(float) * featureVecSize);

	std::ofstream file2(modelOutputDirectory + "/model_std.bin", std::ios::binary);
	file2.write(reinterpret_cast<char*>(stdVec.data()), sizeof(float) * featureVecSize);

	// Train with calculated dataset
	try {
		trainCreateDNN(featureMatrix, labelVec);
	}
	catch (const std::length_error) {
		std::cerr << "Too few or no training samples in source directory! \n";
		std::string audioSourcePath;
		printf("Set full path for audio source needed for model training: ");
		std::getline(std::cin, audioSourcePath);
		SetAudioSourceDirectory(audioSourcePath);
		TrainAndCreateModel();
	}

	return 0;
}

int TinyAudioClassifier::ProcessAudio(float* audioSampleBlock, int numberOfSamples, float& detectorProbability) {

	std::vector<float> audioBlockVec(audioSampleBlock, audioSampleBlock + numberOfSamples);
	vec_t featureVec;

	int isTargetPresent = 0;
	detectorProbability = 0.f;

	int ChunkCounter = 0;

	for (int wavPosition = 0; wavPosition <= numberOfSamples - mainWindowSize; wavPosition += mainWindowHopSize) {

		std::vector<float> audioSubVec(&audioBlockVec[wavPosition], &audioBlockVec[wavPosition + mainWindowSize]);

		featureVec = processMainWindow(audioSubVec);

		// Scale features
		for (int i = 0; i < featureVecSize; i++) {
			featureVec[i] = (featureVec[i] - meanVec[i]) / stdVec[i];
		}

		// Check if the sound is present in subwindow
		isTargetPresent = neural_network.predict_label(featureVec);
		printf("Does target sound exist in chunk %d: %d \n ", ChunkCounter, isTargetPresent);

		// Check if the sound is present in at least one subwindow
		detectorProbability = (float)(isTargetPresent || (int)detectorProbability);

		ChunkCounter++;

	}

	detectorProbability = (float)isTargetPresent;

	return 0;

};

int TinyAudioClassifier::ProcessAudio(int16_t* audioSampleBlock, int numberOfSamples, float& detectorProbability) {

	std::vector<float> audioBlockVec;
	audioBlockVec.reserve(numberOfSamples);

	for (int i = 0; i < numberOfSamples; i++) {
		audioBlockVec[i] = (float)audioBlockVec[i] / pow(2.0f, 15.0f);
	}
	vec_t featureVec;

	int isTargetPresent = 0;

	for (int wavPosition = 0; wavPosition <= numberOfSamples - mainWindowSize; wavPosition += mainWindowHopSize) {

		std::vector<float> audioSubVec(&audioBlockVec[wavPosition], &audioBlockVec[wavPosition + mainWindowSize]);

		featureVec = processMainWindow(audioSubVec);

		// Scale features
		for (int i = 0; i < featureVecSize; i++) {
			featureVec[i] = (featureVec[i] - meanVec[i]) / stdVec[i];
		}

		// Check if the sound is present in at least one block
		isTargetPresent |= neural_network.predict_label(featureVec);
		printf("%f ", detectorProbability);

	}

	detectorProbability = (float)isTargetPresent;

	return 0;

};
