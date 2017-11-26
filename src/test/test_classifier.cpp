/* Tiny Audio Classifier */
/* Copyright (c) 2017 Ivan Vican */

#include <stdlib.h>  
#include <stdio.h>
#include <crtdbg.h>  

#include "..\tiny_audio_classifier_api.h"

/* TO DO */
#define TEST_OFFLINE 0
#define TEST_ONLINE  1

/* Select test here */
#define SELECT_TEST   TEST_OFFLINE

/* Function for running offline test */
void RunOfflineTest()
{
   printf("Executing offline test......\n");

	 SoundClassifier classifier;

	 std::printf("Choose Test (1 -- train model, 2 - load model and test offline: \n");
	 std::string testType;
	 std::getline(std::cin, testType);

	 if (testType == "1") {
		 std::string audioSourcePath;
		 printf("Set full path for audio source needed for model training: ");
		 std::getline(std::cin, audioSourcePath);

		 std::string modelSavePath;
		 printf("Set directory for saving model: ");
		 std::getline(std::cin, modelSavePath);

		 classifier.SetModelOutputDirectory(modelSavePath);
		 classifier.SetAudioSourceDirectory(audioSourcePath);
		 classifier.TrainAndCreateModel();

		 std::string modelFullSavePath = modelSavePath + "/model";

		 printf("Model saved as %s \n", modelFullSavePath.c_str());
	 }
	 else if (testType == "2") {
		 std::string modelLoadPath;
		 printf("Set full path for loading model: ");
		 std::getline(std::cin, modelLoadPath);
		 classifier.LoadSoundClassificationModel(modelLoadPath);

		 std::string audioFilePath;
		 printf("Set full path audio file: ");
		 std::getline(std::cin, audioFilePath);

		 float isSoundPresent;

		 std::vector<float> audioStreamVec = classifier.readAudioFile(audioFilePath.c_str());
		 classifier.ProcessAudio(audioStreamVec.data(), audioStreamVec.size(), isSoundPresent);
	 }
	 else {
		 RunOfflineTest();
	 }

	 std::getchar();

}


/* Function for running online test */
void RunOnlineTest()
{
   printf("Executing online test......\n");
}


/* Entry Point */
void main(void)
{
#if (SELECT_TEST == TEST_OFFLINE)
   RunOfflineTest();
#elif (SELECT_TEST == TEST_ONLINE)
   RunOnlineTest();
#endif

}