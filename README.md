# EmotionRecognition
This repository contains the code for extracting audio features and predicting emotions from audio files using deep learning techniques.

Table of Contents

Overview
File Structure
Getting Started
Datasets Used for Training
References
1. Overview

The provided code is structured in the following manner:

Feature Extraction: Extract various audio features such as ZCR, Chroma_stft, MFCC, RMS, MelSpectogram, Spectral Contrast, Spectral Entropy, and Pitch from .wav audio files.
Model Definition: Define a Conv1D deep learning model to classify emotions based on extracted features.
Model Training: Train the model on preprocessed data.
Model Evaluation: Evaluate the trained model's performance on test data and visualize results.
Prediction on New Data: Predict emotions on new audio files using the pre-trained model and evaluate the distribution of predicted emotions.

2. File Structure
The main files include:
feature_extraction.py: Contains the code for extracting audio features.
model_definition_training.py: Contains the code for defining, training, and evaluating the model.
predict_on_new_data.py: Contains the code for predicting emotions on new audio files.

3. Getting Started
Make sure you have the required libraries installed. This includes:
librosa
numpy
pandas
parselmouth
os
tensorflow
scikit-learn
joblib
matplotlib
Set the appropriate paths for DATA_DIR and OUTPUT_CSV in the feature_extraction.py file to specify where your audio files are stored and where the extracted features should be saved, respectively.
Run the feature extraction code to generate the .csv file with extracted features.
In model_definition_training.py, set the dataset_path variable to the path of the generated .csv file.
Train the model and evaluate its performance.
For predicting emotions on new data, update the paths in predict_on_new_data.py and run it.

5. Datasets Used for Training
RAVDESS: The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) encompasses audio and video data from professional actors expressing a range of emotions. The dataset covers eight distinct emotions and each emotion is captured in two intensity levels. The dataset offers both audio and video recordings, with an equal representation of male and female actors. [1]

TESS: The Toronto Emotional Speech Set (TESS) is designed specifically with the older demographic in mind. TESS consists of emotional speech samples from older adult female actors. It covers seven emotions through 200 lexically matched statements. [2]

IEMOCAP: The Interactive Emotional Dyadic Motion Capture (IEMOCAP) dataset contains approximately 12 hours of audio-visual data, including video, speech, motion capture of facial movements, and text transcriptions. This dataset captures a wide range of emotions in various contexts from ten actors in dyadic sessions. For this project, the emotions 'frustrated' and 'excited' were utilized. [3]


5. emotion_labels = {0: 'neutral', 1: 'happy', 2: 'sad', 3: 'Pleasent_Surprise', 4: 'fearful', 5: 'surprise',
                  6: 'angry', 7: 'disgust', 8: 'calm', 9: 'excited', 10: 'frustrated', 11: 'boredom'}

6. References

Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. PloS one, 13(5), e0196391.
Pichora-Fuller, M. K., et al. (2020). Toronto Emotional Speech Set (TESS).
Busso, C., et al. (2008). IEMOCAP: Interactive emotional dyadic motion capture database. Language resources and evaluation, 42(4), 335.
