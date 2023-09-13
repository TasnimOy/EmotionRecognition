import librosa
import numpy as np
import pandas as pd
import parselmouth
import os

SAMPLE_RATE = 22050
DATA_DIR = ''
OUTPUT_CSV = ''

def extract_features(data, sample_rate):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))

    # Spectral Contrast
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, spectral_contrast))

    # Spectral Entropy
    spectral_entropy = np.mean(-np.sum(np.multiply(mel, np.log(mel)), axis=0))
    result = np.hstack((result, spectral_entropy))

    # Pitch
    pitch = np.mean(librosa.pyin(y=data, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C8'))[1])
    result = np.hstack((result, pitch))

    # Parselmouth (Praat) features
    snd = parselmouth.Sound(data, sampling_frequency=sample_rate)

    return result


def process_audio_files():
    feature_list = []
    file_paths = []  # List to store full file paths

    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            file_path = os.path.join(root, file)  # This is the full path
            if file_path.endswith(".wav"):

                # Laden der WAV-Datei
                data, _ = librosa.load(file_path, sr=SAMPLE_RATE)

                # Feature-Extraktion
                features = extract_features(data, SAMPLE_RATE)

                # Hinzufügen von Features und Dateipfad zur Liste
                feature_list.append(features)
                file_paths.append(file_path)  # Add the full file path to the list

    return feature_list, file_paths

# Extrahierte Funktionen und Dateipfade in eine DataFrame laden
feature_list, file_paths = process_audio_files()
feature_df = pd.DataFrame(feature_list)

# Adding full file path column to the dataframe
feature_df['file_path'] = file_paths

# CSV-Datei speichern
feature_df.to_csv(OUTPUT_CSV, index=False)

print("Extrahierte Funktionen wurden in", OUTPUT_CSV, "gespeichert.")
