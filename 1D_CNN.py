"""
                                        LIBRARIES
"""
import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

"""
                          LOG-MEL SPECTROGRAM FEATURE EXTRACTION
"""


def extract_features(filename, bands=40, frames=501):
    log_spectrograms = []
    X, _ = librosa.load(filename, sr=16000)
    melspec = librosa.feature.melspectrogram(y=X, sr=16000, n_mels=bands, n_fft=512, hop_length=320)
    logspec = librosa.power_to_db(melspec)
    logspec = logspec.T.flatten()[:, np.newaxis].T
    log_spectrograms.append(logspec)
    log_spectrograms = np.asarray(log_spectrograms).reshape(len(log_spectrograms),
                                                            bands, frames)
    return np.array(log_spectrograms)


sample_filename = "audio/train/fold1/cooking/DevNode1_ex43_1.wav"
features = extract_features(sample_filename)
data_points, _ = librosa.load(sample_filename, sr=16000)
print("IN: Initial Data Points =", len(data_points))
print("OUT: Total features =", np.shape(features))

"""
                                  ASSURE PATH EXISTS
"""


def assure_path_exists(path):
    mydir = os.path.join(os.getcwd(), path)
    if not os.path.exists(mydir):
        os.makedirs(mydir)


"""
                                 GET ALL THE LABELS
"""
parent_dir = "audio/train/fold1/"
labels_dict = {}
list_labels = []
list_counts = []
for path, subdirs, files in os.walk(parent_dir):
    path = path.replace(parent_dir, "")
    if len(files) != 0:
        list_labels = np.append(path, list_labels)
        list_counts = np.append(len(files), list_counts).astype(int)

labels_dict = dict(zip(list_labels, list_counts))
plt.bar(range(len(labels_dict)), list(labels_dict.values()), align='center')
plt.xticks(range(len(labels_dict)), list(labels_dict.keys()), rotation=45)
plt.tight_layout()

"""
               SAVE FOLDS WITH FEATURES AND LABELS AS NUMPY ARRAYS
"""
features = np.empty((0, 40, 501))
labels = np.empty(0)
for path, subdirs, files in os.walk(parent_dir):
    path = path.replace(parent_dir, "")
    if len(files) != 0:
        for audio_name in tqdm(files):
            log_specs = extract_features(parent_dir + "/" + path + "/" + audio_name)
            ext_features = np.hstack([log_specs])
            features = np.vstack([features, ext_features])
            labels = np.append(path, labels)

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(labels)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
labels = onehot_encoded
feature_file = "train_mels_fold1_x.npy"
labels_file = "train_mels_fold1_y.npy"
np.save(feature_file, features)
print("Features = ", features.shape)
np.save(labels_file, labels)
print("Labels = ", labels.shape)
