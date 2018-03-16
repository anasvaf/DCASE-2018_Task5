"""
                    PREPROCESSING
"""
import librosa
import numpy as np
from tqdm import tqdm
import pandas as pd
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.join(dir_path, 'UrbanSound8K/audio')

d = []
for path, subdirs, files in os.walk(parent_dir):
    for file in tqdm(files):
        try:
            sound_clip, sr = librosa.load(path+'/'+file)
            label = file.split('-')[1]
            d.append({'Filename': file, 'Label': label})
            df = pd.DataFrame.from_dict(d)
            df.to_csv('Training.csv')
        except Exception as e:
            print("\n", "[Error] Cannot handle audio file. %s" % e, file)
            continue



















































# def extract_features(parent_dir, sub_dirs, file_ext="*.wav", bands=60, frames=41):
#     window_size = 512 * (frames - 1)
#     mfccs = []
#     labels = []
#     for l, sub_dir in enumerate(sub_dirs):
#         for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
#             sound_clip, s = librosa.load(fn)
#             label = fn.split('fold')[1].split('-')[1]
#             for (start, end) in windows(sound_clip,window_size):
#                 if(len(sound_clip[start:end])) == window_size:
#                     signal = sound_clip[start:end]
#                     mfcc = librosa.feature.mfcc(y=signal, sr=s, n_mfcc=bands).T.flatten()[:, np.newaxis].T
#                     mfccs.append(mfcc)
#                     labels.append(label)
#     features = np.asarray(mfccs).reshape(len(mfccs), bands, frames)
#     return np.array(features), np.array(labels, dtype=np.int)