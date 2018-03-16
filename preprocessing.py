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
