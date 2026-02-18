
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import lru_cache
import matplotlib.pyplot as plt; plt.ion()

from librosa.feature import mfcc
from librosa.core.spectrum import stft
from scipy.io import wavfile as wav
import seaborn as sns

from call_finder_rnn import AudioDataset, Files, device
from sklearn.manifold import TSNE

np.random.seed(42)

def feature(a, n_fft_prop=1/3):
    S = np.abs(stft(a,
        n_fft=int(len(a) * n_fft_prop),
        hop_length=int(len(a) * n_fft_prop/2
    )))

    mel_features = mfcc(S=S, n_mfcc=20)
    mel_features = (mel_features - mel_features.mean()) / (mel_features.std() + 1e-6)
    return mel_features.reshape(-1)

def process_file(f, start, end, data_loc=Files.lb_data_loc):
    sr, a = read_audio(f, data_loc=data_loc)
    a = a[int(start * sr):int(end * sr)]
    return feature(a)

@lru_cache(maxsize=50)
def read_audio(f, data_loc=Files.lb_data_loc):
    sr, audio = wav.read(os.path.join(data_loc, f))
    if len(audio.shape) == 2:
        return sr, audio.mean(axis=1)
    else:
        return sr, audio.astype(float)

if __name__ == '__main__':

    data_loader = AudioDataset(device=device)
    calls = data_loader.labels.copy()

    calls.loc[calls.call_type == 'Resonating Note', 'call_type'] = 'Resonate'
    calls = calls.loc[calls.call_type != 'Tsit'].reset_index(drop=True)

    screams = pd.read_excel('data/labelled_data/Screams_Labels.xlsx')
    screams.columns = ['file', 'call_type', 'start', 'end']

    calls = pd.concat([calls, screams], axis=0).reset_index(drop=True)
    calls = calls.loc[calls.end > calls.start].reset_index(drop=True)

    calls.loc[calls.call_type.isin(['Jagged', 'Jagged Trills', 'Jagged Trill']), 'call_type'] = 'Jagged'

    calls['file_with_ext'] = calls['file'] + '.wav'

    X = np.vstack([
        process_file(*calls.loc[i, ['file_with_ext', 'start', 'end']]) \
        for i in tqdm(calls.index)
    ])
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)

    calls['zoo'] = calls.file
    calls.loc[calls.file.str.contains('ML_Test'), 'zoo'] = 'Banham'

    Z = TSNE(random_state=42, perplexity=15).fit_transform(X)

    df = pd.DataFrame(Z, columns=['latent_a', 'latent_b'])
    df['group'] = calls.call_type

    palette = sns.color_palette("Paired", len(df['group'].unique()))

    # Create the scatter plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x='latent_a', y='latent_b', hue='group', style='group', palette=palette)

    # Enhancing the plot
    plt.title("Projection Colored by Call Type")
    plt.xlabel("latent_a")
    plt.ylabel("latent_b")
    plt.legend(title='Call Type', bbox_to_anchor=(1.05, 1), loc=2)
    plt.axis('off')
    plt.tight_layout()
