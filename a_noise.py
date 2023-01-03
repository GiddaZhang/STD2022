import librosa
import soundfile as sf
import numpy as np
import os

def add_noise(data):
    wn = np.random.normal(0, 5, len(data))
    data_noise = np.where(data != 0.0, data.astype('float64') + 0.02 * wn, 0.0).astype(np.float32)
    return data_noise

# data, fs = librosa.core.load('./test_audio/clean/0000.wav')
# data_noise = add_noise(data)
# # librosa.output.write_wav('./test_audio/noise/add_0000.wav', data_noise, fs)
# sf.write('./test_audio/noise/add_0000.wav', data_noise, fs)

for path in os.listdir('./Train/audio/'):
    data, fs = librosa.core.load('./Train/audio/' + path)
    data_noise = add_noise(data)
    # librosa.output.write_wav('./test_audio/noise/add_0000.wav', data_noise, fs)
    sf.write('./Train/noised audio/' + path, data_noise, fs)
