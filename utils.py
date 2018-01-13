import librosa
import numpy as np
import scipy


def extract_spectrogram_windows(path, max_frequency=128, window_len=512):
    """
    Extracts spectrogram of sound file, divided into equally timed windows
    :param path: path to file
    :param max_frequency: maximum frequency. If current frequency is higher, cut it to max frequency
    :param window_len: length of single window
    :return: tuple of spectrogram windows of size max_frequency x windows_len
    """
    timeseries = librosa.core.load(path, sr=44100)[0]
    _, _, spectrogram = scipy.signal.spectrogram(timeseries)
    spectrogram = np.flipud(np.array(spectrogram))
    spectrogram = spectrogram[:max_frequency, :]
    n_windows = spectrogram.shape[1] // window_len
    windows = [spectrogram[:, i * window_len:(i + 1) * window_len] for i in range(n_windows)]
    return windows