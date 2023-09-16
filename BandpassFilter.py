from scipy.signal import butter, filtfilt



def bandpass_filter(signal, sample_rate, lowcut, highcut):

    nyquist_freq = 0.5*sample_rate
    low = lowcut/nyquist_freq
    high = highcut/nyquist_freq
    order = 2

    b, a = butter(order, [low,high], 'bandpass', analog=False)
    filtered_signal = filtfilt(b, a, signal, axis=0)

    return filtered_signal