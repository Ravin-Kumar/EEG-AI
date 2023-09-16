import time
import pickle
import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.interpolate import interp2d
from scipy.signal import cwt, morlet2
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout, BatchNormalization, Reshape

from BandpassFilter import bandpass_filter


def get_data(folder, fileno):

    filename = f"{folder}00{fileno}" if fileno < 10 else f"{folder}0{fileno}" if 10<=fileno<=99 else f"{folder}{fileno}"
    with open(f'D:\cs\python\seizure-detection-AI\datasets\Bonn\{folder}\{filename}.txt', 'r') as signal_file:
        data = np.array([int(x) for x in signal_file.read().split('\n')[:-1]])
    
    return data

fs = 173.6
WIDTH_LENGTH, WIDTH_PATTERN = 100, 2

freqs = np.linspace(1, fs/2, WIDTH_LENGTH)
widths = (6*fs)/(2*freqs*np.pi)

x = np.empty((500, WIDTH_LENGTH, 4097), dtype=np.complex128)
y = np.zeros(500, dtype=np.uint8)

idx = 0

for folder in 'ZONFS':
    for filename in range(1, 101):
        data = get_data(folder, filename)
        data = bandpass_filter(data, 173.6, 0.53, 40)
        x[idx] = cwt(data, morlet2, widths)

        if folder == 'S':
            y[idx] = 1

        idx += 1

y_vals = np.arange(WIDTH_LENGTH)
x_vals = np.arange(4097)

x_resized = np.empty((500,32,32), dtype=np.complex128)

for idx, image in enumerate(x):
    f_real = interp2d(x_vals,y_vals,image.real,kind='cubic')
    f_imag = interp2d(x_vals,y_vals,image.imag,kind='cubic')
    y_new = np.linspace(0, WIDTH_LENGTH, 32)
    x_new = np.linspace(0, 4097, 32)
    x_resized[idx].real = f_real(x_new, y_new)
    x_resized[idx].imag = f_imag(x_new, y_new)

def get_classification_metric(y_test, pred):
    fscore = metrics.f1_score(y_test, np.argmax(pred, axis=1), average='binary')
    return fscore

def build_model(hp):
    model = tf.keras.models.Sequential()
    model.add(Conv2D(hp.Int('filter1', 8, 96, 8), (5,5), activation='relu', input_shape=(32,32,1)))
    if hp.Choice('BN1', values=(0,1)):
        model.add(BatchNormalization())
    model.add(MaxPool2D(hp.Int('stride1', 1, 4)))
    model.add(Dropout(hp.Choice('dropout_rate1', values=[0.0,0.2,0.25,0.5])))
    model.add(Conv2D(hp.Int(f'filter2', 16, 64, 16), (5,5), activation='relu'))
    if hp.Choice(f'BN2', values=(0,1)):
        model.add(BatchNormalization())
    model.add(MaxPool2D(hp.Int(f'stride2', 1, 4), padding='same'))
    model.add(Dropout(hp.Choice('dropout_rate2', values=[0.0,0.2,0.25,0.5])))
    model.add(Dense(hp.Int(f'dense neuron no', 50, 1000, 50)))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adadelta(learning_rate=hp.Choice('learning_rate', values=[0.0005, 0.001, 0.01])), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    return model

model_name = f'CNNSM-wp{WIDTH_PATTERN}-wl{WIDTH_LENGTH}-A'

LOG_DIR = f'./models/{model_name}/{int(time.time())}'

tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=20,
    executions_per_trial=3,
    directory=LOG_DIR)

x_train, x_test, y_train, y_test = train_test_split(x_resized, y, test_size=0.2, stratify=y)


tuner.search(x_train, y_train, epochs=50, batch_size=4, validation_data=(x_test, y_test))

with open(f"./seizure-detection-AI/models/tuner_{model_name}.pkl", "wb+") as f:
    pickle.dump(tuner, f)

print(tuner.get_best_hyperparameters()[0].values)