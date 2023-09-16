import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import morlet2, cwt
from scipy.interpolate import interp2d
from BandpassFilter import bandpass_filter
import sys


def get_data(folder, fileno):

    filename = f"{folder}00{fileno}" if fileno < 10 else f"{folder}0{fileno}" if 10<=fileno<=99 else f"{folder}{fileno}"
    with open(f'D:\cs\python\seizure-detection-AI\datasets\Bonn\{folder}\{filename}.txt', 'r') as signal_file:
        data = np.array([int(x) for x in signal_file.read().split('\n')[:-1]])
    
    return data


class CNN(tf.keras.Model):

    def __init__(self):
        super().__init__()

        self.network = tf.keras.Sequential([
            Conv2D(16, (5,5), activation='relu', input_shape=(32,32,1)),
            MaxPool2D(strides=2),
            Conv2D(64, (5,5), activation='relu'),
            MaxPool2D(strides=2),
            Dense(1000),
            Flatten(),
            Dense(2)#, activation='softmax')
        ])
    
    def call(self, x):
        return self.network(x)


def main():

    CNN_model = CNN()

    #! Processing the Data

    y_train = np.zeros(400, dtype=np.uint8)
    y_test = np.zeros(100, dtype=np.uint8)

    for i in range(320, 400):
        y_train[i] = 1
    for i in range(80, 100):
        y_test[i] = 1

    fs = 173.6
    try:
        WIDTH_LENGTH, WIDTH_PATTERN = sys.argv[1], sys.argv[2]
    except IndexError:
        WIDTH_LENGTH, WIDTH_PATTERN = 200, 2

    freqs = np.linspace(1, fs/2, WIDTH_LENGTH)
    widths = (6*fs)/(2*freqs*np.pi)
    #widths = np.linspace(1, fs/2, WIDTH_LENGTH)

    x_train_input = np.empty((400, WIDTH_LENGTH, 4097), dtype=np.complex128)
    x_test_input = np.empty((100, WIDTH_LENGTH, 4097), dtype=np.complex128)

    data_inputs = (x_train_input, x_test_input)

    for idx, data_input in enumerate(data_inputs):
        if idx == 0:
            ranges = (1,81)
        else:
            ranges = (81,101)
        
        idx = 0

        for folder in 'ZONFS':
            for filename in range(ranges[0], ranges[1]):
                data = get_data(folder, filename)
                data = bandpass_filter(data, 173.6, 0.53, 40)
                data_input[idx] = cwt(data, morlet2, widths)

                idx += 1


    y = np.arange(WIDTH_LENGTH)
    x = np.arange(4097)
    xx, yy = np.meshgrid(x,y)

    x_train_input_resized = np.empty((400,32,32), dtype=np.complex128)
    x_test_input_resized = np.empty((100,32,32), dtype=np.complex128)

    data_inputs = (x_train_input, x_test_input)

    for data_input_idx in (0,1):
        for idx, image in enumerate(data_inputs[data_input_idx]):
            f_real = interp2d(x,y,image.real,kind='cubic')
            f_imag = interp2d(x,y,image.imag,kind='cubic')
            y_new = np.linspace(0, WIDTH_LENGTH, 32)
            x_new = np.linspace(0, 4097, 32)
            if data_input_idx == 0:
                x_train_input_resized[idx].real = f_real(x_new, y_new)
                x_train_input_resized[idx].imag = f_imag(x_new, y_new)
            else:
                x_test_input_resized[idx].real = f_real(x_new, y_new)
                x_test_input_resized[idx].imag = f_imag(x_new, y_new)


    #!  Training the model

    CNN_model.compile(optimizer=tf.keras.optimizers.Adadelta(), loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])
    history = CNN_model.fit(x_train_input_resized, y_train, batch_size=4, epochs=50, validation_data=(x_test_input_resized, y_test), shuffle=True)

    model_name = f'CNN-wp{WIDTH_PATTERN}-wl{WIDTH_LENGTH}-A'
    CNN_model.save(f'./models/{model_name}')

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.savefig(f'./models/{model_name}.jpg')
    plt.show()


if __name__ == '__main__':
    main()