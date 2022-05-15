#It uses the Wavelet transform with Complex Morlet wavelet to compute the Spectrogram,
# after rFFT is used to obtain the Modulation Spectrogram

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import am_analysis as ama


def plot_signal(x, fs, name = None, label = None):
    time_vector = np.arange(x.shape[0]) / fs

    plt.plot(time_vector, x, label = label, linewidth=1)
    plt.xlabel('time (s)')
    plt.xlim([time_vector.min(), time_vector.max()])

    if name is None:
        name = 'Signal-01'

    plt.title(name)
    plt.legend()
    plt.draw()

def explore_wavelet_ama_gui(x, fs_arg, name_arg=None, c_map='viridis'):
    # Default Modulation Analysis parameters
    parameters = {}
    parameters['n_cycles'] = 9  # number of cycles (for Complex Morlet)
    parameters['freq_range'] = None  # limits [min, max] for the conventional frequency axis (Hz)
    parameters['mfreq_range'] = None  # limits [min, max] for the modulation frequency axis (Hz)
    parameters['freq_color'] = None  # limits [min, max] for the power in Spectrogram (dB)
    parameters['mfreq_color'] = None  # limits [min, max] for the power in Modulation Spectrogram (dB)

    n_cycles = round(parameters['n_cycles'])
    fs = fs_arg
    name = name_arg
    gc_map = c_map

    # compute and plot complete spectrogram
    x_spectrogram = ama.wavelet_spectrogram(x, fs, n_cycles, channel_names=name)

    # plot spectrogram for full signal
    plt.subplot(1, 1, 1)
    ama.plot_spectrogram_data(x_spectrogram)

    # plot full signal
    # plt.subplot(1, 1, 1)
    # plot_signal(x, fs, name, label = 'raw signal')
    #
    # # plot reproduced full signal
    # x_r = ama.iwavelet_spectrogram(x_spectrogram)
    # plt.subplot(1, 1, 1)
    # plot_signal(x_r, fs, name, label = 'reproduced signal')


if __name__ == '__main__':
    os.chdir('../data/SmallMisalignment08_09/headerRemoved')
    fs = 2000.0

    for csvFilename in os.listdir('.'):
        if not csvFilename.endswith('.csv'):
            continue  # skip non-csv files
        print('open ' + csvFilename + '...')

        # Read the CSV file
        real = []
        csvFileObj = open(csvFilename)
        readerObj = csv.reader(csvFileObj)
        for row in readerObj:
            x = float(row[0])
            y = float(row[1])
            real.append([x, y])

        x_list, y_list = [row[0] for row in real], [row[1] for row in real]

        # Wavelet Modulation Spectrogram
        explore_wavelet_ama_gui(np.array(y_list), fs, csvFilename[0:4])
        plt.show()