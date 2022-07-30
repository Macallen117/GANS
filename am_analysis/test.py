#It uses the Wavelet transform with Complex Morlet wavelet to compute the Spectrogram,
# after rFFT is used to obtain the Modulation Spectrogram

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import am_analysis as ama


def plot_signal(ax, x, name, label):
    ax.plot(x, label = label, linewidth=1)
    ax.set_xlabel('time (s)')
    ax.set_title(name)


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
    ama.plot_spectrogram_data(x_spectrogram)

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    # plot full signal
    plot_signal(ax1, x.squeeze(), name, label = 'raw signal')

    # plot reproduced full signal
    x_r = ama.iwavelet_spectrogram(x_spectrogram)
    plot_signal(ax2, x_r.squeeze(), name, label = 'reproduced signal')


if __name__ == '__main__':
    os.chdir('../data/MA_1D_CYCLE')
    fs = 4000.0

    for csvFilename in os.listdir('.'):
        if not csvFilename.endswith('.csv'):
            continue  # skip non-csv files
        print('open ' + csvFilename + '...')

        # Read the CSV file
        rows = []
        csvFileObj = open(csvFilename)
        readerObj = csv.reader(csvFileObj)
        for row in readerObj:
            row = list(map(float, row))
            rows.append(row)

        x_list, y_list = [row[0:-1] for row in rows], [row[-1] for row in rows]

        # Wavelet Modulation Spectrogram
        explore_wavelet_ama_gui(np.array(x_list[0]), fs, csvFilename[8:-4])
        plt.legend()
        plt.show()