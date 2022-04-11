#It uses the Wavelet transform with Complex Morlet wavelet to compute the Spectrogram,
# after rFFT is used to obtain the Modulation Spectrogram

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import am_analysis as ama


def first_run():
    global n_cycles
    global x
    global x_spectrogram
    global channel_names
    global fs
    global parameters
    global ix_channel
    global fig
    global gc_map

    fig.clear()

    n_cycles = round(parameters['n_cycles'])

    # compute and plot complete spectrogram
    x_spectrogram = ama.wavelet_spectrogram(x, fs, n_cycles, channel_names=[channel_names[ix_channel]])
    x_wavelet_modspec = ama.wavelet_modulation_spectrogram(x, fs, n_cycles=n_cycles, fft_factor_x=2,
                                                           channel_names=[channel_names[ix_channel]])
    plt.subplot(4, 2, (6, 8))
    ama.plot_modulation_spectrogram_data(x_wavelet_modspec, f_range=parameters['freq_range'],
                                         modf_range=parameters['mfreq_range'], c_range=parameters['mfreq_color'],
                                         c_map=gc_map)

    # plot spectrogram for full signal
    plt.subplot(4, 2, (3, 4))
    ama.plot_spectrogram_data(x_spectrogram, f_range=parameters['freq_range'], c_range=parameters['freq_color'],
                              c_map=gc_map)


    # plot full signal
    plt.subplot(4, 2, (1, 2))
    ama.plot_signal(x, fs, channel_names[ix_channel])

    plt.colorbar()
    plt.show()
    return


def explore_wavelet_ama_gui(x, fs_arg, channel_names_arg=None, c_map='viridis'):
    # Global variables
    global ix_channel
    global n_channels
    global cid
    global fig
    global parameters
    global gc_map
    global n_cycles
    global name
    global fs
    global channel_names

    fs = fs_arg
    channel_names = channel_names_arg
    gc_map = c_map

    # % Amplitude Modulation Analysis
    # Default Modulation Analysis parameters
    parameters = {}
    parameters['n_cycles'] = 6  # number of cycles (for Complex Morlet)
    parameters['freq_range'] = None  # limits [min, max] for the conventional frequency axis (Hz)
    parameters['mfreq_range'] = None  # limits [min, max] for the modulation frequency axis (Hz)
    parameters['freq_color'] = None  # limits [min, max] for the power in Spectrogram (dB)
    parameters['mfreq_color'] = None  # limits [min, max] for the power in Modulation Spectrogram (dB)

    # initial channel and segment
    ix_channel = 0

    # Live GUI
    fig = plt.figure()
    first_run()


if __name__ == '__main__':
    os.chdir('../dataset/headerRemoved')

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

    x = np.array(y_list)
    fs = 240.0
    # Wavelet Modulation Spectrogram
    explore_wavelet_ama_gui(x, fs, ['signal1'])