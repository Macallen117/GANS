import pywt
import numpy as np
import matplotlib.pyplot as plt
import csv
import os


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

    x_list,y_list =[row[0] for row in real], [row[1] for row in real]

    # fig, ax = plt.subplots(1, 1)
    # ax.plot(x_list, y_list)
    # plt.axis('off')
    # plt.show()

    coef, freqs = pywt.cwt(y_list, np.arange(1, 1475), 'gaus1')
    plt.matshow(coef)  # doctest: +SKIP
    plt.show()  # doctest: +SKIP
    csvFileObj.close()

