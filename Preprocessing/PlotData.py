import csv
import os
import random

import matplotlib.pyplot as plt


def minmax_normalize(list):
    maximum = max(list)
    minimum = min(list)
    list = [2 * (i - minimum) / (maximum - minimum) - 1 for i in list]
    return list

def plot1():
    fig, ax = plt.subplots()
    ax.set_xlabel('time')
    ax.set_ylabel('power')
    ax.set_title('raw signal')
    line_width = 0.5
    for csvFilename in os.listdir('.'):
        if not csvFilename.endswith('.csv'):
            continue  # skip non-csv files
        # Read the CSV file
        real = []
        csvFileObj = open(csvFilename)
        readerObj = csv.reader(csvFileObj)
        for row in readerObj:
            x = float(row[0])
            y = float(row[1])
            real.append([x, y])
        x_list, y_list = [row[0] for row in real], [row[1] for row in real]
        y_list = minmax_normalize(y_list)
        if csvFilename.startswith('L005_300to500_s200-01'):
            rgb = (random.random(), random.random(), random.random())
            ax.plot(x_list, y_list, label = 'L005', c = rgb, linewidth=line_width)
        # elif csvFilename.startswith('L010_300to500_s200-01'):
        #     rgb = (random.random(), random.random(), random.random())
        #     ax.plot(x_list, y_list, label = 'L010', c = rgb, linewidth=line_width)
        # elif csvFilename.startswith('L015_300to500_s200-01'):
        #     rgb = (random.random(), random.random(), random.random())
        #     ax.plot(x_list, y_list, label='L015', c = rgb, linewidth=line_width)
        # elif csvFilename.startswith('L020_300to500_s200-01'):
        #     rgb = (random.random(), random.random(), random.random())
        #     ax.plot(x_list, y_list, label = 'L020', c = rgb, linewidth=line_width)
        # elif csvFilename.startswith('L025_300to500_s200-01'):
        #     rgb = (random.random(), random.random(), random.random())
        #     ax.plot(x_list, y_list, label = 'L025', c = rgb, linewidth=line_width)
        elif csvFilename.startswith('N_300to500_s200-01'):
            rgb = (random.random(), random.random(), random.random())
            ax.plot(x_list, y_list, label = 'N', c = rgb, linewidth=line_width)
        elif csvFilename.startswith('R005_300to500_s200-01'):
            rgb = (random.random(), random.random(), random.random())
            ax.plot(x_list, y_list, label = 'R005', c = rgb, linewidth=line_width)
        # elif csvFilename.startswith('R010_300to500_s200-01'):
        #     rgb = (random.random(), random.random(), random.random())
        #     ax.plot(x_list, y_list, label = 'R010', c = rgb, linewidth=line_width)
        # elif csvFilename.startswith('R015_300to500_s200-01'):
        #     rgb = (random.random(), random.random(), random.random())
        #     ax.plot(x_list, y_list, label = 'R015', c = rgb, linewidth=line_width)
        # elif csvFilename.startswith('R020_300to500_s200-01'):
        #     rgb = (random.random(), random.random(), random.random())
        #     ax.plot(x_list, y_list, label = 'R020', c = rgb, linewidth=line_width)
        # elif csvFilename.startswith('R025_300to500_s200-01'):
        #     rgb = (random.random(), random.random(), random.random())
        #     ax.plot(x_list, y_list, label = 'R025', c = rgb, linewidth=line_width)
    ax.legend()
    plt.show()
    csvFileObj.close()



if __name__ == '__main__':
    os.chdir('../data/SmallMisalignment08_09/headerRemoved')
    plot1()
