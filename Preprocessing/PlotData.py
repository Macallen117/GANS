import csv
import os
import matplotlib.pyplot as plt
from matplotlib import ticker

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

    # plt.scatter(x_list[0:200], y_list[0:200])
    fig, ax = plt.subplots(1, 1)
    ax.plot(x_list, y_list)
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(200))
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(5))

    plt.show()
    csvFileObj.close()