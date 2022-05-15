import csv
import os
import matplotlib.pyplot as plt

def remove_header():
    # removeCsvHeader.py - Removes the header from all CSV files in the current
    # os.makedirs('../datasetaa/headerRemoved', exist_ok=True)
    os.chdir('../data/MA_1D')
    # Loop through every file in the current working directory.

    for csvFilename in os.listdir('.'):
        if not csvFilename.endswith('.csv'):
            continue  # skip non-csv files

        print('Removing header from ' + csvFilename + '...')

        # Read the CSV file in (skipping first row).
        csvRows = []
        csvFileObj = open(csvFilename)
        readerObj = csv.reader(csvFileObj)

        for row in readerObj:
            if readerObj.line_num <= 25:
                continue  # skip rows
            if readerObj.line_num >= 1501:
                continue  # skip rows
            value = row[0].split("\t")
            csvRows.append(value)
        csvFileObj.close()

        # Write out the CSV file.
        csvFileObj = open(os.path.join('../MA_1D', csvFilename), 'w',
                          newline='')
        csvWriter = csv.writer(csvFileObj)
        for row in csvRows:
            csvWriter.writerow(row)

def write_label():
    os.chdir('../data/SmallMisalignment08_09/headerRemoved')
    csvRows = []
    for csvFilename in os.listdir('.'):
        if not csvFilename.endswith('.csv'):
            continue  # skip non-csv files

        print('opening ' + csvFilename + '...')

        csvRow = []
        csvFileObj = open(csvFilename)
        readerObj = csv.reader(csvFileObj)

        for row in readerObj:
            value = row[1].split("\t")
            csvRow.append(float(value[0]))
        if csvFilename.startswith('L005'):
            csvRow.append(0)
            csvRow.append(0)
        elif csvFilename.startswith('L010'):
            csvRow.append(1)
            csvRow.append(0)
        elif csvFilename.startswith('L015'):
            csvRow.append(2)
            csvRow.append(0)
        elif csvFilename.startswith('L020'):
            csvRow.append(3)
            csvRow.append(0)
        elif csvFilename.startswith('L025'):
            csvRow.append(4)
            csvRow.append(0)
        elif csvFilename.startswith('N'):
            csvRow.append(5)
            csvRow.append(1)
        elif csvFilename.startswith('R005'):
            csvRow.append(6)
            csvRow.append(2)
        elif csvFilename.startswith('R010'):
            csvRow.append(7)
            csvRow.append(2)
        elif csvFilename.startswith('R015'):
            csvRow.append(8)
            csvRow.append(2)
        elif csvFilename.startswith('R020'):
            csvRow.append(9)
            csvRow.append(2)
        elif csvFilename.startswith('R025'):
            csvRow.append(10)
            csvRow.append(2)
        csvRows.append(csvRow)
        csvFileObj.close()

    # Write out the CSV file.
    csvFileObj = open('../../MA_1D/dataset_real.csv', 'w',
                    newline='')
    csvWriter = csv.writer(csvFileObj)
    for csvRow in csvRows:
        csvWriter.writerow(csvRow)

if __name__ == '__main__':
    # remove_header()
    write_label()
