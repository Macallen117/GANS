import csv
import os
import matplotlib.pyplot as plt

# removeCsvHeader.py - Removes the header from all CSV files in the current
os.makedirs('../dataset/headerRemoved', exist_ok=True)
os.chdir('../dataset/SmallMisalignment08_09')
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
    csvFileObj = open(os.path.join('../headerRemoved', csvFilename), 'w',
                      newline='')
    csvWriter = csv.writer(csvFileObj)
    for row in csvRows:
        csvWriter.writerow(row)