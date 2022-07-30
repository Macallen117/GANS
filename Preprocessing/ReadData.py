import csv

import torch
import os
import copy
import random
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, auc, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


from torch import nn, optim
from torchsummary import summary
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset

import torch.nn.functional as F


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
            # csvRow.append(0)
            csvRow.append(0)
            flag = False
        elif csvFilename.startswith('L010'):
            # csvRow.append(1)
            csvRow.append(0)
            flag = False
        elif csvFilename.startswith('L015'):
            # csvRow.append(2)
            csvRow.append(0)
            flag = False
        elif csvFilename.startswith('L020'):
            # csvRow.append(3)
            csvRow.append(0)
            flag = False
        elif csvFilename.startswith('L025'):
            # csvRow.append(4)
            csvRow.append(0)
            flag = True
        elif csvFilename.startswith('N'):
            # csvRow.append(5)
            csvRow.append(1)
            flag = False
        elif csvFilename.startswith('R005'):
            # csvRow.append(6)
            csvRow.append(2)
            flag = False
        elif csvFilename.startswith('R010'):
            # csvRow.append(7)
            csvRow.append(2)
            flag = False
        elif csvFilename.startswith('R015'):
            # csvRow.append(8)
            csvRow.append(2)
            flag = False
        elif csvFilename.startswith('R020'):
            # csvRow.append(9)
            csvRow.append(2)
            flag = False
        elif csvFilename.startswith('R025'):
            # csvRow.append(10)
            csvRow.append(2)
            flag = False
        if flag == True:
            csvRows.append(csvRow)
        csvFileObj.close()

    # Write out the CSV file.
    csvFileObj = open('../../MA_1D/dataset_real_0.25.csv', 'w',
                    newline='')
    csvWriter = csv.writer(csvFileObj)
    for csvRow in csvRows:
        csvWriter.writerow(csvRow)

def readRawData(path):
    df = pd.read_csv(path, header=None)
    df.rename(columns={1475: 'target'}, inplace=True)

    scaler_real = preprocessing.MinMaxScaler()
    scaler_real.fit(df.iloc[:, :-1].T)
    df.iloc[:, :-1] = scaler_real.transform(df.iloc[:, :-1].T).T

    df.reset_index(drop=True, inplace=True)
    return df

def create_circle(dataframe):
    df = pd.DataFrame()
    target = dataframe.iloc[:,-1]

    fore_interval = 184
    back_interval = 166
    start = 187
    id = 0

    df1 = dataframe.iloc[:,start-fore_interval:start+back_interval]
    df1 = df1.T.reset_index(drop=True).T
    df1 = pd.concat([df1,target],axis = 1)

    # plt.plot(dataframe.iloc[id,start-fore_interval:start+back_interval])
    # plt.plot(df1.iloc[id,:-1])

    start = start + 2*187

    df2 = dataframe.iloc[:,start-fore_interval:start+back_interval]
    df2 = df2.T.reset_index(drop=True).T
    df2 = pd.concat([df2,target],axis = 1)
    df = pd.concat([df1,df2],axis = 0, ignore_index= True)

    # plt.plot(dataframe.iloc[id,start-fore_interval:start+back_interval])
    # plt.plot(df2.iloc[id, :-1])

    start = start + 2*187

    df3 = dataframe.iloc[:,start-fore_interval:start+back_interval]
    df3 = df3.T.reset_index(drop=True).T
    df3 = pd.concat([df3,target],axis = 1)
    df = pd.concat([df,df3],axis = 0, ignore_index= True)

    # plt.plot(dataframe.iloc[id,start-fore_interval:start+back_interval])
    # plt.plot(df3.iloc[id, :-1])
    start = start + 2*187

    df4 = dataframe.iloc[:,start-fore_interval:start+back_interval]
    df4 = df4.T.reset_index(drop=True).T
    df4 = pd.concat([df4,target],axis = 1)
    df = pd.concat([df,df4],axis = 0, ignore_index= True)

    # plt.plot(dataframe.iloc[id,start-fore_interval:start+back_interval])
    # plt.plot(df4.iloc[id, :-1])
    # plt.show()
    df = df.sample(frac=1.0)
    df.reset_index(drop=True, inplace=True)
    return df

if __name__ == '__main__':
    real_path = '../data/MA_1D/dataset_real_'
    real_005_path = '../data/MA_1D/dataset_real_005.csv'
    real_010_path = '../data/MA_1D/dataset_real_010.csv'
    real_015_path = '../data/MA_1D/dataset_real_015.csv'
    real_020_path = '../data/MA_1D/dataset_real_020.csv'
    real_025_path = '../data/MA_1D/dataset_real_025.csv'
    real_N_path = '../data/MA_1D/dataset_real_N.csv'

    fake_path = '../data/MA_1D/dataset_fake_'
    fake_005_path = '../data/MA_1D/dataset_fake_005.csv'
    fake_010_path = '../data/MA_1D/dataset_fake_010.csv'
    fake_015_path = '../data/MA_1D/dataset_fake_015.csv'
    fake_020_path = '../data/MA_1D/dataset_fake_020.csv'
    fake_025_path = '../data/MA_1D/dataset_fake_025.csv'
    fake_N_path = '../data/MA_1D/dataset_fake_N.csv'

    real_cycle_path = '../data/MA_1D_CYCLE/dataset_real_'
    real_005_cycle_path = '../data/MA_1D_CYCLE/dataset_real_005.csv'
    real_010_cycle_path = '../data/MA_1D_CYCLE/dataset_real_010.csv'
    real_015_cycle_path = '../data/MA_1D_CYCLE/dataset_real_015.csv'
    real_020_cycle_path = '../data/MA_1D_CYCLE/dataset_real_020.csv'
    real_025_cycle_path = '../data/MA_1D_CYCLE/dataset_real_025.csv'
    real_N_cycle_path = '../data/MA_1D_CYCLE/dataset_real_N.csv'

    fake_cycle_path = '../data/MA_1D_CYCLE/dataset_fake_'
    fake_005_cycle_path = '../data/MA_1D_CYCLE/dataset_fake_005.csv'
    fake_010_cycle_path = '../data/MA_1D_CYCLE/dataset_fake_010.csv'
    fake_015_cycle_path = '../data/MA_1D_CYCLE/dataset_fake_015.csv'
    fake_020_cycle_path = '../data/MA_1D_CYCLE/dataset_fake_020.csv'
    fake_025_cycle_path = '../data/MA_1D_CYCLE/dataset_fake_025.csv'
    fake_N_cycle_path = '../data/MA_1D_CYCLE/dataset_fake_N.csv'

    # remove_header()
    write_label()

    # df = readRawData(real_005_path)
    # df = create_circle(df)
    # df.to_csv(real_005_cycle_path,index=False,header=False)
    #
    # df = readRawData(real_010_path)
    # df = create_circle(df)
    # df.to_csv(real_010_cycle_path,index=False,header=False)
    #
    # df = readRawData(real_015_path)
    # df = create_circle(df)
    # df.to_csv(real_015_cycle_path,index=False,header=False)
    #
    # df = readRawData(real_020_path)
    # df = create_circle(df)
    # df.to_csv(real_020_cycle_path,index=False,header=False)

    df = readRawData(real_025_path)
    df = create_circle(df)
    df.to_csv(real_025_cycle_path,index=False,header=False)

    # df = readRawData(real_N_path)
    # df = create_circle(df)
    # df.to_csv(real_N_cycle_path,index=False,header=False)
