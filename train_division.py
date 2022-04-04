#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import StratifiedKFold
import argparse
import os

def import_txt(filename):
    all_data = []
    with open(filename) as f:
        reader = f.readlines()       
        for row in reader:
            all_data.append(row.replace("\n", "").split(','))          
    return pd.DataFrame(all_data, columns = ["seq", "label"])
 
def output_csv_pandas(filename, data):
    data.to_csv(filename, index = None)

##################################################################
parser = argparse.ArgumentParser()
parser.add_argument('-i1', '--infile1', help='file')
parser.add_argument('-dp', '--datapath', help='path')
parser.add_argument('-fold', '--kfold', help='value')

train_file = parser.parse_args().infile1
out_path = parser.parse_args().datapath
kfold = parser.parse_args().kfold
kfold=int(kfold)

training_data = import_txt(train_file )
print(training_data)

count=0
skf = StratifiedKFold(n_splits = kfold, shuffle=True)
for train_index, val_index in skf.split(training_data, training_data['label']):
    count += 1
    os.makedirs(out_path + "/cross_val/" + str(count), exist_ok = True)
    output_csv_pandas(out_path+ "/cross_val/" + str(count) + "/cv_train_" + str(count) + ".csv", training_data.loc[train_index,:].reset_index(drop=True))
    output_csv_pandas(out_path+ "/cross_val/" + str(count) + "/cv_val_" + str(count) + ".csv", training_data.loc[val_index,:].reset_index(drop=True))

