#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import argparse
import pandas as pd
from gensim.models import word2vec
#import logging

def sep_word(data, num):
    res = []

    for i, seq in enumerate(data):
        res.append([seq[j: j+ num] for j in range(len(seq) - num + 1)])
        
    return res

####################################################################
start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('-w', '--w2v', help='path')
parser.add_argument('-i1', '--infile1', help='file')
parser.add_argument('-i2', '--infile2', help='file')
parser.add_argument('-sw', '--seqwin', help='value')
parser.add_argument('-k', '--kmer', help='value')
parser.add_argument('-s', '--size', help='value')
parser.add_argument('-e', '--epochs', help='value')
parser.add_argument('-sg', '--sg', help='value')
parser.add_argument('-window', '--window', help='value')

infile1 = parser.parse_args().infile1
infile2 = parser.parse_args().infile2
w2v_path = parser.parse_args().w2v
seqwin = parser.parse_args().seqwin
kmer = parser.parse_args().kmer
size = parser.parse_args().size
epochs = parser.parse_args().epochs
sg = parser.parse_args().sg
window = parser.parse_args().window

seqwin = int(seqwin) #40
kmer = int(kmer) #4
size = int(size) #100
epochs = int(epochs)
sg = int(sg) # default 1
window = int(window) #100

#sequence preparation
df1 = pd.read_csv(infile1, sep=',', header=None) #CV
df2 = pd.read_csv(infile2, sep=',', header=None) #test

df1_p = df1[df1[1]==1]
df1_n = df1[df1[1]==0]
df2_shuf = df2.sample(frac=1).reset_index(drop=True)

df3 = pd.concat([df1_p, df2_shuf, df1_n])

#data triming and padding
sequences = df3[0].tolist()
for i in range(len(sequences)):
   if len(sequences[i]) > seqwin:
      sequences[i]=sequences[i][0:seqwin]
   sequences[i] = sequences[i].ljust(seqwin, 'X')

for i in range(0, df1_p.shape[0]):
   sequences[i]=sequences[i] +'1,'
for i in range(df1_p.shape[0]+df2_shuf.shape[0], df1_p.shape[0]+df2_shuf.shape[0]+df1_n.shape[0]):
   sequences[i]=sequences[i] +'0,'      
for i in range(df1_p.shape[0], df1_p.shape[0]+df2_shuf.shape[0]):
   sequences[i]=sequences[i] +'X,'
#print(sequences)

# word2vec training
words = sep_word(sequences, kmer)
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
model = word2vec.Word2Vec(words, vector_size = size, min_count = 1, window = window - kmer + 1, epochs = epochs, sg = sg) 
model.save(w2v_path + "/av_w2v_" + str(kmer) + '_' + str(size) + '_' + str(epochs) + '_' + str(window) + '_' + str(sg)  + ".pt")

print('elapsed time:', time.time()-start)

