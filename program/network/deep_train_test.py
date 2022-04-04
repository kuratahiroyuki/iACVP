#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import sys
import os
import pandas as pd
import torch
import torch.nn.utils.rnn as rnn
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.autograd import Variable
#import torch_optimizer as optim
from torch import optim
from torch.utils.data import BatchSampler
import numpy as np
from numpy import argmax
import joblib
import argparse
from gensim.models import KeyedVectors
from gensim.models import word2vec
import copy
import json
from TX_network import TX
from LSTM_network import Lstm
#from GRU_network_bidirectional import bGRU
from LSTM_network_bidirectional import bLSTM
#from CNN_LSTM_network import CNN_LSTM
from CNN_network import CNN
import collections
import time
import pickle
import sklearn.metrics as metrics
from sklearn.metrics import precision_recall_curve
from loss_func import CBLoss
from metrics import cofusion_matrix, sensitivity, specificity, auc, mcc, accuracy, precision, recall, f1, cutoff, AUPRC
from sklearn.model_selection import StratifiedKFold
metrics_dict = {"sensitivity":sensitivity, "specificity":specificity, "accuracy":accuracy,"mcc":mcc,"auc":auc,"precision":precision,"recall":recall,"f1":f1,"AUPRC":AUPRC}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_json(filename):
    with open(filename) as f:
        data = json.load(f)     
    return data

def file_input_csv(filename, index_col = None):
    data = pd.read_csv(filename, index_col = index_col)
    return data

def pad_input_csv(filename, seqwin, index_col = None):
    df1 = pd.read_csv(filename, index_col = index_col)
    seq = df1.loc[:,'seq'].tolist()
    #data triming and padding
    for i in range(len(seq)):
       if len(seq) > seqwin:
         seq[i]=seq[i][0:seqwin]
       seq[i] = seq[i].ljust(seqwin, 'X')
    for i in range(len(seq)):
       df1.loc[i,'seq'] = seq[i]   
    return df1

def aa_dict_construction():
   AA = 'ARNDCQEGHILKMFPSTWYVBJOZX'
   keys=[]
   vectors=[]
   for i, key in enumerate(AA) :
      base=np.zeros(25)
      keys.append(key)
      base[i]=1
      vectors.append(base)
   aa_dict = dict(zip(keys, vectors))
   return aa_dict
   
def emb_seq_BE(seq, aa_dict, num):
   seq_emb = np.array([np.array([aa_dict[seq[i + k]] for k in range(num)]).reshape([25 * num]) for i in range(len(seq) - num + 1)])
   return seq_emb
  
def emb_seq_w2v(seq, w2v_model, num):
    seq_emb = np.array([np.array(w2v_model.wv[seq[i:i+num]]) for i in range(len(seq) - num + 1)])
    return seq_emb

class pv_data_sets():
    def __init__(self, data_sets, encode_method, aa_dict, kmer, w2v_model):
        super().__init__()
        self.seq = data_sets["seq"].values.tolist() #"sequence"
        self.labels = np.array(data_sets["label"].values.tolist()).reshape([len(data_sets["label"].values.tolist()),1]).astype(np.float32)
        self.encode_method = encode_method
        self.aa_dict = aa_dict
        self.kmer = kmer
        self.w2v_model = w2v_model
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        if self.encode_method == 'BE':
          emb_mat = emb_seq_BE(self.seq[idx], self.aa_dict, self.kmer) 
        elif self.encode_method == 'W2V':
          emb_mat = emb_seq_w2v(self.seq[idx], self.w2v_model, self.kmer)
        else:
          print('no encoding method')
          exit()
        label = self.labels[idx]       
        return torch.tensor(emb_mat).float().to(device), torch.tensor(label).to(device)

class train_test_process():
    def __init__(self, out_path, loss_type = "balanced", tra_batch_size = 128, val_batch_size = 128, test_batch_size = 32, features = 100, lr = 0.001, n_epoch = 10000, early_stop = 25, thresh = 0.5): #lr = 0.00001,
        self.out_path = out_path
        self.tra_batch_size = tra_batch_size
        self.val_batch_size = val_batch_size
        self.features = features  #not necessary
        self.lr = lr
        self.n_epoch = n_epoch
        self.early_stop = early_stop
        self.thresh = thresh
        self.loss_type = loss_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def training_testing(self, train_data_sets, val_data_sets, test_data_sets, deep_method, encode_method, seqwin, kmer, w2v_model, vector_size):
        os.makedirs(self.out_path + "/data_model", exist_ok=True)
        
        aa_dict = aa_dict_construction()
       
        tra_data_all = pv_data_sets(train_data_sets, encode_method, aa_dict, kmer, w2v_model)
        train_loader = DataLoader(dataset = tra_data_all, batch_size = self.tra_batch_size, shuffle=True)

        val_data_all = pv_data_sets(val_data_sets, encode_method, aa_dict, kmer, w2v_model)
        val_loader = DataLoader(dataset = val_data_all, batch_size = self.val_batch_size, shuffle=True)
        
        if encode_method == 'W2V':
              self.features = vector_size
        else:
              self.features = 25*kmer   
              
        if deep_method == 'CNN':
           net = CNN(features = self.features, time_size = seqwin - kmer + 1).to(device)             
        elif deep_method == 'LSTM':  
           net = Lstm(features = self.features, lstm_hidden_size = 128).to(device)
        elif deep_method == 'bLSTM': 
           net = bLSTM(features = self.features, lstm_hidden_size = 128).to(device)
        elif deep_method == 'TX' :        
           net = TX(n_layers=3, d_model=self.features, n_heads=4, d_dim=100, d_ff=400, time_seq=seqwin-kmer+1).to(device)                         
        else:
           print('no net exist')
           exit()
               
        opt = optim.Adam(params = net.parameters(), lr = self.lr)
         
        if(self.loss_type == "balanced"):
            criterion = nn.BCELoss()
            
        min_loss = 1000
        early_stop_count = 0
        with open(self.out_path + "/cv_result.txt", 'w') as f:
            print(self.out_path, file = f, flush=True)
            print("The number of training data:" + str(len(train_data_sets)), file = f, flush=True)
            print("The number of validation data:" + str(len(val_data_sets)), file = f, flush=True)
                      
            for epoch in range(self.n_epoch):
                train_losses, val_losses = [], []
                self.train_probs, self.train_labels = [], []
                           
                print("epoch_" + str(epoch + 1) + "=====================", file = f, flush=True) 
                print("train...", file = f, flush=True)
                net.train()
                
                for i, (emb_mat, label) in enumerate(train_loader):
                    opt.zero_grad()
                    outputs = net(emb_mat)
                    
                    if(self.loss_type == "balanced"):
                        loss = criterion(outputs, label)
                    elif(self.loss_type == "imbalanced"):
                        loss = CBLoss(label, outputs, 0.9999, 2)
                    else:
                        print("ERROR::You can not specify the loss type.")

                    loss.backward()
                    opt.step()
                    
                    train_losses.append(float(loss.item()))
                    self.train_probs.extend(outputs.cpu().clone().detach().squeeze(1).numpy().flatten().tolist())
                    self.train_labels.extend(label.cpu().clone().detach().squeeze(1).numpy().flatten().tolist())

                train_thresh = 0.5
                print("train_loss:: value: %f, epoch: %d" % (sum(train_losses) / len(train_losses), epoch + 1), file = f, flush=True) 
                print("train_loss:: value: %f, epoch: %d, time: %f" % (sum(train_losses) / len(train_losses), epoch + 1, time.time()-start)) 
                print("val_threshold:: value: %f, epoch: %d" % (train_thresh, epoch + 1), file = f, flush=True)
                for key in metrics_dict.keys():
                    if(key != "auc" and key != "AUPRC"):
                        metrics = metrics_dict[key](self.train_labels, self.train_probs, thresh = train_thresh)
                    else:
                        metrics = metrics_dict[key](self.train_labels, self.train_probs)
                    print("train_" + key + ": " + str(metrics), file = f, flush=True)
                    
                tn_t, fp_t, fn_t, tp_t = cofusion_matrix(self.train_labels, self.train_probs, thresh = train_thresh)
                print("train_true_negative:: value: %f, epoch: %d" % (tn_t, epoch + 1), file = f, flush=True)
                print("train_false_positive:: value: %f, epoch: %d" % (fp_t, epoch + 1), file = f, flush=True)
                print("train_false_negative:: value: %f, epoch: %d" % (fn_t, epoch + 1), file = f, flush=True)
                print("train_true_positive:: value: %f, epoch: %d" % (tp_t, epoch + 1), file = f, flush=True)

                print("validation...", file = f, flush=True)
                
                net.eval()
                self.val_probs, self.val_labels = [], []
                for i, (emb_mat, label) in enumerate(val_loader):
                    with torch.no_grad():
                        outputs = net(emb_mat)
                        
                    if(self.loss_type == "balanced"):
                        loss = criterion(outputs, label)
                    elif(self.loss_type == "imbalanced"):
                        loss = CBLoss(label, outputs, 0.9999, 2)
                    else:
                        print("ERROR::You can not specify the loss type.")

                    if(np.isnan(loss.item()) == False):
                        val_losses.append(float(loss.item()))
                        
                    self.val_probs.extend(outputs.cpu().detach().squeeze(1).numpy().flatten().tolist())
                    self.val_labels.extend(label.cpu().detach().squeeze(1).numpy().flatten().tolist()) 
                
                loss_epoch = sum(val_losses) / len(val_losses)

                val_thresh = 0.5

                print("validation_loss:: value: %f, epoch: %d" % (loss_epoch, epoch + 1), file = f, flush=True)
                print("val_threshold:: value: %f, epoch: %d" % (val_thresh, epoch + 1), file = f, flush=True)
                for key in metrics_dict.keys():
                    if(key != "auc" and key != "AUPRC"):
                        metrics = metrics_dict[key](self.val_labels, self.val_probs, thresh = val_thresh)
                    else:
                        metrics = metrics_dict[key](self.val_labels, self.val_probs)
                    print("validation_" + key + ": " + str(metrics), file = f, flush=True)
                
                tn_t, fp_t, fn_t, tp_t = cofusion_matrix(self.val_labels, self.val_probs, thresh = val_thresh)
                print("validation_true_negative:: value: %f, epoch: %d" % (tn_t, epoch + 1), file = f, flush=True)
                print("validation_false_positive:: value: %f, epoch: %d" % (fp_t, epoch + 1), file = f, flush=True)
                print("validation_false_negative:: value: %f, epoch: %d" % (fn_t, epoch + 1), file = f, flush=True)
                print("validation_true_positive:: value: %f, epoch: %d" % (tp_t, epoch + 1), file = f, flush=True)

                if loss_epoch < min_loss:
                    early_stop_count = 0
                    min_loss = loss_epoch
                    os.makedirs(self.out_path + "/data_model", exist_ok=True)
                    os.chdir(self.out_path + "/data_model")
                    torch.save(net.state_dict(), "deep_model")

                    final_thresh = 0.5
                    final_val_probs = self.val_probs  
                    final_val_labels = self.val_labels
                    final_train_probs = self.train_probs
                    final_train_labels = self.train_labels
                    
                else:
                    early_stop_count += 1
                    if early_stop_count >= self.early_stop:
                        print('Traning can not improve from epoch {}\tBest loss: {}'.format(epoch + 1 - self.early_stop, min_loss), file = f, flush=True)
                        break # Simulation continues until the end of epochs
                    
            #print(val_thresh, file = f, flush=True)
            for key in metrics_dict.keys():
                if(key != "auc" and key != "AUPRC"):
                    train_metrics = metrics_dict[key](final_train_labels,final_train_probs,thresh = final_thresh)
                    val_metrics = metrics_dict[key](final_val_labels,final_val_probs, thresh = final_thresh)
                else:
                    train_metrics = metrics_dict[key](final_train_labels, final_train_probs)
                    val_metrics = metrics_dict[key](final_val_labels, final_val_probs)
                print("train_" + key + ": " + str(train_metrics), file = f, flush=True)
                print("val_" + key + ": " + str(val_metrics), file = f, flush=True)
 
 
        ### testing process        
        with open(self.out_path + "/test_result.txt", 'w') as f:  
            print(self.out_path, file = f, flush=True)
            print("The number of testing data:" + str(len(test_data_sets)), file = f, flush=True)
            
            self.test_probs, self.test_labels = [], []
            
            print("testing...", file = f, flush=True)
            net.eval()
            test_data_all = pv_data_sets(test_data_sets, encode_method, aa_dict, kmer, w2v_model)#
            test_loader = DataLoader(dataset = test_data_all, batch_size = 32, shuffle=False) #batch size 変更
            
            for i, (emb_mat, label) in enumerate(test_loader):
                with torch.no_grad():                  
                    outputs = net(emb_mat) # deep learning model 1個ずつ処理できるのか？
                        
                self.test_probs.extend(outputs.cpu().detach().squeeze(1).numpy().flatten().tolist())
                self.test_labels.extend(label.cpu().detach().squeeze(1).numpy().flatten().tolist()) 
                
            #print("test_threshold:: value: %f" % (str(self.thresh)), file = f, flush=True)
            for key in metrics_dict.keys():
                if(key != "auc" and key != "AUPRC"):
                    test_metrics = metrics_dict[key](self.test_labels, self.test_probs, thresh = self.thresh)
                else:
                    test_metrics = metrics_dict[key](self.test_labels, self.test_probs)
                print("test_" + key + ": " + str(test_metrics), file = f, flush=True)
                
            tn_t, fp_t, fn_t, tp_t = cofusion_matrix(self.test_labels, self.test_probs, thresh = self.thresh)
            print("test_true_negative:: value: %f" % (tn_t), file = f, flush=True)
            print("test_false_positive:: value: %f" % (fp_t), file = f, flush=True)
            print("test_false_negative:: value: %f" % (fn_t), file = f, flush=True)
            print("test_true_positive:: value: %f" % (tp_t), file = f, flush=True)


###############################################################################################################            
start=time.time()
                 
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--intrain', help='Path')
parser.add_argument('-it', '--intest', help='Path')
parser.add_argument('-o', '--outpath', help='Path')
parser.add_argument('-l', '--losstype', help='Path', default = "balanced", choices=["balanced", "imbalanced"])
parser.add_argument('-w', '--w2vmodel', help='Path')
parser.add_argument('-dm', '--deeplearn', help='Path')
parser.add_argument('-en', '--encode', help='Path')
parser.add_argument('-f', '--fold', help='Path')
parser.add_argument('-sw', '--seqwin', help='Path')
parser.add_argument('-k', '--kmer', help='Path')
parser.add_argument('-s', '--size', help='Path')
parser.add_argument('-e', '--epochs', help='Path')
parser.add_argument('-sg', '--sg', help='Path')
parser.add_argument('-window', '--window', help='Path')

path = parser.parse_args().intrain
test_path = parser.parse_args().intest
out_path = parser.parse_args().outpath
loss_type = parser.parse_args().losstype
w2v_model = parser.parse_args().w2vmodel
deep_method = parser.parse_args().deeplearn
encode_method = parser.parse_args().encode
fold = parser.parse_args().fold
seqwin = parser.parse_args().seqwin
kmer = parser.parse_args().kmer
size = parser.parse_args().size
epochs=parser.parse_args().epochs
sg = parser.parse_args().sg
window = parser.parse_args().window
kfold = int(fold)
seqwin = int(seqwin) #4
kmer = int(kmer) #4
vector_size = int(size) #100
epochs = int(epochs) #16
sg = int(sg) # default 1
window = int(window) #

if encode_method == 'W2V':
   w2v_model = word2vec.Word2Vec.load(w2v_model)
   os.makedirs(out_path + '/' + deep_method + '/' + encode_method + '_' + str(kmer) + '_' + str(size) + '_' + str(epochs) + '_' + str(window) + '_' + str(sg), exist_ok=True)
   out_path =  out_path + '/' + deep_method + '/' + encode_method + '_' + str(kmer) + '_' + str(size) + '_' + str(epochs) + '_' + str(window) + '_' + str(sg)

else:
   w2v_model = []
   os.makedirs(out_path + '/' + deep_method + '/' + encode_method + '_' + str(kmer), exist_ok=True)
   out_path =  out_path + '/' + deep_method + '/' + encode_method + '_' + str(kmer)
   
# cross validation
for i in range(1, kfold+1):
    '''
    train_dataset = file_input_csv(path + "/" + str(i) + "/cv_train_" + str(i) + ".csv" ,index_col = None)
    val_dataset = file_input_csv(path + "/" + str(i) + "/cv_val_" + str(i) + ".csv" ,index_col = None)
    test_dataset = file_input_csv(test_path, index_col = None)
    '''    
    train_dataset = pad_input_csv(path + "/" + str(i) + "/cv_train_" + str(i) + ".csv", seqwin, index_col = None)
    val_dataset = pad_input_csv(path + "/" + str(i) + "/cv_val_" + str(i) + ".csv", seqwin, index_col = None)
    test_dataset = pad_input_csv(test_path, seqwin, index_col = None)
        
    net = train_test_process(out_path + "/" + str(i), loss_type = loss_type, features=25*kmer) 
    net.training_testing(train_dataset, val_dataset, test_dataset, deep_method, encode_method, seqwin, kmer, w2v_model, vector_size)
    
    output = pd.DataFrame([net.train_probs, net.train_labels], index = ["prob", "label"]).transpose()
    output.to_csv(out_path + "/" + str(i) + "/train_roc.csv")
    output = pd.DataFrame([net.val_probs, net.val_labels], index = ["prob", "label"]).transpose()
    output.to_csv(out_path + "/" + str(i) + "/val_roc.csv")
    
    #independent test
    output = pd.DataFrame([net.test_probs, net.test_labels], index = ["prob", "label"]).transpose()
    output.to_csv(out_path + "/" + str(i) + "/test_roc.csv")

print('total time:', time.time()-start)


