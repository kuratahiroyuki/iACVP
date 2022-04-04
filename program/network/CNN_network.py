#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 19:48:12 2020

@author: kurata
"""
#old verison
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
#from pytorch_model_summary import summary
import numpy as np
import copy
import torch
import torch.nn.utils.rnn as rnn

class CNN(nn.Module):
    def __init__(self, features, time_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(features, 32, kernel_size=5, stride=1, padding = 2)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=5, stride=1, padding = 2)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dense = nn.Linear(32 * int(int(time_size/2)/2),1)
        self.relu = nn.ReLU()
        self.sigmoid_func = nn.Sigmoid()
        self.dropout = nn.Dropout(p = 0.4)

    def forward(self, emb_mat):
        output = torch.transpose(emb_mat, -1, -2)
        output = self.conv1(output)
        output = self.relu(output)
        output = self.maxpool(output)
        output = self.dropout(output)
        
        output = self.conv2(output)
        output = self.relu(output)
        output = self.maxpool(output)
        output = self.dropout(output)
        
        output = output.view(-1, output.size(1) * output.size(2))
        
        return self.sigmoid_func(self.dense(output))

"""
#a = torch.ones(10, 7)
#b = torch.rand(10, 7)
#c = torch.rand(10, 7)
vec = torch.ones(20, 26, 100)
#vec = rnn.pad_sequence([a,b,c], batch_first=True)
#vec = rnn.pack_padded_sequence(vec, [10,10,10], batch_first=True, enforce_sorted=False).float()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = CNN(features = 100)
output = net.forward(vec.float())

temp = output.detach().numpy()
temp_2 = output[:, -1].detach().numpy()

#temp = output.detach().numpy()


a = torch.ones(10, 7)
b = torch.ones(6, 7) * 4
c = torch.ones(6, 7) * 3
h_vec = rnn.pad_sequence([a,b,c], batch_first=True)
h_vec = rnn.pack_padded_sequence(h_vec, [10,6,6], batch_first=True, enforce_sorted=False).cuda().float()
a = torch.ones(2, 7)
b = torch.ones(3, 7) * 2
c = torch.ones(4, 7) * 5
v_vec = rnn.pad_sequence([a,b,c], batch_first=True)
v_vec = rnn.pack_padded_sequence(v_vec, [2, 3, 4], batch_first=True, enforce_sorted=False).cuda().float()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = unitNET(device, features = 7).cuda()
out = net.forward(h_vec.cuda().float(), v_vec.cuda().float())



 

path="/Users/tsukiyamashou/Desktop/kurata_lab/data/data_sets"
human_vector=load_pickle("train_human_vector.joblib",path)[0:10,:,:]
virus_vector=load_pickle("train_virus_vector.joblib",path)[0:10,:,:]
human_length=load_pickle("train_human_length.joblib",path)
virus_length=load_pickle("train_virus_length.joblib",path)
label=load_pickle("train_label.joblib",path)[:10,:]

print("data was loaded")

human_length = human_length.reshape([len(human_length)])[0:10]
virus_length = virus_length.reshape([len(virus_length)])[0:10]

human_vector = torch.from_numpy(human_vector.astype(np.float32)).clone()
virus_vector = torch.from_numpy(virus_vector.astype(np.float32)).clone()
human_length = torch.from_numpy(human_length.astype(np.float32)).clone()
virus_length = torch.from_numpy(virus_length.astype(np.float32)).clone()


human_we_mat = torch.nn.utils.rnn.pack_padded_sequence(human_vector,human_length,batch_first=True,enforce_sorted=False)
virus_we_mat = torch.nn.utils.rnn.pack_padded_sequence(virus_vector,virus_length,batch_first=True,enforce_sorted=False)

human_cp_mat,_=torch.nn.utils.rnn.pad_packed_sequence(copy.deepcopy(human_we_mat),batch_first =True)
virus_cp_mat,_=torch.nn.utils.rnn.pad_packed_sequence(copy.deepcopy(virus_we_mat),batch_first =True)

features=21
lstm_hidden_size = 100
dense_hidden = 50


deepnet = unitNET(features, lstm_hidden_size, dense_hidden).to("cpu")
print(summary(deepnet,human_we_mat,human_cp_mat,virus_we_mat,virus_cp_mat))
#temp_1 = deepnet.att_wieght_h.data.detach().numpy().copy()
#temp_2 = deepnet.temp_h.data.detach().numpy().copy()
 


#human_net=unitNET(features, lstm_hidden_size_w, dense_hidden_w, main_lstm_hidden_w)

#virus_net=unitNET(features, lstm_hidden_size_w, dense_hidden_w, main_lstm_hidden_w)

#lstm_w = nn.LSTM(features,lstm_hidden_size_w,batch_first=True, bidirectional=True)
#lstm_m = nn.LSTM(features,main_lstm_hidden_w,batch_first=True, bidirectional=True)

#dense_w_1 = nn.Linear(lstm_hidden_size_w*2,dense_hidden_w)
#dense_w_2 = nn.Linear(dense_hidden_w,1)

#batch_norm = nn.BatchNorm1d(main_lstm_hidden_w*2)

#broad_one = torch.ones(1,features)


lstm_human = nn.LSTM(features,lstm_hidden_size,batch_first=True, bidirectional=True)
lstm_virus = nn.LSTM(features,lstm_hidden_size,batch_first=True, bidirectional=True)
        
dense_1_human = nn.Linear(lstm_hidden_size*2,dense_hidden_1)
dense_2_human = nn.Linear(dense_hidden_1,1)
dense_1_virus = nn.Linear(lstm_hidden_size*2,dense_hidden_1)
dense_2_virus = nn.Linear(dense_hidden_1,1)
        
dense_both_1 = nn.Linear(cla_lstm_hidden*2*2,200)
dense_both_2 = nn.Linear(200,100)
dense_both_3 = nn.Linear(100,50)
dense_both_4 = nn.Linear(50,1)
        
batch_norm_human = nn.BatchNorm1d(cla_lstm_hidden*2)
batch_norm_virus = nn.BatchNorm1d(cla_lstm_hidden*2)
        
batch_norm_both_1 = nn.BatchNorm1d(200)
batch_norm_both_2 = nn.BatchNorm1d(100)
batch_norm_both_3 = nn.BatchNorm1d(50)
        
broad_one = torch.ones(1,lstm_hidden_size*2)


output_h, _ = lstm_human(human_we_mat)
output_v, _ = lstm_virus(virus_we_mat)

hidden_state_h, length_h = torch.nn.utils.rnn.pad_packed_sequence(output_h,batch_first =True)
hidden_state_v, length_v = torch.nn.utils.rnn.pad_packed_sequence(output_v,batch_first =True)

output_h = F.relu(dense_1_human(hidden_state_h))
output_v = F.relu(dense_1_virus(hidden_state_v))

output_h = dense_2_human(output_h)
output_v = dense_2_virus(output_v)

for i in range(output_h.size()[0]):
    output_h[i,0:length_h[i],:] = F.softmax(output_h[i,0:length_h[i],:].detach(),dim=0)
    output_v[i,0:length_v[i],:] = F.softmax(output_v[i,0:length_v[i],:].detach(),dim=0)
    
att_wieght_h = output_h
att_wieght_v = output_v

output_h = torch.matmul(att_wieght_h,broad_one)
output_v = torch.matmul(att_wieght_v,broad_one)
        
output_h = hidden_state_h*output_h
output_v = hidden_state_v*output_v

output_h = output_h.sum(dim=1)
output_v = output_v.sum(dim=1)


output = torch.cat([output_h,output_v],dim=1)

output = F.sigmoid(dense_both_1(output))
output = batch_norm_both_1(output)
        
output = F.sigmoid(dense_both_2(output))
output = batch_norm_both_2(output)
        
output = F.sigmoid(dense_both_3(output))
output = batch_norm_both_3(output)
        
output = F.sigmoid(dense_both_4(output))
"""


#output = torch.cat([output_1,output_2],dim=1)


#out1=output_h.detach().numpy().copy()
#out2=hidden_state_h.detach().numpy().copy()
#out2=temp.detach().numpy().copy()
#out3=output.detach().numpy().copy()
#output_lengths.detach().numpy().copy()
#hidd=hidden[0].to('cpu').detach().numpy().copy()
#hidde=hidden[1].to('cpu').detach().numpy().copy()

#out=output[0].to('cpu').detach().numpy().copy()































