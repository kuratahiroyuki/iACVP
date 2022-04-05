#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import torch
import torch.nn.utils.rnn as rnn

class bLSTM(nn.Module):
    def __init__(self, features, lstm_hidden_size):
        super(bLSTM, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size
        
        self.lstm = nn.LSTM(features, self.lstm_hidden_size, batch_first=True, bidirectional=True)
        
        self.dense_1 = nn.Linear(self.lstm_hidden_size*2, 1)
        self.sigmoid_func = nn.Sigmoid()

    def forward(self, emb_mat):
        hidden_state, _ = self.lstm(emb_mat)
        
        out_final = hidden_state[:, -1][:, :self.lstm_hidden_size]
        out_first = hidden_state[:, 0][:, self.lstm_hidden_size:]
        bilstm_out = torch.cat([out_final, out_first], dim = 1)
        
        return self.sigmoid_func(self.dense_1(bilstm_out))

