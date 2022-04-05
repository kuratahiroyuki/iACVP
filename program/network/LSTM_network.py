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

class Lstm(nn.Module):
    def __init__(self, features, lstm_hidden_size):
        super(Lstm, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size
        
        self.lstm = nn.LSTM(features, self.lstm_hidden_size, batch_first=True)
        
        self.dense_1 = nn.Linear(self.lstm_hidden_size, 1)
        self.sigmoid_func = nn.Sigmoid()

    def forward(self, emb_mat):
        hidden_state, _ = self.lstm(emb_mat)
        
        return self.sigmoid_func(self.dense_1(hidden_state[:, -1]))

