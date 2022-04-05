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

