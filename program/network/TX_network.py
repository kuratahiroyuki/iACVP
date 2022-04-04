#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Kyutech Kurata laboratory 2021

import math
import torch
import torch.nn as nn
import numpy as np


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        #self.dropout = nn.Dropout(p = 0.3)

    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        #print("aaa")
        #scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        #print("bbb")
        attn = nn.Softmax(dim=-1)(scores)
        #attn = self.dropout(attn)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_dim):
        super(MultiHeadAttention, self).__init__()
        self.d_q = d_dim
        self.d_k = d_dim
        self.d_v = d_dim
        self.n_heads = n_heads
        self.d_model = d_model
        
        self.W_Q = nn.Linear(d_model, self.d_q * self.n_heads)
        self.W_K = nn.Linear(d_model, self.d_k * self.n_heads)
        self.W_V = nn.Linear(d_model, self.d_v * self.n_heads)
        self.dense_l = nn.Linear(self.n_heads * self.d_v, self.d_model)
        self.layernorm = nn.LayerNorm(self.d_model)
    def forward(self, Q, K, V):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_q).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        #attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)# attn_mask : [batch_size x n_heads x len_q x len_k]

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = ScaledDotProductAttention(self.d_k)(q_s, k_s, v_s)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v) # context: [batch_size x len_q x n_heads * d_v]
        output = self.dense_l(context)
        return self.layernorm(output + residual), attn # output: [batch_size x len_q x d_model]

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        #self.dropout = nn.Dropout(p = 0.3)

    def forward(self, x):
        residual = x
        # (batch_size, len_seq, d_model) -> (batch_size, len_seq, d_ff) -> (batch_size, len_seq, d_model)
        return self.layer_norm(self.fc2(gelu(self.fc1(x))) + residual)
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_dim, d_ff):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, n_heads, d_dim)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, enc_inputs):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn
    
class TX(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_dim, d_ff, time_seq):
        super(TX, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_dim, d_ff) for _ in range(n_layers)])
        self.dense_1 = nn.Linear(d_model * time_seq, 1)
        self.sigmoid_func = nn.Sigmoid()
    
    def forward(self, output):
        self.attn_list = []
        for layer in self.layers:
            output, enc_self_attn = layer(output)
            self.attn_list.append(enc_self_attn)
   
        output = output.view(output.size(0), -1)
        self.interim_output = output # Kurata added it 

        return self.sigmoid_func(self.dense_1(output))
    

