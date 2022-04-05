#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix

def sensitivity(y_true, y_prob, thresh=0.8):
    #y_true = y_true.cpu().detach().numpy()
    #y_prob = (y_prob.cpu().detach().numpy() + 1 - thresh).astype(np.int16)
    y_true = np.array(y_true)
    y_prob = (np.array(y_prob) + 1 -thresh).astype(np.int16)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_prob).ravel()
    return tp / (tp + fn)


def specificity(y_true, y_prob, thresh=0.8):
    #y_true = y_true.cpu().detach().numpy()
    #y_prob = (y_prob.cpu().detach().numpy() + 1 - thresh).astype(np.int16)
    y_true = np.array(y_true)
    y_prob = (np.array(y_prob) + 1 -thresh).astype(np.int16)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_prob).ravel()
    return tn / (tn + fp)


def auc(y_true, y_prob):
    #y_true = y_true.cpu().detach().numpy()
    #y_prob = y_prob.cpu().detach().numpy()
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    return metrics.roc_auc_score(y_true, y_prob)

def mcc(y_true, y_prob, thresh=0.8):
    #y_true = y_true.cpu().detach().numpy()
    #y_prob = (y_prob.cpu().detach().numpy() + 1 - thresh).astype(np.int16)
    y_true = np.array(y_true)
    y_prob = (np.array(y_prob) + 1 -thresh).astype(np.int16)
    return metrics.matthews_corrcoef(y_true, y_prob)

def accuracy(y_true, y_prob, thresh=0.8):
    #y_true = y_true.cpu().detach().numpy()
    #y_prob = (y_prob.cpu().detach().numpy() + 1 - thresh).astype(np.int16)
    y_true = np.array(y_true)
    y_prob = (np.array(y_prob) + 1 -thresh).astype(np.int16)
    return metrics.accuracy_score(y_true, y_prob)

def cutoff(y_true,y_prob):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob,drop_intermediate=False)
    #return thresholds[np.argmin(np.sqrt(((1-tpr)**2)+(fpr**2)))],fpr, tpr, thresholds
    return thresholds[np.argmax(np.array(tpr) - np.array(fpr))],np.array(tpr) - np.array(fpr), fpr, tpr, thresholds

def precision(y_true, y_prob, thresh = 0.8):
    y_true = np.array(y_true)
    y_prob = (np.array(y_prob) + 1 -thresh).astype(np.int16)
    return metrics.precision_score(y_true,y_prob)

def recall(y_true, y_prob, thresh = 0.8):
    y_true = np.array(y_true)
    y_prob = (np.array(y_prob) + 1 -thresh).astype(np.int16)
    return metrics.recall_score(y_true,y_prob)

def f1(y_true, y_prob, thresh = 0.8):
    y_true = np.array(y_true)
    y_prob = (np.array(y_prob) + 1 -thresh).astype(np.int16)
    return metrics.f1_score(y_true,y_prob)

def AUPRC(y_true, y_prob):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    return metrics.average_precision_score(y_true, y_prob)

def cofusion_matrix(y_true,y_prob, thresh = 0.8):
    y_true = np.array(y_true)
    y_prob = (np.array(y_prob) + 1 -thresh).astype(np.int16)
    tn, fp, fn, tp = confusion_matrix(y_true, y_prob).ravel()
    #tn, fp, fn, tp = cm.flatten()

    return tn, fp, fn, tp


