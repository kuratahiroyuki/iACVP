#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.metrics import precision_recall_curve
from metrics import cofusion_matrix, sensitivity, specificity, auc, mcc, accuracy, precision, recall, f1, cutoff, AUPRC

metrics_dict = {"sensitivity":sensitivity, "specificity":specificity, "accuracy":accuracy,"mcc":mcc,"auc":auc,"precision":precision,"recall":recall,"f1":f1,"AUPRC":AUPRC}

def measure_evaluation(score, inpath, roc_file, kfold):  
   for i in range(kfold):
      infile = inpath + '/' + str(i+1) + '/' + roc_file
      result = np.loadtxt(infile, delimiter=',', skiprows=1)

      prob=result[:,1]
      label=result[:,2]
         
      for key in metrics_dict.keys():
            if(key != "auc" and key != "AUPRC"):
                    test_metrics = metrics_dict[key](label, prob, thresh = 0.5)
            else:
                    test_metrics = metrics_dict[key](label, prob)
            #print("test_" + key + ": " + str(test_metrics),  flush=True)
            
            if key =='sensitivity'  :
               score.iloc[i,0]= metrics_dict[key](label, prob, thresh = 0.5)
            elif key =='specificity' :
               score.iloc[i,1]= metrics_dict[key](label, prob, thresh = 0.5)
            elif key =='accuracy' :
               score.iloc[i,2]= metrics_dict[key](label, prob, thresh = 0.5)
            elif key =='mcc' :
               score.iloc[i,3]= metrics_dict[key](label, prob, thresh = 0.5)
            elif key =='auc' :
               score.iloc[i,4]= metrics_dict[key](label, prob)   
            elif key =='precision' :                    
               score.iloc[i,5]= metrics_dict[key](label, prob, thresh = 0.5)  
            elif key =='recall' :
               score.iloc[i,6]= metrics_dict[key](label, prob, thresh = 0.5)                            
            elif key =='f1' :
               score.iloc[i,7]= metrics_dict[key](label, prob, thresh = 0.5)      
            elif key =='AUPRC' :
               score.iloc[i,8]= metrics_dict[key](label, prob)                            
            else:
               continue
                
            tn_t, fp_t, fn_t, tp_t = cofusion_matrix(label, prob, thresh = 0.5)
            #print("test_true_negative:: value: %f" % (tn_t), flush=True)
            #print("test_false_positive:: value: %f" % (fp_t), flush=True)
            #print("test_false_negative:: value: %f" % (fn_t), flush=True)
            #print("test_true_positive:: value: %f" % (tp_t), flush=True)
                               
   means=score.astype(float).mean(axis='index')
   means=pd.DataFrame(np.array(means).reshape(1,-1), index= ['means'], columns=columns_measure)
   score=pd.concat([score, means])
   
   return score


####################################################################################
data_path='../data/result'

kfold=5
w2v_param ='kmer' #'kmer', 'epochs', 'size', 'sg', 'window' 
deep_method_item = ['RF', 'TX'] # 'RF' 'SVM' 'TX','CNN','bLSTM'
encode_method_item = ['W2V','BE'] # 'W2V', 'BE'

test_roc_file='test_roc.csv' # input
val_roc_file='val_roc.csv'
val_measure ='val_measures.csv' # output
test_measure ='test_measures.csv'

index_fold =[i+1 for i in range(kfold)]
if w2v_param == 'kmer':
   index_value = [1,2,3,4,5,6,7,8,9,10] #kmer
elif w2v_param == 'epochs':
   index_value = [2, 4, 8, 12, 16, 20, 30, 40, 50, 100] #epochs
elif w2v_param == 'size':
   index_value = [2, 4, 8, 16, 32, 64, 100, 128, 256, 512] #size
elif w2v_param == 'sg':
   index_value =[1,2,3,4,5,6,7,8,9,10] #kmer
else: 
   index_value = [10, 20, 30, 40] #window

columns_measure= ['Sensitivity', 'Specificity', 'Accuracy', 'MCC', 'AUC', 'Precision', 'Recall', 'F1','AUPRC']
score_test = pd.DataFrame(data=[], index=index_value, columns=columns_measure)
score_val  = pd.DataFrame(data=[], index=index_value, columns=columns_measure)

for encode_method in encode_method_item: #['W2V', 'BE']
   if w2v_param == 'kmer':
      param = '%s_kmer' %encode_method
   elif w2v_param == 'epochs':
      param = '%s_epochs' %encode_method
   elif w2v_param == 'size':
      param = '%s_size' %encode_method
   elif w2v_param == 'sg':
      param = '%s_sg' %encode_method
   else:
      param = '%s_window' %encode_method
   for deep_method in deep_method_item :   # ['RF'] ['TX','CNN','bLSTM']
      for j in index_value:
         
         if encode_method == 'W2V':
            if w2v_param == 'kmer':
               encode_method_2='W2V_%s_100_4_20_1' % j # kmer
            elif w2v_param == 'epochs':
               encode_method_2='W2V_6_100_%s_20_1' % j # epochs
            elif w2v_param == 'size':
               encode_method_2='W2V_6_%s_4_20_1' % j # size
            elif w2v_param == 'sg':
               encode_method_2='W2V_%s_100_4_20_0' % j # kmer 
            else:
               encode_method_2='W2V_6_100_4_%s_1' % j # window          
         else:
            encode_method_2 ='BE_%s' % j
   
         inpath = data_path + '/' + deep_method + '/' + encode_method_2
         outpath= data_path + '/' + deep_method
         
         score=pd.DataFrame(data=[], index=index_fold, columns=columns_measure)
         scores = measure_evaluation(score, inpath, val_roc_file, kfold)
         scores.to_csv('%s/val_measures.csv' %inpath, header=True, index=True)
         score_val.loc[j,:] = scores.loc['means',:] 
               
         score=pd.DataFrame(data=[], index=index_fold, columns=columns_measure)
         scores = measure_evaluation(score, inpath, test_roc_file, kfold)
         scores.to_csv('%s/test_measures.csv' %inpath, header=True, index=True)
         score_test.loc[j,:] = scores.loc['means',:]
               
      print(score_val)
      print(score_test)  
      score_test.to_csv('%s/%s_'% (outpath, param) + test_measure, header=True, index=True)
      score_val.to_csv('%s/%s_'% (outpath, param) + val_measure, header=True, index=True)     

