import os
import time
import argparse
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from gensim.models import word2vec
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import r2_score,  mean_squared_error, mean_absolute_error

def emb_seq_w2v(seq_mat, w2v_model, num):
    num_sample = len(seq_mat)   
    for j in range(num_sample):
      seq=seq_mat[j]
      if j == 0:
         seq_emb = np.array([np.array(w2v_model.wv[seq[i:i+num]]) for i in range(len(seq) - num + 1)])
      else: 
         seq_enc = np.array([np.array(w2v_model.wv[seq[i:i+num]]) for i in range(len(seq) - num + 1)])
         seq_emb = np.append(seq_emb, seq_enc, axis=0)
         
    seq_emb = seq_emb.reshape(num_sample,len(seq) - num + 1, -1)  #1088 x 35 x 100
    seq_emb = seq_emb.reshape(num_sample,1,-1).squeeze() #1088 X3500
    return seq_emb

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
   
def emb_seq_BE(seq_mat, aa_dict, num):
   num_sample = len(seq_mat) 
   for j in range(num_sample):
      seq = seq_mat[j]
      if j == 0:
         seq_emb = np.array([np.array([aa_dict[seq[i + k]] for k in range(num)]).reshape([25 * num]) for i in range(len(seq) - num + 1)])
      else : 
         seq_enc = np.array([np.array([aa_dict[seq[i + k]] for k in range(num)]).reshape([25 * num]) for i in range(len(seq) - num + 1)])
         seq_emb = np.append(seq_emb, seq_enc, axis=0)
   seq_emb = seq_emb.reshape(num_sample, len(seq) - num + 1, -1)  #1088 x 35 x 100
   seq_emb = seq_emb.reshape(num_sample, 1, -1).squeeze() #1088 X3500
   return seq_emb
   
#dataset reading
def pad_input_csv(filename, seqwin, index_col = None):
    df1 = pd.read_csv(filename, delimiter=',',index_col = index_col)
    seq = df1.loc[:,'seq'].tolist()
    #data triming and padding
    for i in range(len(seq)):
       if len(seq) > seqwin:
         seq[i]=seq[i][0:seqwin]
       seq[i] = seq[i].ljust(seqwin, 'X')
    for i in range(len(seq)):
       df1.loc[i,'seq'] = seq[i]   
    return df1
      
def pickle_save(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)

def pickle_read(path):
    with open(path, "rb") as f:
        res = pickle.load(f)      
    return res
    
def pickle_dump(obj, path):
    with open(path, mode='wb') as f:
        pickle.dump(obj,f)

def pickle_load(path):
    with open(path, mode='rb') as f:
        data = pickle.load(f)
        return data    


#############################################################################################
start = time.time()

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
test_file = parser.parse_args().intest
out_path_0 = parser.parse_args().outpath
loss_type = parser.parse_args().losstype
w2v_model = parser.parse_args().w2vmodel

machine_method = parser.parse_args().deeplearn
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

#setting
#data_path ="/home/kurata/myproject/py3/ml_av1/data/dataset"
#out_path_0 = "/home/kurata/myproject/py3/ml_av1/data/result"
#w2v_path = '/home/kurata/myproject/py3/ml_av1/data/w2v_model'


for dummy in range(1):
   if encode_method == 'W2V':
      #w2v_model = w2v_path + '/' + 'av_w2v_%s_100_4_20_1.pt' %kmer
      w2v_model = word2vec.Word2Vec.load(w2v_model)
      os.makedirs(out_path_0 + '/' + machine_method + '/' + encode_method + '_' + str(kmer) + '_' + str(size) + '_' + str(epochs) + '_' + str(window) + '_' + str(sg), exist_ok=True)
      out_path =  out_path_0 + '/' + machine_method + '/' + encode_method + '_' + str(kmer) + '_' + str(size) + '_' + str(epochs) + '_' + str(window) + '_' + str(sg)
   else:
      os.makedirs(out_path_0 + '/' + machine_method + '/' + encode_method + '_' + str(kmer), exist_ok=True)
      out_path =  out_path_0 + '/' + machine_method + '/' + encode_method + '_' + str(kmer)
       
   for i in range(1, kfold+1):
      os.makedirs(out_path + "/" + str(i) + "/data_model", exist_ok=True)
      modelname= "machine_model.sav"

      train_dataset = pad_input_csv(path + "/" + str(i) + "/cv_train_" + str(i) + ".csv", seqwin, index_col = None)
      val_dataset = pad_input_csv(path + "/" + str(i) + "/cv_val_" + str(i) + ".csv", seqwin, index_col = None)
      test_dataset = pad_input_csv(test_file, seqwin, index_col = None)

      train_seq = train_dataset['seq'].tolist()
      val_seq = val_dataset['seq'].tolist()
      test_seq = test_dataset['seq'].tolist()
   
      if encode_method == 'W2V':
         train_X = emb_seq_w2v(train_seq, w2v_model, kmer)
         train_y = train_dataset['label'].to_numpy() 
         valid_X = emb_seq_w2v(val_seq, w2v_model, kmer)
         valid_y = val_dataset['label'].to_numpy() 
         test_X = emb_seq_w2v(test_seq, w2v_model, kmer)
         test_y = test_dataset['label'].to_numpy()
      else:
         aa_dict = aa_dict_construction()
         train_X = emb_seq_BE(train_seq, aa_dict, kmer)
         train_y = train_dataset['label'].to_numpy() 
         valid_X = emb_seq_BE(val_seq, aa_dict, kmer)
         valid_y = val_dataset['label'].to_numpy() 
         test_X = emb_seq_BE(test_seq, aa_dict, kmer)
         test_y = test_dataset['label'].to_numpy() 
      
      cv_result = np.zeros((len(valid_y), 2))
      cv_result[:, 1] = valid_y
      test_result = np.zeros((len(test_y), 2))
      test_result[:,1] = test_y   #score:one of two, label
     
      
      if machine_method == 'RF':
         model = RandomForestClassifier(max_depth=4, random_state=0, n_estimators=100)
         clf = model.fit(train_X, train_y)
      elif machine_method == 'SVM':    
         model = svm.SVC(probability=True)
         clf = model.fit(train_X, train_y)
      else:
         print('No learning method')
   
      pickle.dump(clf, open(out_path + "/" + str(i) + "/data_model/machine_model.asv",'wb'))
      #clf= pickle.load(open(out_path + "/" + str(i) + "/data_model/machine_model.asv",'rb'))
   
      #CV
      score = clf.predict_proba(valid_X)
      cv_result[:, 0] = score[:,1]
      print(cv_result)

      #independent test
      if test_dataset.shape[0] != 0:
         test_result[:, 0] = clf.predict_proba(test_X)[:,1]
      print(test_result)
         
      #CV  
      cv_output = pd.DataFrame(cv_result,  columns=['prob', 'label'] )
      cv_output.to_csv(out_path  + "/" + str(i) + "/val_roc.csv")  #prob, label

      #independent test
      test_output = pd.DataFrame(test_result,  columns=['prob', 'label'] )
      test_output.to_csv(out_path  + "/" + str(i) + "/test_roc.csv")  #prob, label
   
print('elapsed time', time.time() - start)

