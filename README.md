# iACVP
This package is used for prediction of anti-coronavirus peptides (ACVPs).

# Environment
 >anaconda 4.11.0  
 >python 3.8.8  
 >pytorch 1.9.0  
 >scikit-learn 1.0.2  
 >gensim 4.0.1
  
# Execution
# 1 Setting iACVP directory  
 The directory structure given in the github should be conserved. 
 
# 2 Construction of training and test datasets  
./  
$sh dataconst.sh  
 (train_division.py,  test_fasta.py)  
 
# 3 Word2vec model construction  
./program/w2v  
$sh w2vconst.sh  
 (word2vec_acvp.py)  
 
**NOTE**  
 Use epochs=20  for prediction of AVPs  
 Use epochs=4   for prediction of ACVPs  
 
# 4 Training and testing of W2V-based and BE-based classifiers  
## 4-1 Simulation  
./program/network  
$sh main.sh  
 (deep_train_test.py, machine_train_test.py)  

## 4-2 Analysis (calculation of measures)  
./program  
$python analysis_net.py  　

# 5. Training and testing of RF and SVM classifiers with conventional encodings  
## 5-1 Simulation  
./program/ml  
$sh submain.sh  
 (conv_train_test.py)  

## 5-2 Analysis (calculation of measures)  
./program  
$python analysis_ml.py   

