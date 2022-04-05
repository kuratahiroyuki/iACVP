#!/bin/bash
current_path=/home/kurata/myproject/py3/ACVP_predictor

train_path=${current_path}/data/dataset/cross_val
test_file=${current_path}/data/dataset/independent_test/independent_test.csv
result_path=${current_path}/data/result

kfold=5
kmer=1
seqwin=40

deep_method=RF # RF SVM
for encode_method in AAC CTDT PAAC CKSAAP CKSAAGP AAINDEX BLOSUM62 
do 
echo ${encode_method}
python conv_train_test.py  --intrain ${train_path} --intest ${test_file} --outpath ${result_path} --deeplearn ${deep_method}  --encode ${encode_method} --fold ${kfold} --seqwin ${seqwin} --kmer ${kmer}
done


