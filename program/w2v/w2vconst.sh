#!/bin/sh
current_path=/home/kurata/myproject/py3/ACVP_predictor

program=${current_path}/program/w2v
data_path=${current_path}/data/dataset
w2v_path=${current_path}/data/w2v_model
infile1=${current_path}/data/dataset/AVP_train.txt
infile2=${current_path}/data/dataset/ACVP_M_test.txt

seqwin=40
size=100 
epochs=4 #20
sg=1  # 0 1
window=20 
              
for kmer in 1 2 3 4 5 6 7 8 9 10
do
python ${program}/word2vec_acvp.py --infile1 ${infile1} --infile2 ${infile2} --w2v ${w2v_path} --seqwin ${seqwin} --kmer ${kmer} --size ${size} --epochs ${epochs} --sg ${sg} --window ${window}
done

