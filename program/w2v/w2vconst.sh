#!/bin/sh
home_path=../..

program=${home_path}/program/w2v
data_path=${home_path}/data/dataset
w2v_path=${home_path}/data/w2v_model
infile1=${home_path}/data/dataset/AVP_train.txt
infile2=${home_path}/data/dataset/ACVP_M_test.txt

seqwin=40
size=100 
epochs=4 #20
sg=1  # 0 1
window=20 
              
for kmer in 1 
do
python word2vec_acvp.py --infile1 ${infile1} --infile2 ${infile2} --w2v ${w2v_path} --seqwin ${seqwin} --kmer ${kmer} --size ${size} --epochs ${epochs} --sg ${sg} --window ${window}
done

