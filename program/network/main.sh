#!/bin/bash
home_path=../..

train_path=${home_path}/data/dataset/cross_val
test_file=${home_path}/data/dataset/independent_test/independent_test.csv
result_path=${home_path}/data/result
w2v_path=${home_path}/data/w2v_model

kfold=5 
seqwin=40
size=100
epochs=4 
sg=1
window=20

deep_method=RF  #RF, SVM
encode_mathod=W2V # W2V, BE
for kmer in 1 #2 3 4 5 6 7 8 9 10
do
w2v_model=${w2v_path}/av_w2v_${kmer}_${size}_${epochs}_${window}_${sg}.pt

python machine_train_test.py  --intrain ${train_path} --intest ${test_file} --outpath ${result_path} --losstype "balanced" --deeplearn ${deep_method}  --encode ${encode_mathod} --fold ${kfold} --w2vmodel ${w2v_model} --seqwin ${seqwin} --kmer ${kmer} --size ${size} --epochs ${epochs} --sg ${sg} --window ${window}
done

exit 1

deep_method=RF  #RF, SVM
encode_mathod=BE # W2V, BE
for kmer in 1 2 3 4 5 6 7 8 9 10
do
w2v_model=${w2v_path}/av_w2v_${kmer}_${size}_${epochs}_${window}_${sg}.pt

python machine_train_test.py  --intrain ${train_path} --intest ${test_file} --outpath ${result_path} --losstype "balanced" --deeplearn ${deep_method}  --encode ${encode_mathod} --fold ${kfold} --w2vmodel ${w2v_model} --seqwin ${seqwin} --kmer ${kmer} --size ${size} --epochs ${epochs} --sg ${sg} --window ${window}
done

deep_method=SVM  #RF, SVM
encode_mathod=W2V # W2V, BE
for kmer in 1 2 3 4 5 6 7 8 9 10
do
w2v_model=${w2v_path}/av_w2v_${kmer}_${size}_${epochs}_${window}_${sg}.pt

python machine_train_test.py  --intrain ${train_path} --intest ${test_file} --outpath ${result_path} --losstype "balanced" --deeplearn ${deep_method}  --encode ${encode_mathod} --fold ${kfold} --w2vmodel ${w2v_model} --seqwin ${seqwin} --kmer ${kmer} --size ${size} --epochs ${epochs} --sg ${sg} --window ${window}
done

deep_method=SVM  #RF, SVM
encode_mathod=BE # W2V, BE
for kmer in 1 2 3 4 5 6 7 8 9 10
do
w2v_model=${w2v_path}/av_w2v_${kmer}_${size}_${epochs}_${window}_${sg}.pt

python machine_train_test.py  --intrain ${train_path} --intest ${test_file} --outpath ${result_path} --losstype "balanced" --deeplearn ${deep_method}  --encode ${encode_mathod} --fold ${kfold} --w2vmodel ${w2v_model} --seqwin ${seqwin} --kmer ${kmer} --size ${size} --epochs ${epochs} --sg ${sg} --window ${window}
done


deep_method=TX  #TX, CNN, bLSTM
encode_mathod=W2V # W2V, BE
for kmer in 1 2 3 4 5 6 7 8 9 10 
do
w2v_model=${w2v_path}/av_w2v_${kmer}_${size}_${epochs}_${window}_${sg}.pt
python deep_train_test.py  --intrain ${train_path} --intest ${test_file} --outpath ${result_path} --losstype "balanced" --deeplearn ${deep_method}  --encode ${encode_mathod} --fold ${kfold} --w2vmodel ${w2v_model} --seqwin ${seqwin} --kmer ${kmer} --size ${size} --epochs ${epochs} --sg ${sg} --window ${window}
done


deep_method=TX  #TX, CNN, bLSTM
encode_mathod=BE # W2V, BE
for kmer in 1 2 3 4 5 6 7 8 9 10 
do
w2v_model=${w2v_path}/av_w2v_${kmer}_${size}_${epochs}_${window}_${sg}.pt

python deep_train_test.py  --intrain ${train_path} --intest ${test_file} --outpath ${result_path} --losstype "balanced" --deeplearn ${deep_method}  --encode ${encode_mathod} --fold ${kfold} --w2vmodel ${w2v_model} --seqwin ${seqwin} --kmer ${kmer} --size ${size} --epochs ${epochs} --sg ${sg} --window ${window}
done

deep_method=CNN  #TX, CNN, bLSTM
encode_mathod=W2V # W2V, BE
for kmer in 1 2 3 4 5 6 7 8 9 10 
do
w2v_model=${w2v_path}/av_w2v_${kmer}_${size}_${epochs}_${window}_${sg}.pt
python deep_train_test.py  --intrain ${train_path} --intest ${test_file} --outpath ${result_path} --losstype "balanced" --deeplearn ${deep_method}  --encode ${encode_mathod} --fold ${kfold} --w2vmodel ${w2v_model} --seqwin ${seqwin} --kmer ${kmer} --size ${size} --epochs ${epochs} --sg ${sg} --window ${window}
done


deep_method=CNN  #TX, CNN, bLSTM
encode_mathod=BE # W2V, BE
for kmer in 1 2 3 4 5 6 7 8 9 10 
do
w2v_model=${w2v_path}/av_w2v_${kmer}_${size}_${epochs}_${window}_${sg}.pt

python deep_train_test.py  --intrain ${train_path} --intest ${test_file} --outpath ${result_path} --losstype "balanced" --deeplearn ${deep_method}  --encode ${encode_mathod} --fold ${kfold} --w2vmodel ${w2v_model} --seqwin ${seqwin} --kmer ${kmer} --size ${size} --epochs ${epochs} --sg ${sg} --window ${window}
done

deep_method=bLSTM  #TX, CNN, bLSTM
encode_mathod=W2V # W2V, BE
for kmer in 1 2 3 4 5 6 7 8 9 10 
do
w2v_model=${w2v_path}/av_w2v_${kmer}_${size}_${epochs}_${window}_${sg}.pt
python deep_train_test.py  --intrain ${train_path} --intest ${test_file} --outpath ${result_path} --losstype "balanced" --deeplearn ${deep_method}  --encode ${encode_mathod} --fold ${kfold} --w2vmodel ${w2v_model} --seqwin ${seqwin} --kmer ${kmer} --size ${size} --epochs ${epochs} --sg ${sg} --window ${window}
done


deep_method=bLSTM  #TX, CNN, bLSTM
encode_mathod=BE # W2V, BE
for kmer in 1 2 3 4 5 6 7 8 9 10 
do
w2v_model=${w2v_path}/av_w2v_${kmer}_${size}_${epochs}_${window}_${sg}.pt

python deep_train_test.py  --intrain ${train_path} --intest ${test_file} --outpath ${result_path} --losstype "balanced" --deeplearn ${deep_method}  --encode ${encode_mathod} --fold ${kfold} --w2vmodel ${w2v_model} --seqwin ${seqwin} --kmer ${kmer} --size ${size} --epochs ${epochs} --sg ${sg} --window ${window}
done

