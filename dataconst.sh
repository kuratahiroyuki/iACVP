#!/bin/sh

data_path=./data/dataset
infile1=${data_path}/AVP_train.txt    #AVP_train_ACVP_E.txt
infile2=${data_path}/ACVP_M_test.txt  #ACVP_E_test.txt, 

test_fasta=${data_path}/independent_test/independent_test.fa
test_csv=${data_path}/independent_test/independent_test.csv

kfold=5

python ./train_division.py --infile1 ${infile1} --datapath ${data_path} --kfold ${kfold} 

python ./test_fasta.py --infile2 ${infile2} --outfile1 ${test_fasta} --outfile2 ${test_csv} 



