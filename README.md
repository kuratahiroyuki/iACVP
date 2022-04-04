
**Environment**

 python3.8.8  
 pytorch 1.9.0  
 gensim 4.0.1  
 scikit-learn1.0.2  
 
 
**Execution**

1. Setting current path  

 Users need to change the program_path according to their system in the following programs.  
 In our case, we set program_path=/home/kurata/myproject/py3/ACVP_predictor.  
 dataconst.sh    
 w2vconst.sh     
 main.sh   
 shbmain.sh   
 analysis_net.py   
 analysis_ml.py   
 
2. Construction of training and test datasets  
./  　
>sh dataconst.sh  
 train_division.py  
 test_fasta.py  　　
 
3. Word2vec model construction  
./program/w2v  
>sh w2vconst.sh  
 word2vec_acvp.py  
 
NOTE  
 Use epochs=20  for prediction of AVPs  
 Use epochs=4   for prediction of ACVPs  
 
4. Training and testing of W2V-based and BE-based classifiers  
4-1 Simulation  
./program/network  
>sh main.sh  
 deep_train_test.py  
 machine_train_test.py  

4-2 Analysis (calculation of measures)  
./program  
>python analysis_net.py  　

5. Training and testing of RF and SVM classifiers with conventional encodings  
5-1 Simulation  
./program/ml  
>sh submain.sh  
 conv_train_test.py  
5-2 Analysis (calculation of measures)  
./program  
>python analysis_ml.py   

