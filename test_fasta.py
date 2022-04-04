#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-i1', '--infile2', help='file')
parser.add_argument('-o1', '--outfile1', help='file')
parser.add_argument('-o2', '--outfile2', help='file')

test_txt = parser.parse_args().infile2
test_fasta = parser.parse_args().outfile1
test_csv = parser.parse_args().outfile2

test = pd.read_csv(test_txt , header=None)
test = test.rename(columns={0:'seq',1:'label'})
print(test)

with open(test_fasta, 'w') as fout:
   for i in range(test.shape[0]):
      if test.iloc[i,1] == 1:
         fout.write('>pep_%s|1|label\n'%i)
         fout.write(test.iloc[i,0])
         fout.write('\n')
      else:
         fout.write('>pep_%s|0|label\n'%i)
         fout.write(test.iloc[i,0])
         fout.write('\n')

test.to_csv(test_csv, index=None)
    
