import pandas as pd
import numpy as np
import re, math
from collections import Counter

def AAC(fastas, **kw):
    AA = kw['order'] if kw['order'] != None else 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []
    '''
    #header = ['#', 'label']
    header = []  
    for i in AA:
        header.append(i)   
    encodings.append(header)
    '''
    for i in fastas:
        sequence, label = i[0], i[1]
        sequence = re.sub('X', '', sequence) # addition
        count = Counter(sequence)
        for key in count:
            count[key] = count[key]/len(sequence)
        #code = [sequence, label]
        code = []
        for aa in AA:
            code.append(count[aa])
        encodings.append(code)
    return encodings

def get_min_sequence_length(fastas):
    minLen = 10000
    for i in fastas:
        if minLen > len(i[0]):
            minLen = len(i[0])
    return minLen

def generateGroupPairs(groupKey):
    gPair = {}
    for key1 in groupKey:
        for key2 in groupKey:
            gPair[key1 + '.' + key2] = 0
    return gPair

def CTDT(fastas, **kw):
    group1 = {
        'hydrophobicity_PRAM900101': 'RKEDQN',
        'hydrophobicity_ARGP820101': 'QSTNGDE',
        'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
        'hydrophobicity_PONP930101': 'KPDESNQT',
        'hydrophobicity_CASG920101': 'KDEQPSRNTG',
        'hydrophobicity_ENGD860101': 'RDKENQHYP',
        'hydrophobicity_FASG890101': 'KERSQD',
        'normwaalsvolume': 'GASTPDC',
        'polarity':        'LIFWCMVY',
        'polarizability':  'GASDT',
        'charge':          'KR',
        'secondarystruct': 'EALMQKRH',
        'solventaccess':   'ALFCGIVW'
    }
    group2 = {
        'hydrophobicity_PRAM900101': 'GASTPHY',
        'hydrophobicity_ARGP820101': 'RAHCKMV',
        'hydrophobicity_ZIMJ680101': 'HMCKV',
        'hydrophobicity_PONP930101': 'GRHA',
        'hydrophobicity_CASG920101': 'AHYMLV',
        'hydrophobicity_ENGD860101': 'SGTAW',
        'hydrophobicity_FASG890101': 'NTPG',
        'normwaalsvolume': 'NVEQIL',
        'polarity':        'PATGS',
        'polarizability':  'CPNVEQIL',
        'charge':          'ANCQGHILMFPSTWYV',
        'secondarystruct': 'VIYCWFT',
        'solventaccess':   'RKQEND'
    }
    group3 = {
        'hydrophobicity_PRAM900101': 'CLVIMFW',
        'hydrophobicity_ARGP820101': 'LYPFIW',
        'hydrophobicity_ZIMJ680101': 'LPFYI',
        'hydrophobicity_PONP930101': 'YMFWLCVI',
        'hydrophobicity_CASG920101': 'FIWC',
        'hydrophobicity_ENGD860101': 'CVLIMF',
        'hydrophobicity_FASG890101': 'AYHWVMFLIC',
        'normwaalsvolume': 'MHKFRYW',
        'polarity':        'HQRKNED',
        'polarizability':  'KMHFRYW',
        'charge':          'DE',
        'secondarystruct': 'GNPSD',
        'solventaccess':   'MSPTHY'
    }

    groups = [group1, group2, group3]
    property = (
    'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101', 'hydrophobicity_PONP930101',
    'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
    'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

    encodings = []
    '''
    #header = ['#', 'label']
    header = []
    for p in property:
        for tr in ('Tr1221', 'Tr1331', 'Tr2332'):
            header.append(p + '.' + tr)
    encodings.append(header)
    '''
    for i in fastas:
        sequence, label = i[0],  i[1] #
        sequence = re.sub('X', '', sequence) # addition
        #code = [name, label]
        code = []
        aaPair = [sequence[j:j + 2] for j in range(len(sequence) - 1)]
        for p in property:
            c1221, c1331, c2332 = 0, 0, 0
            for pair in aaPair:
                if (pair[0] in group1[p] and pair[1] in group2[p]) or (pair[0] in group2[p] and pair[1] in group1[p]):
                    c1221 = c1221 + 1
                    continue
                if (pair[0] in group1[p] and pair[1] in group3[p]) or (pair[0] in group3[p] and pair[1] in group1[p]):
                    c1331 = c1331 + 1
                    continue
                if (pair[0] in group2[p] and pair[1] in group3[p]) or (pair[0] in group3[p] and pair[1] in group2[p]):
                    c2332 = c2332 + 1
            code = code + [c1221/len(aaPair), c1331/len(aaPair), c2332/len(aaPair)]
        encodings.append(code)
    return encodings


def CKSAAGP(fastas, gap=5, **kw):
    if gap < 0:
        print('Error: the gap should be equal or greater than zero' + '\n\n')
        return 0

    if get_min_sequence_length(fastas) < gap + 2:
        print('Error: all the sequence length should be larger than the (gap value) + 2 = ' + str(gap + 2) + '\n\n')
        return 0

    group = {
        'alphaticr': 'GAVLMI',
        'aromatic': 'FYW',
        'postivecharger': 'KRH',
        'negativecharger': 'DE',
        'uncharger': 'STCPNQ'
    }
    AA = kw['order'] if kw['order'] != None else 'ACDEFGHIKLMNPQRSTVWY'
    #AA = 'ARNDCQEGHILKMFPSTWYV'

    groupKey = group.keys()

    index = {}
    for key in groupKey:
        for aa in group[key]:
            index[aa] = key

    gPairIndex = []
    for key1 in groupKey:
        for key2 in groupKey:
            gPairIndex.append(key1 + '.' + key2)

    encodings = []
    '''
    #header = ['#', 'label']
    header = []
    for g in range(gap + 1):
        for p in gPairIndex:
            header.append(p + '.gap' + str(g))
    encodings.append(header)
    '''
    for i in fastas:
        sequence, label = i[0],i[1]
        sequence = re.sub('X', '', sequence) # addition
        #code = [name, label]
        code = []
        for g in range(gap + 1):
            gPair = generateGroupPairs(groupKey)
            sum = 0
            for p1 in range(len(sequence)):
                p2 = p1 + g + 1
                if p2 < len(sequence) and sequence[p1] in AA and sequence[p2] in AA:
                    gPair[index[sequence[p1]] + '.' + index[sequence[p2]]] = gPair[index[sequence[p1]] + '.' + index[
                        sequence[p2]]] + 1
                    sum = sum + 1

            if sum == 0:
                for gp in gPairIndex:
                    code.append(0)
            else:
                for gp in gPairIndex:
                    code.append(gPair[gp] / sum)
        encodings.append(code)
    return encodings
    

def get_min_sequence_length(fastas):
    minLen = 10000
    for i in fastas:
        if minLen > len(i[0]):
            minLen = len(i[0])
    return minLen

def CKSAAP(fastas, gap=0, **kw): #revision gap=5 default
    if gap < 0:
        print('Error: the gap should be equal or greater than zero' + '\n\n')
        return 0

    if get_min_sequence_length(fastas) < gap + 2:
        print('Error: all the sequence length should be larger than the (gap value) + 2 = ' + str(gap + 2) + '\n\n')
        return 0

    AA = kw['order'] if kw['order'] != None else 'ACDEFGHIKLMNPQRSTVWY'
    #AA='ARNDCQEGHILKMFPSTWYVX'
    
    encodings = []
    aaPairs = []
    for aa1 in AA:
        for aa2 in AA:
            aaPairs.append(aa1 + aa2)
    '''
    #header = ['#', 'label']
    header =[]
    for g in range(gap + 1):
        for aa in aaPairs:
            header.append(aa + '.gap' + str(g))
    encodings.append(header)
    '''
    for i in fastas:
        sequence, label = i[0], i[1]
        sequence = re.sub('X', '', sequence) # addition
        #code = [name, label]
        code = []
        for g in range(gap + 1):
            myDict = {}
            for pair in aaPairs:
                myDict[pair] = 0
            sum = 0
            for index1 in range(len(sequence)):
                index2 = index1 + g + 1
                if index1 < len(sequence) and index2 < len(sequence) and sequence[index1] in AA and sequence[
                    index2] in AA:
                    myDict[sequence[index1] + sequence[index2]] = myDict[sequence[index1] + sequence[index2]] + 1
                    sum = sum + 1
            for pair in aaPairs:
                code.append(myDict[pair] / sum)
        encodings.append(code)
    return encodings


def get_min_sequence_length_1(fastas):
    minLen = 10000
    for i in fastas:
        if minLen > len(re.sub('X', '', i[0])):
            minLen = len(re.sub('X', '', i[0]))
    return minLen

def Rvalue(aa1, aa2, AADict, Matrix):
    return sum([(Matrix[i][AADict[aa1]] - Matrix[i][AADict[aa2]]) ** 2 for i in range(len(Matrix))]) / len(Matrix)


def PAAC(fastas, lambdaValue=5, w=0.05, **kw):# adjusted
    #print(fastas)
    if get_min_sequence_length_1(fastas) < lambdaValue + 1:
        print(
            'Error: all the sequence length should be larger than the lambdaValue+1: ' + str(lambdaValue + 1) + '\n\n')
        return 0

    dataFile = './PAAC.txt' # adjusted
    with open(dataFile) as f:
        records = f.readlines()
    AA = ''.join(records[0].rstrip().split()[1:])
    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i
    AAProperty = []
    AAPropertyNames = []
    for i in range(1, len(records)):
        array = records[i].rstrip().split() if records[i].rstrip() != '' else None
        AAProperty.append([float(j) for j in array[1:]])
        AAPropertyNames.append(array[0])

    AAProperty1 = []
    for i in AAProperty:
        meanI = sum(i) / 20
        fenmu = math.sqrt(sum([(j - meanI) ** 2 for j in i]) / 20)
        AAProperty1.append([(j - meanI) / fenmu for j in i])
    
    encodings=[]
    '''
    #header = ['#', 'label']
    header = []
    for aa in AA:
        header.append('Xc1.' + aa)
    for n in range(1, lambdaValue + 1):
        header.append('Xc2.lambda' + str(n))
    encodings.append(header)
    '''
    
    for i in fastas:
        sequence, label = i[0], i[1]
        sequence = re.sub('X', '', sequence) # addition
        #code = [name, label]
        code = []
        theta = []
        for n in range(1, lambdaValue + 1):
            theta.append(
                sum([Rvalue(sequence[j], sequence[j + n], AADict, AAProperty1) for j in range(len(sequence) - n)]) / (
                    len(sequence) - n))
        myDict = {}
        for aa in AA:
            myDict[aa] = sequence.count(aa)
        code = code + [myDict[aa] / (1 + w * sum(theta)) for aa in AA]
        code = code + [(w * j) / (1 + w * sum(theta)) for j in theta]
        encodings.append(code)

    return encodings

def check_fasta_with_equal_length(fastas):
    status = True
    lenList = set()
    for i in fastas:
        lenList.add(len(i[0]))
    if len(lenList) == 1:
        return True
    else:
        return False
        
def AAINDEX(fastas, props=None, **kw):
    if check_fasta_with_equal_length(fastas) == False:
        print('Error: for "AAINDEX" encoding, the input fasta sequences should be with equal length. \n\n')
        return 0
    AA = kw['order'] if kw['order'] != None else 'ACDEFGHIKLMNPQRSTVWY'
    #AA = 'ARNDCQEGHILKMFPSTWYV'

    #fileAAindex = re.sub('descproteins$', '', os.path.split(os.path.realpath(__file__))[
    #    0]) + r'\data\AAindex.txt' if platform.system() == 'Windows' else re.sub('descproteins$', '', os.path.split(
    #    os.path.realpath(__file__))[0]) + r'/data/AAindex.txt'
    #fileAAindex = data_path + '/descproteins/data/AAindex.txt'
    fileAAindex = './AAindex.txt'
    with open(fileAAindex) as f:
        records = f.readlines()[1:]

    AAindex = []
    AAindexName = []
    for i in records:
        AAindex.append(i.rstrip().split()[1:] if i.rstrip() != '' else None)
        AAindexName.append(i.rstrip().split()[0] if i.rstrip() != '' else None)

    index = {}
    for i in range(len(AA)):
        index[AA[i]] = i

    #  use the user inputed properties
    if props:
        tmpIndexNames = []
        tmpIndex = []
        for p in props:
            if AAindexName.index(p) != -1:
                tmpIndexNames.append(p)
                tmpIndex.append(AAindex[AAindexName.index(p)])
        if len(tmpIndexNames) != 0:
            AAindexName = tmpIndexNames
            AAindex = tmpIndex
    
    encodings = []
    '''
    #header = ['#', 'label']
    header = []
    for pos in range(1, len(fastas[0][0]) + 1):
        for idName in AAindexName:
            header.append('SeqPos.' + str(pos) + '.' + idName)
    encodings.append(header)
    '''
    for i in fastas:
        sequence, label = i[0], i[1]
        #code = [name, label]
        code = []
        for aa in sequence:
            if aa == 'X':
                for j in AAindex:
                    code.append(0)
                continue
            for j in AAindex:
                code.append(j[index[aa]])
        encodings.append(code)
    return encodings

def BLOSUM62(fastas, **kw):

    blosum62 = {
        'A': [4,  -1, -2, -2, 0,  -1, -1, 0, -2,  -1, -1, -1, -1, -2, -1, 1,  0,  -3, -2, 0],  # A
        'R': [-1, 5,  0,  -2, -3, 1,  0,  -2, 0,  -3, -2, 2,  -1, -3, -2, -1, -1, -3, -2, -3], # R
        'N': [-2, 0,  6,  1,  -3, 0,  0,  0,  1,  -3, -3, 0,  -2, -3, -2, 1,  0,  -4, -2, -3], # N
        'D': [-2, -2, 1,  6,  -3, 0,  2,  -1, -1, -3, -4, -1, -3, -3, -1, 0,  -1, -4, -3, -3], # D
        'C': [0,  -3, -3, -3, 9,  -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1], # C
        'Q': [-1, 1,  0,  0,  -3, 5,  2,  -2, 0,  -3, -2, 1,  0,  -3, -1, 0,  -1, -2, -1, -2], # Q
        'E': [-1, 0,  0,  2,  -4, 2,  5,  -2, 0,  -3, -3, 1,  -2, -3, -1, 0,  -1, -3, -2, -2], # E
        'G': [0,  -2, 0,  -1, -3, -2, -2, 6,  -2, -4, -4, -2, -3, -3, -2, 0,  -2, -2, -3, -3], # G
        'H': [-2, 0,  1,  -1, -3, 0,  0,  -2, 8,  -3, -3, -1, -2, -1, -2, -1, -2, -2, 2,  -3], # H
        'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4,  2,  -3, 1,  0,  -3, -2, -1, -3, -1, 3],  # I
        'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2,  4,  -2, 2,  0,  -3, -2, -1, -2, -1, 1],  # L
        'K': [-1, 2,  0,  -1, -3, 1,  1,  -2, -1, -3, -2, 5,  -1, -3, -1, 0,  -1, -3, -2, -2], # K
        'M': [-1, -1, -2, -3, -1, 0,  -2, -3, -2, 1,  2,  -1, 5,  0,  -2, -1, -1, -1, -1, 1],  # M
        'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0,  0,  -3, 0,  6,  -4, -2, -2, 1,  3,  -1], # F
        'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7,  -1, -1, -4, -3, -2], # P
        'S': [1,  -1, 1,  0,  -1, 0,  0,  0,  -1, -2, -2, 0,  -1, -2, -1, 4,  1,  -3, -2, -2], # S
        'T': [0,  -1, 0,  -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1,  5,  -2, -2, 0],  # T
        'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1,  -4, -3, -2, 11, 2,  -3], # W
        'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2,  -1, -1, -2, -1, 3,  -3, -2, -2, 2,  7,  -1], # Y
        'V': [0,  -3, -3, -3, -1, -2, -2, -3, -3, 3,  1,  -2, 1,  -1, -2, -2, 0,  -3, -1, 4],  # V
        'X': [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # -
    }
    encodings = []
    '''
    #header = ['#', 'label']
    header = []
    for i in range(1, len(fastas[0][0]) * 20 + 1):
        header.append('blosum62.F'+str(i))
    encodings.append(header)
    '''
    for i in fastas:
        sequence, label = i[0], i[1]
        #code = [name, label]
        code = []
        for aa in sequence:
            code = code + blosum62[aa]
        encodings.append(code)
    return encodings
