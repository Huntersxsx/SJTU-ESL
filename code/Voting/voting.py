import pandas as pd
import os
import numpy as np

data_path = './files/'

AllL = []
VoteL = []

data_files = [filename for filename in sorted(os.listdir(data_path)) \
                           if os.path.isfile(os.path.join(data_path,filename))]

def ReadCSV(path):
    L = []
    fields = ['id', 'categories']
    df = pd.read_csv(path, usecols=fields, sep='\t')
    for index, item in enumerate(df.categories):
        L.append(item)
    return L


for fname in data_files:
    tempL = ReadCSV(os.path.join(data_path,fname))
    AllL.append(tempL)

AllLT = list(zip(*AllL))

for i in range(len(AllL[0])):
    L = list(AllLT[i][:])
    right = max(L, key=L.count)
    VoteL.append(right)

idx = [x for x in range(len(VoteL))]
data = zip(idx, VoteL)
save_file = pd.DataFrame(data=data)
save_file.to_csv('VoteResult.csv', encoding="utf-8", header=False, index=False, sep='\t')




