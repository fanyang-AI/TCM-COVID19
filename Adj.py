
import networkx as nx
import numpy as np
from torch.utils import data
from torch.utils.data import DataLoader
import torch,csv
import pandas as pd
filename = r'C:\Users\Desktop\SDNE-based-on-Pytorch-master\matrix2.csv'


data = csv.reader(open(filename,encoding='utf-8-sig'))
dataall = []
for i in data:
   # print(i)
    dataall.append(i)


G = nx.Graph()
Adj = np.zeros([504, 504], dtype=np.int32)
for i in range(len(dataall)):
    for j in range(len(dataall)):

        if dataall[i][j]!= '0':
             print(float(dataall[i][j]))
             G.add_edge(i, j)
             Adj[i, j] = float(dataall[i][j])

             #print(Adj[i, j])
             #Adj[i, j] = format( Adj[i, j],'.5f')

Adj = torch.FloatTensor(Adj)
#np.set_printoptions(formatter={'float': '{: 0.5f}'.format})
print(Adj)
print(Adj.shape)