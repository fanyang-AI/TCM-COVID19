
import pandas as pd
import numpy as np
import os
import csv

# mapping
entity2id = {}
id2entity = {}
with open(r"C:\Users\Desktop\drugid5.txt", newline='', encoding='gbk') as csvfile:
    reader = csv.DictReader(csvfile, delimiter='\t', fieldnames=['id', 'entity'])
    for row_val in reader:
        #print(row_val)
        id = row_val['id']
        entity = row_val['entity']

        entity2id[entity] = int(id)
        id2entity[int(id)] = entity

#print("Number of entities: {}".format(len(entity2id)))



# 提取嵌入后的向量
entity_emb = []

with open(r"C:\Users\Desktop\01matrix5.txt", newline='', encoding='gbk') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')
    for row_val in reader:
        #print(len(row_val))
        entity_emb_i = []
        for i in row_val:

            #print(i)
            i = eval(i)

            entity_emb_i.append(i)
            #print(entity_emb_i)
        entity_emb.append(entity_emb_i)
#print(entity_emb[0])



# Calculate entity cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(entity_emb)
#print(similarity[76:][:76])
#print(len(similarity[76][:76]))
matrix = []
for i in similarity[76:]:
    magnitude = []
    for j in i[:76]:
        magnitude.append(j)
    matrix.append(magnitude)
#print(len(matrix))
sortidx = []
for sim in matrix:
    #sim = sorted(sim,reverse=True)
    #print(sim)
    idx = np.argsort(sim)
    #print(idx)
    sortidx.append(idx)
#print(len(sortidx[0]))

simdrug = []
for idxlist in sortidx:
    simdrug_i = []
    for i in idxlist:
        drug = id2entity[i]
        simdrug_i.append(drug)
    simdrug.append(simdrug_i)
for i in simdrug:
    print(i[-10:])
#print(simdrug[:][:5])




