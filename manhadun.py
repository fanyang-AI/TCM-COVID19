import pandas as pd
import numpy as np
import os
import csv


def Manhadun(m, n):
    m = np.array(m)
    n = np.array(n)
    return np.sum(np.abs(m-n))


# mapping
entity2id = {}
id2entity = {}
with open(r"C:\Users\Desktop\nodeid.txt", newline='', encoding='gbk') as csvfile:
    reader = csv.DictReader(csvfile, delimiter='\t', fieldnames=['entity', 'id'])
    for row_val in reader:
        # print(row_val)
        id = row_val['id']
        entity = row_val['entity']

        entity2id[entity] = int(id)
        id2entity[int(id)] = entity

# print("Number of entities: {}".format(len(entity2id)))


# 提取嵌入后的向量
entity_emb = []
with open(r"C:\Users\Desktop\vec128.txt", newline='', encoding='gbk') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')
    for row_val in reader:
        # print(len(row_val))
        entity_emb_i = []
        for i in row_val:
            #print(i)
            i = eval(i)

            entity_emb_i.append(i)
            #print(entity_emb_i)
        entity_emb.append(entity_emb_i)
#print(entity_emb)

# General Entity Embedding Clustering

from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn.utils import check_random_state
from sklearn.manifold import TSNE

X_embedded = TSNE(n_components=2, n_jobs=40).fit_transform(entity_emb).T
# print(X_embedded)

dataset_id = {}
with open(r"C:\Users\Desktop\nodeid.txt", newline='', encoding='gbk') as csvfile:
    reader = csv.DictReader(csvfile, delimiter='\t', fieldnames=['entity', 'id'])
    for row_val in reader:
        id = int(row_val['id'])
        # print(id)

        if id <= 479:
            entity_key = "drug"
            if dataset_id.get(entity_key, None) is None:
                dataset_id[entity_key] = []
            dataset_id[entity_key].append(row_val['id'])
        else:
            entity_key = "protein"
            if dataset_id.get(entity_key, None) is None:
                dataset_id[entity_key] = []
            dataset_id[entity_key].append(row_val['id'])
# print(dataset_id["protein"])
"""
p = cm.rainbow(int(255/2 * 1))
for key, val in dataset_id.items():
    val = np.asarray(val, dtype=np.long)

    plt.plot(X_embedded[0][val], X_embedded[1][val], '.', label=key)
    plt.legend(bbox_to_anchor=(0, 0, 1.45, 1.0))

plt.show()
"""

distances_total = []
for i in entity_emb:
    #print(i)
    distances_i = []
    for j in entity_emb:
        distances = Manhadun(i,j)
        #print(distances)
        distances_i.append(distances)
        #print(distances_i)
    distances_total.append(distances_i)
#print(distances_total[0])


#print(distances_total[479])
#print(distances_total[481])
protein_id = {}
for key in dataset_id["protein"]:
    protein_key = key
    if protein_id.get(protein_key, None) is None:
        protein_id[protein_key] = []
    protein_id[protein_key].append(id2entity[int(key)])
print(protein_id)

cluster_id = eval(input("请输入需要分析的蛋白的id(头和尾，逗号分开)："))
print(cluster_id)
rankid = []
disrank = []
for i in range(cluster_id[0], cluster_id[1] + 1):
    # print(i)
    # print(distances_total[i])
    disrank.append(distances_total[i][:480])
    # print(disrank)
# for i in range(2):
# print(disrank[i])
topdis = []
for dist in disrank:
    rankid = sorted(range(len(dist)), key=lambda k: dist[k])
    topdis.append(rankid)
print(len(topdis))

topdisnode = []
for top in topdis:
    topdisnode_i = []
    for i in top:
        topdisnode_i.append(id2entity[i])
    topdisnode.append(topdisnode_i)
for i in topdisnode:
    print(i[:10])

