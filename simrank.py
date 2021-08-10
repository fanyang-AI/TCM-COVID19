
import pandas as pd
import numpy as np
import os
import csv

# mapping
entity2id = {}
id2entity = {}
with open(r"C:\Users\Desktop\nodeid.txt", newline='', encoding='gbk') as csvfile:
    reader = csv.DictReader(csvfile, delimiter='\t', fieldnames=['entity', 'id'])
    for row_val in reader:
        #print(row_val)
        id = row_val['id']
        entity = row_val['entity']

        entity2id[entity] = int(id)
        id2entity[int(id)] = entity

#print("Number of entities: {}".format(len(entity2id)))



# 提取嵌入后的向量
entity_emb = []

with open(r"C:\Users\Desktop\vec2.txt", newline='', encoding='gbk') as csvfile:
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
#print(len(entity_emb[0]))

#General Entity Embedding Clustering

from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn.utils import check_random_state
from sklearn.manifold import TSNE


X_embedded = TSNE(n_components=2, n_jobs=40).fit_transform(entity_emb).T
#print(X_embedded)

dataset_id = {}
with open( r"C:\Users\Desktop\nodeid.txt", newline='', encoding='gbk') as csvfile:
    reader = csv.DictReader(csvfile, delimiter='\t', fieldnames=['entity','id'])
    for row_val in reader:
        id = int(row_val['id'])
        #print(id)

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
#print(dataset_id["protein"])
"""
p = cm.rainbow(int(255/2 * 1))
for key, val in dataset_id.items():
    val = np.asarray(val, dtype=np.long)

    plt.plot(X_embedded[0][val], X_embedded[1][val], '.', label=key)
    plt.legend(bbox_to_anchor=(0, 0, 1.45, 1.0))

plt.show()
"""

# Calculate entity cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(entity_emb)
#print(similarity[480:][:479])
protein_id = {}
for key in dataset_id["protein"]:
    protein_key = key
    if protein_id.get(protein_key, None) is None:
        protein_id[protein_key] = []
    protein_id[protein_key].append(id2entity[int(key)])
print(protein_id)


cluster_id = eval(input("请输入需要聚类的蛋白的id(可多个，逗号分开)："))
#print(cluster_id)
rankid = []
simrank = []
for i in range(479):
    simrank.append(similarity[i][cluster_id])
#print(simrank)
rankid = sorted(range(len(simrank)),key=lambda k:simrank[k])
print(rankid)

simranknode = []
for i in rankid:
    simranknode.append(id2entity[i])
print(simranknode)
"""
cossim_total = []
for i in range(479):
    for j in cluster_id:
        cossim = []
        cossim.append(similarity[i][j])
        print(similarity[i][j])
    cossim_total.append(cossim)


cossim_ave = []
for sim in cossim_total:
    sim_ave = np.mean(sim)
    cossim_ave.append(sim_ave)
cossim_ave_sort = np.sort(cossim_ave)
print(cossim_ave_sort)
cossim_ave_sort_indx = np.argsort(cossim_ave)
print(cossim_ave_sort_indx[:100])
for i in cossim_ave_sort_indx[:100]:
    print(id2entity[i])
"""

