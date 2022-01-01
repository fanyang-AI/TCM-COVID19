# -*- coding: utf-8 -*-
"""
@author: weipan
"""
import pandas as pd
import numpy as np
medi_feature = pd.read_csv("D:\\py_data\\chinese_medicine\\feature.csv",index_col=0,encoding="utf-8")
medi_name = pd.read_csv("D:\py_data\chinese_medicine\guide_sdne.csv",encoding='utf-8')
guide_medi = list(medi_name.iloc[:,0].dropna())
guide_medifeature = medi_feature.loc[guide_medi,:]


sdne_medi = list(medi_name.iloc[:,1].dropna())
sdne_medifeature = medi_feature.loc[sdne_medi,:]

from sklearn.metrics.pairwise import cosine_similarity

result = cosine_similarity(sdne_medifeature,guide_medifeature)

final = list()

for i in range(len(result)):
    ssample = result[i,:]
    index = np.argsort(-ssample)[0:11]
    i_result = np.array(guide_medi)[index]
    for j in range(len(i_result)):
        final.append(i_result[j])
final = np.array(final).reshape(-1,11)
final = pd.DataFrame(final)
final.index = sdne_medi
final.to_csv("D:\py_data\chinese_medicine\result_drug_effect_similarity.csv",encoding="gbk")













