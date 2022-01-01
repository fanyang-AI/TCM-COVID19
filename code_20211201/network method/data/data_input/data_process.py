# -*- coding: utf-8 -*-
"""
data process matrix
@author:wei pan
"""

#create coefficient matrix
import numpy as np
import pandas as pd 

medicine_name_order = pd.read_table("../input/yaomingzi/entity2id.txt",header=None,encoding='gbk',index_col=0)
medicine_name_order = list(medicine_name_order.index)[0:480]

#row:medicine  column:compond
medi2comp = pd.read_csv("../input/womendeai/zhongyao_hhw_matrix_zuizhong_all.csv",index_col=0)
medi2comp = medi2comp.fillna(value=0)
medi2comp = medi2comp.loc[medicine_name_order,:]
#print(medi2comp.shape) (480,12735)
#row:compond column:protein
comp2prot = pd.read_csv("../input/womendeai/myda.csv",index_col=0)
comp2prot = comp2prot.fillna(value=0)
comp_names = list(medi2comp.columns)
comp2prot = comp2prot.loc[comp_names,:]
#print(comp2prot.shape) (12735,24)
def normalized(x):
    if np.max(x)==0:
        x == 0
    else:
        x -= np.min(x) #为了稳定地计算softmax概率， 一般会减掉最大的那个元素
        x = x / (np.max(x)-np.min(x))
    return x

#compond absorb rate
comp_caco2 = pd.read_csv("../input/hhwcaco2/zhongyao_hhw_CACO2.csv",encoding='gbk')
comp_caco2 = comp_caco2.loc[:,["name","MOL_ID","CACO2"]]
comp_caco2 = comp_caco2.fillna(value=0)
comp_caco2.head
comp_caco2.loc[:,"CACO2"] = comp_caco2.loc[:,"CACO2"] - min(comp_caco2.loc[:,"CACO2"])
comp_caco2.loc[:,"CACO2"] = comp_caco2.loc[:,"CACO2"]/(max(comp_caco2.loc[:,"CACO2"])-min(comp_caco2.loc[:,"CACO2"]))
#print(comp_caco2.shape)
#缺失值归一化后为0.780952
comp_caco2 = comp_caco2.drop_duplicates(["name","MOL_ID"],keep='first')
#print(comp_caco2.shape)
select_medi = medi2comp.iloc[0]
cand_comp = comp_caco2[comp_caco2["name"]==select_medi.name]
cand_comp = cand_comp.sort_values(by='MOL_ID')
#print(cand_comp)
comp_absorb_rate = np.array(cand_comp.loc[:,"CACO2"]).reshape(-1,1)
#comp_absorb_rate
select_comp = select_medi[select_medi==1.0].index.tolist()
medi_combine_prot = comp2prot.loc[select_comp,]
medi_combine_prot = medi_combine_prot.sort_values(by='name')
medi_combine_prot = np.array(medi_combine_prot)
medi_combine_prot = comp_absorb_rate * medi_combine_prot
medi_combine_prot = -np.sum(medi_combine_prot,axis=0).reshape((1,-1))
medi_combine_prot = normalized(medi_combine_prot)
#medi_combine_prot

for i in range(1,medi2comp.shape[0]):
    #print(i)
    select_medi = medi2comp.iloc[i]
    cand_comp = comp_caco2[comp_caco2["name"]==select_medi.name]
    cand_comp = cand_comp.sort_values(by='MOL_ID')
    select_comp = select_medi[select_medi==1.0].index.tolist()
    cand_comp = comp_caco2[comp_caco2["name"]==select_medi.name]
    if len(select_comp) != 0:
        candicate = comp2prot.loc[select_comp,]
        candicate = candicate.sort_values(by='name')
        if candicate.shape[0] != cand_comp.shape[0]:
            cand_comp = cand_comp[cand_comp["MOL_ID"].isin(list(candicate.index))]
            comp_absorb_rate = np.array(cand_comp.loc[:,"CACO2"]).reshape(-1,1)
            candicate = np.array(candicate)
            candicate = comp_absorb_rate * candicate
            candicate = -np.sum(candicate,axis=0).reshape((1,-1))
            candicate = normalized(candicate)
            medi_combine_prot = np.concatenate((medi_combine_prot,candicate),axis=0)    
        else:
            comp_absorb_rate = np.array(cand_comp.loc[:,"CACO2"]).reshape(-1,1)
            candicate = np.array(candicate)
            candicate = comp_absorb_rate * candicate
            candicate = -np.sum(candicate,axis=0).reshape((1,-1))
            candicate = normalized(candicate)
            medi_combine_prot = np.concatenate((medi_combine_prot,candicate),axis=0)
    else:
        candicate = np.zeros(24)
        medi_combine_prot = np.concatenate((medi_combine_prot,candicate),axis=0)
print(medi_combine_prot.shape)


medi_combine_prot[:,0] = 20*medi_combine_prot[:,0]
medi_combine_prot[:,1:3] = 10*medi_combine_prot[:,1:3]
medi_combine_prot[:,3] = 15*medi_combine_prot[:,3]
print(medi_combine_prot.shape)
medi_combine_prot_T = medi_combine_prot.T

print(medi_combine_prot.shape)
#medi_combine_prot[:,4:] = 1*medi_combine_prot[:,4:]

#set:medicine-A;compond-B;protein-C
A2B = np.array(medi2comp)
A2B_rowsum = np.sum(A2B,axis=1)
A2B_prob = A2B/A2B_rowsum[:,None]
A2B_prob = np.nan_to_num(A2B_prob)

B2A = np.transpose(A2B)
B2A_rowsum = np.sum(B2A,axis=1)
B2A_prob = B2A/B2A_rowsum[:,None]
B2A_prob = np.nan_to_num(B2A_prob)

B2C = np.array(comp2prot)
B2C[B2C>-7] = 0
B2C[B2C<=-7] = 1
B2C_rowsum = np.sum(B2C,axis=1)
B2C_prob = B2C/B2C_rowsum[:,None]
B2C_prob = np.nan_to_num(B2C_prob)

C2B = np.transpose(B2C)
C2B_rowsum = np.sum(C2B,axis=1)
C2B_prob = C2B/C2B_rowsum[:,None]
C2B_prob = np.nan_to_num(C2B_prob)

#result matrix sets M
M11 = np.dot(A2B_prob,B2A_prob)
M22 = np.dot(C2B_prob,B2C_prob)
M12 = np.dot(A2B_prob,B2C_prob)
M21 = np.dot(C2B_prob,B2A_prob)
M1 = np.concatenate((M11,M12),axis=1)
M2 = np.concatenate((M21,M22),axis=1)
M = np.concatenate((M1,M2),axis=0)
#coefficient_matrix create graph

from_list = []
to_list = []
for i in range(M.shape[0]):
    for j in range(M.shape[0]):
        if i != j :
            if M[i,j] > 0 :
                from_list.append(i+1)
                to_list.append(j+1)
adj_biaohhh = np.array([from_list,to_list]).T
adj_biaohhh = adj_biaohhh.astype('int32')
np.savetxt("./biaobiao.txt",adj_biaohhh,delimiter=" ")

#coefficient matrix
aaa = np.concatenate((M11,medi_combine_prot),axis=1)
bbb = np.concatenate((medi_combine_prot_T,M22),axis=1)
coefficient_matrix = np.concatenate((aaa,bbb),axis=0)
#coefficient_matrix = pd.DataFrame(coefficient_matrix)
#coefficient_matrix.to_csv("./coefficient_matrix.csv")





