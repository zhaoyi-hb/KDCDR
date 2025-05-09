import pandas as pd
import os
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import json
import re
import ast
from zhipuai import ZhipuAI

def pprint(str_,f):
    print(str_)
    print(str_,end='\n',file=f)
def filter_data(filePath):
    data = []
    ratings = pd.read_csv(filePath, delimiter=",", encoding="latin1")
    ratings.columns = ['userId', 'itemId', 'Rating','timesteamp']
    
    rate_size_dic_i=ratings.groupby('itemId').size()
    choosed_index_del_i=rate_size_dic_i.index[rate_size_dic_i<10]
    ratings=ratings[~ratings['itemId'].isin(list(choosed_index_del_i))]
    
    user_unique=list(ratings['userId'].unique())  
    movie_unique=list(ratings['itemId'].unique()) 

    u=len(user_unique)
    i=len(movie_unique)
    rating_num = len(ratings)
    return u,i,rating_num,user_unique,ratings
def get_min_group_size(ratings):
    rate_size_dic_u=ratings.groupby('userId').size()
    return min(rate_size_dic_u)
def reindex_data(ratings1,dic_u=None):
    data = []
    if dic_u is None:
        user_unique=list(ratings1['userId'].unique())  
        user_index=list(range(0,len(user_unique)))
        dic_u=dict(zip(user_unique,user_index))
    movie_unique1=list(ratings1['itemId'].unique()) 
    movie_index1=list(range(0,len(movie_unique1)))
    dic_m1=dict(zip(movie_unique1,movie_index1))
    for element in ratings1.values:
        data.append((dic_u[element[0]], dic_m1[element[1]], 1, element[3]))
    data = sorted(data,key=lambda x:x[0])
    return data,dic_u,dic_m1
def get_common_data(data1,data2,user_common):
    rating_new_1= data1[data1['userId'].isin(common_user)]
    rating_new_2 = data2[data2['userId'].isin(common_user)]
    return rating_new_1,rating_new_2
def get_unique_lenth(ratings):
    r_n = len(ratings)
    user_unique=list(ratings['userId'].unique())  
    movie_unique=list(ratings['itemId'].unique()) 
    u=len(user_unique)
    i=len(movie_unique)
    return u,i,r_n
def filter_user(ratings1,ratings2):
    rate_size_dic_u1=ratings1.groupby('userId').size()
    rate_size_dic_u2=ratings2.groupby('userId').size()
    choosed_index_del_u1=rate_size_dic_u1.index[rate_size_dic_u1<5]
    choosed_index_del_u2=rate_size_dic_u2.index[rate_size_dic_u2<5]
    ratings1=ratings1[~ratings1['userId'].isin(list(choosed_index_del_u1)+list(choosed_index_del_u2))]
    ratings2=ratings2[~ratings2['userId'].isin(list(choosed_index_del_u1)+list(choosed_index_del_u2))]
    return ratings1,ratings2
def write_to_txt(data,file):
    f = open(file,'w+')
    f.write('reviewerID,asin,overall,unixReviewTime\n')
    data = sorted(data, key=lambda x:x[3])
    for i in data:
        line = ','.join([str(x) for x in i])+'\n'
        f.write(line)
    f.close
def get_common_user(data1,data2):
    common_user = list(set(data1).intersection(set(data2)))
    return len(common_user),common_user

def get_json_data(item,item_path,flag='source'):
    
    item_tmp = {}
    item_title = {}
    
    with open(item_path, 'r', encoding='utf-8') as f:
        for line in f:
            for tmp in  line.split(','):
                if 'asin' in tmp:
                    idtm_id = tmp.split(':')[-1][2:-1]
                    item_tmp[idtm_id] = ''
                    item_title[idtm_id] = ''
                if 'description' in tmp:
                    item_tmp[idtm_id]  +=  tmp.split(':')[-1]
                if 'title' in tmp:
                    item_tmp[idtm_id] =  tmp.split(':')[-1] + item_tmp[idtm_id]
                    # item_title[idtm_id] += tmp.split(':')[-1]
            categories_str = re.search(r"'categories':\s*(\[\[.*?\]\])", line).group(1)
            item_title[idtm_id] += categories_str
    i = 0
    j = 0
    item_attr = [] 
    item_title_attr = []         
    for key in item:
        if key in item_tmp.keys() and item_tmp[key]!= '':
            item_attr.append(item_tmp[key])
        else:
            i += 1
            item_attr.append('None')

        if key in item_title.keys() and item_title[key]!= '':
            item_title_attr.append(item_title[key])
        else:
            j += 1
            item_title_attr.append('None')

    print('%s item attr id None is %d' % (flag, i))
    print('%s item title id None is %d' % (flag, j))
    
    return item_attr,item_title_attr


def get_embedding(item,file_path,item_name_path,flag='source'):
    print('========%s bert embedding==========' % flag)
    
    # 获取item的属性
    item_attr,item_title_attr = get_json_data(item,item_name_path,flag)        
    client = ZhipuAI(api_key="49fa3d935e4341cbb49dd5ac4122c093.Qtb2rFUx1WQxy9bA") 
    item_embeddings = encode_text_with_bert(item_attr,client, batch_size=64)

    print(f"嵌入向量形状：{item_embeddings.shape}")
    np.save(file_path+'zhipu_item_embeddings.npy', item_embeddings)
    with open(file_path+'zhipu_item_title.txt', "w", encoding="utf-8") as f:
        for string in item_title_attr:
            f.write(string + "\n")

def encode_text_with_bert(text_list, model, batch_size):
    
    embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i + batch_size]
        response = model.embeddings.create(
            model="embedding-3", #填写需要调用的模型编码
            dimensions = 512,
            input=batch,

        )
        for embed in response.data:
            embeddings.append(embed.embedding)
    return np.array(embeddings)

#===========================================================================================================
data_path = './Data'
data_name_s = 'sport'
data_name_t = 'cell'
save_path_s = './Data/data/' + data_name_s+'_'+data_name_t+'/'
save_path_t = './Data/data/' + data_name_t+'_'+data_name_s+'/'
print(f'save_path_s: {save_path_s}')
print(f'save_path_t: {save_path_t}')
if not os.path.exists(save_path_s):
    os.makedirs(save_path_s)
if not os.path.exists(save_path_t):
    os.makedirs(save_path_t)
filepath1 = 'Data/Amazon/ratings_Sports_and_Outdoors.csv'
filepath2 = 'Data/Amazon/ratings_Cell_Phones_and_Accessories.csv'
save_file1 = save_path_s + 'new_reindex.txt'
save_file2 = save_path_t + 'new_reindex.txt'
f_path= save_path_t+'%s_%s_data_info.txt'%(data_name_s,data_name_t)
f = open(f_path,'w+')
u_num,i_num,r_num,user_unique,data = filter_data(filepath1)
u_num2,i_num2,r_num2,user_unique2,data2 = filter_data(filepath2)

c_n, common_user =get_common_user(user_unique,user_unique2)
pprint('raw_data1 info : %d %d %d'%(u_num,i_num,r_num),f)
pprint('raw_data2 info : %d %d %d'%(u_num2,i_num2,r_num2),f)
pprint('common user num %d'%c_n,f)
new_data_1,new_data_2 = get_common_data(data,data2,common_user)
new_data_1,new_data_2 =filter_user(new_data_1,new_data_2)
u,i ,r= get_unique_lenth(new_data_1)
u2,i2 ,r2= get_unique_lenth(new_data_2)
pprint('after common_data1 info : %d %d %d %.6f'%(u,i,r,r/(u*i)),f)
pprint('after common_data2 info : %d %d %d %.6f'%(u2,i2,r2,r2/(u2*i2)),f)
data1,dic_u,dic_item = reindex_data(new_data_1)
data2,dic_u2,dic_item2 = reindex_data(new_data_2,dic_u)

item_name_path_s = 'Data/AmazonJson/meta_Sports_and_Outdoors.json'
item_name_path_t = 'Data/AmazonJson/meta_Cell_Phones_and_Accessories.json'
get_embedding(dic_item,save_path_s,item_name_path_s,flag='source')
get_embedding(dic_item2,save_path_t,item_name_path_t,flag='target')

min1 = get_min_group_size(new_data_1)
min2 = get_min_group_size(new_data_2)
assert dic_u == dic_u2,'user_dic not same'
pprint('min user group size is %d %d'%(min1,min2),f)
pprint('filter way: user>=%d,item>=%d'%(5,10),f)
write_to_txt(data1,save_file1)
write_to_txt(data2,save_file2)
pprint('write data finished!',f)
print("*"*100)



